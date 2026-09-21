import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
BATCH_DIR = DATA_DIR / "batches"
RESULT_DIR = Path("results")

MA_SHORT = 5
MA_LONG = 25
SLOPE_LOOKBACK = 5
SLOPE_RISING_PCT = 0.25
ROLLING_HIGH = 60
EPISODE_START_DD = -15.0
EPISODE_RESET_DD = -5.0
TARGET_DD_MIN = -20.0
TARGET_DD_MAX = -15.0

HOLD_DAYS = 10
STOP_PCT = -3.0


def load_archive():
    paths = sorted(BATCH_DIR.glob("batch_*.parquet"))
    if not paths:
        raise RuntimeError("No archive batches found.")

    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        cols = [c for c in [
            "Code","Date","O","H","L","C",
            "AdjO","AdjH","AdjL","AdjC","ArchiveMarket"
        ] if c in df.columns]
        frames.append(df[cols].copy())

    df = pd.concat(frames, ignore_index=True)

    if "ArchiveMarket" in df.columns:
        df = df[df["ArchiveMarket"] == "プライム"].copy()

    for raw, adj in [("O","AdjO"),("H","AdjH"),("L","AdjL"),("C","AdjC")]:
        if adj in df.columns:
            df[raw] = df[adj].where(df[adj].notna(), df[raw])

    for c in ["O","H","L","C"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Code"] = df["Code"].astype(str)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.drop_duplicates(["Code","Date"], keep="last")
    return df.sort_values(["Code","Date"]).reset_index(drop=True)


def prepare(g):
    g = g.dropna(subset=["O","H","L","C"]).sort_values("Date").reset_index(drop=True).copy()
    g["MA5"] = g["C"].rolling(MA_SHORT).mean()
    g["MA25"] = g["C"].rolling(MA_LONG).mean()
    g["MA25_SLOPE5_PCT"] = (g["MA25"] / g["MA25"].shift(SLOPE_LOOKBACK) - 1) * 100
    g["HIGH60"] = g["C"].rolling(ROLLING_HIGH, min_periods=ROLLING_HIGH).max()
    g["DD60_PCT"] = (g["C"] / g["HIGH60"] - 1) * 100
    g["GC"] = (g["MA5"].shift(1) <= g["MA25"].shift(1)) & (g["MA5"] > g["MA25"])

    active = False
    seq = 0
    seqs = []
    for i in range(len(g)):
        dd = g["DD60_PCT"].iloc[i]
        if pd.isna(dd):
            seqs.append(0)
            continue

        if not active and dd <= EPISODE_START_DD:
            active = True
            seq = 0
        elif active and dd > EPISODE_RESET_DD:
            active = False
            seq = 0

        if active and bool(g["GC"].iloc[i]):
            seq += 1

        seqs.append(seq if active else 0)

    g["GC_SEQ"] = seqs
    return g


def is_target_signal(g, i):
    if not bool(g["GC"].iloc[i]):
        return False
    slope = g["MA25_SLOPE5_PCT"].iloc[i]
    dd = g["DD60_PCT"].iloc[i]
    if pd.isna(slope) or pd.isna(dd):
        return False
    return (
        slope > SLOPE_RISING_PCT
        and TARGET_DD_MIN <= dd < TARGET_DD_MAX
        and int(g["GC_SEQ"].iloc[i]) == 2
    )


def is_gakutto(g, i):
    # Same broad definition used in the prior GC exit study:
    # MA5 was rising on the prior step, turns down today, and today's candle is bearish.
    if i < 2:
        return False
    vals = [g["MA5"].iloc[i-2], g["MA5"].iloc[i-1], g["MA5"].iloc[i]]
    if any(pd.isna(v) for v in vals):
        return False

    prior_rising = g["MA5"].iloc[i-1] > g["MA5"].iloc[i-2]
    turn_down = g["MA5"].iloc[i] < g["MA5"].iloc[i-1]
    bearish = g["C"].iloc[i] < g["O"].iloc[i]
    return prior_rising and turn_down and bearish


def make_result(g, signal_i, entry_i, exit_i, entry_price, exit_price, reason):
    return {
        "code": str(g["Code"].iloc[signal_i]),
        "signal_date": str(pd.Timestamp(g["Date"].iloc[signal_i]).date()),
        "entry_date": str(pd.Timestamp(g["Date"].iloc[entry_i]).date()),
        "exit_date": str(pd.Timestamp(g["Date"].iloc[exit_i]).date()),
        "return_pct": (exit_price / entry_price - 1) * 100,
        "hold_sessions_to_exit": int(exit_i - entry_i),
        "exit_reason": reason,
    }


def simulate_fixed10(g, signal_i):
    entry_i = signal_i + 1
    exit_i = entry_i + HOLD_DAYS
    if exit_i >= len(g):
        return None

    entry = float(g["O"].iloc[entry_i])
    exitp = float(g["O"].iloc[exit_i])
    if not (np.isfinite(entry) and np.isfinite(exitp) and entry > 0):
        return None
    return make_result(g, signal_i, entry_i, exit_i, entry, exitp, "fixed10")


def simulate_stop3(g, signal_i):
    entry_i = signal_i + 1
    fixed_exit_i = entry_i + HOLD_DAYS
    if fixed_exit_i >= len(g):
        return None

    entry = float(g["O"].iloc[entry_i])
    if not np.isfinite(entry) or entry <= 0:
        return None

    stop_price = entry * (1 + STOP_PCT / 100.0)

    for i in range(entry_i, fixed_exit_i):
        o = float(g["O"].iloc[i])
        l = float(g["L"].iloc[i])

        if o <= stop_price:
            return make_result(g, signal_i, entry_i, i, entry, o, "stop_gap")
        if l <= stop_price:
            return make_result(g, signal_i, entry_i, i, entry, stop_price, "stop_-3%")

    exitp = float(g["O"].iloc[fixed_exit_i])
    return make_result(g, signal_i, entry_i, fixed_exit_i, entry, exitp, "fixed10")


def simulate_gakutto(g, signal_i):
    entry_i = signal_i + 1
    fixed_exit_i = entry_i + HOLD_DAYS
    if fixed_exit_i >= len(g):
        return None

    entry = float(g["O"].iloc[entry_i])
    if not np.isfinite(entry) or entry <= 0:
        return None

    # Confirm gakutto at close; execute next open if it happens before fixed exit.
    for i in range(entry_i, fixed_exit_i):
        if is_gakutto(g, i):
            exit_i = i + 1
            if exit_i <= fixed_exit_i and exit_i < len(g):
                exitp = float(g["O"].iloc[exit_i])
                return make_result(
                    g, signal_i, entry_i, exit_i, entry, exitp, "gakutto_next_open"
                )

    exitp = float(g["O"].iloc[fixed_exit_i])
    return make_result(g, signal_i, entry_i, fixed_exit_i, entry, exitp, "fixed10")


def metrics(df):
    if df.empty:
        return {"n": 0}

    r = df["return_pct"].dropna()
    pos = r[r > 0]
    neg = r[r < 0]
    gp = pos.sum()
    gl = -neg.sum()
    pf = gp / gl if gl > 0 else None

    return {
        "n": int(len(r)),
        "win_rate_pct": round(float((r > 0).mean() * 100), 2),
        "avg_return_pct": round(float(r.mean()), 3),
        "median_return_pct": round(float(r.median()), 3),
        "profit_factor": round(float(pf), 3) if pf is not None else None,
        "avg_hold_sessions_to_exit": round(float(df["hold_sessions_to_exit"].mean()), 2),
    }


def summarize(df, split):
    older = df[df["signal_date_dt"] < split]
    recent = df[df["signal_date_dt"] >= split]

    yearly = {}
    for year, part in df.groupby(df["signal_date_dt"].dt.year):
        yearly[str(int(year))] = metrics(part)

    return {
        "overall_10y": metrics(df),
        "older_5y": metrics(older),
        "recent_5y": metrics(recent),
        "yearly": yearly,
        "exit_reasons": {
            str(k): int(v) for k, v in df["exit_reason"].value_counts().to_dict().items()
        },
    }


def main():
    df = load_archive()

    rows = {
        "fixed10": [],
        "stop3_then_fixed10": [],
        "gakutto_then_fixed10": [],
    }

    for code, group in df.groupby("Code", sort=True):
        g = prepare(group)
        for i in range(max(MA_LONG + SLOPE_LOOKBACK, ROLLING_HIGH), len(g) - HOLD_DAYS - 2):
            if not is_target_signal(g, i):
                continue

            a = simulate_fixed10(g, i)
            b = simulate_stop3(g, i)
            c = simulate_gakutto(g, i)

            if a: rows["fixed10"].append(a)
            if b: rows["stop3_then_fixed10"].append(b)
            if c: rows["gakutto_then_fixed10"].append(c)

    archive_start = pd.Timestamp(df["Date"].min()).normalize()
    split = archive_start + pd.DateOffset(years=5)

    summary = {
        "strategy": "GC rising + DD15-20 + exact 2nd GC",
        "entry": "next open after GC",
        "comparison": {
            "fixed10": "sell at open after 10 trading sessions",
            "stop3_then_fixed10": "-3% intraday stop (gap uses open), otherwise fixed 10-day exit",
            "gakutto_then_fixed10": "MA5 prior step rising -> turns down today + bearish candle, exit next open; otherwise fixed 10-day exit",
        },
        "archive_start": str(pd.Timestamp(df["Date"].min()).date()),
        "archive_end": str(pd.Timestamp(df["Date"].max()).date()),
        "split_date": str(split.date()),
        "variants": {},
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    all_trade_frames = []
    for name, recs in rows.items():
        tdf = pd.DataFrame(recs)
        if tdf.empty:
            summary["variants"][name] = {"overall_10y": {"n": 0}}
            continue

        tdf["signal_date_dt"] = pd.to_datetime(tdf["signal_date"])
        summary["variants"][name] = summarize(tdf, split)

        out = tdf.drop(columns=["signal_date_dt"]).copy()
        out["variant"] = name
        all_trade_frames.append(out)

    if all_trade_frames:
        pd.concat(all_trade_frames, ignore_index=True).to_csv(
            RESULT_DIR / "gc_seq2_exit_methods_trades.csv", index=False
        )

    (RESULT_DIR / "gc_seq2_exit_methods_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
