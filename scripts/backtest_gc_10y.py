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


def load_archive():
    paths = sorted(BATCH_DIR.glob("batch_*.parquet"))
    if not paths:
        raise RuntimeError("No archive batches found.")

    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        cols = [c for c in ["Code","Date","O","H","L","C","AdjO","AdjH","AdjL","AdjC","ArchiveMarket"] if c in df.columns]
        frames.append(df[cols].copy())

    df = pd.concat(frames, ignore_index=True)

    # Match the prior GC study universe: Prime only.
    if "ArchiveMarket" in df.columns:
        df = df[df["ArchiveMarket"] == "プライム"].copy()

    for raw, adj in [("O","AdjO"),("H","AdjH"),("L","AdjL"),("C","AdjC")]:
        if adj in df.columns:
            if raw in df.columns:
                df[raw] = df[adj].where(df[adj].notna(), df[raw])
            else:
                df[raw] = df[adj]

    for c in ["O","H","L","C"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Code"] = df["Code"].astype(str)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.drop_duplicates(["Code","Date"], keep="last")
    return df.sort_values(["Code","Date"]).reset_index(drop=True)


def prepare(g):
    g = g.dropna(subset=["O","C"]).sort_values("Date").reset_index(drop=True).copy()
    g["MA5"] = g["C"].rolling(MA_SHORT).mean()
    g["MA25"] = g["C"].rolling(MA_LONG).mean()
    g["MA25_SLOPE5_PCT"] = (g["MA25"] / g["MA25"].shift(SLOPE_LOOKBACK) - 1) * 100
    g["HIGH60"] = g["C"].rolling(ROLLING_HIGH, min_periods=ROLLING_HIGH).max()
    g["DD60_PCT"] = (g["C"] / g["HIGH60"] - 1) * 100
    g["GC"] = (g["MA5"].shift(1) <= g["MA25"].shift(1)) & (g["MA5"] > g["MA25"])

    # Deep-drop episode sequence: starts once drawdown <= -15%, remains active until > -5%.
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
            seqs.append(seq)
        else:
            seqs.append(seq if active else 0)

    g["GC_SEQ"] = seqs
    return g


def trade_from_signal(g, i):
    entry_i = i + 1
    exit_i = entry_i + HOLD_DAYS
    if exit_i >= len(g):
        return None
    entry = float(g["O"].iloc[entry_i])
    exitp = float(g["O"].iloc[exit_i])
    if not (np.isfinite(entry) and np.isfinite(exitp) and entry > 0):
        return None
    ret = (exitp / entry - 1) * 100
    return {
        "code": str(g["Code"].iloc[i]),
        "signal_date": str(pd.Timestamp(g["Date"].iloc[i]).date()),
        "entry_date": str(pd.Timestamp(g["Date"].iloc[entry_i]).date()),
        "exit_date": str(pd.Timestamp(g["Date"].iloc[exit_i]).date()),
        "return_pct": ret,
        "dd60_pct": float(g["DD60_PCT"].iloc[i]) if pd.notna(g["DD60_PCT"].iloc[i]) else None,
        "ma25_slope5_pct": float(g["MA25_SLOPE5_PCT"].iloc[i]) if pd.notna(g["MA25_SLOPE5_PCT"].iloc[i]) else None,
        "gc_seq": int(g["GC_SEQ"].iloc[i]),
    }


def metrics(df):
    if df.empty:
        return {"n": 0}
    r = df["return_pct"]
    pos = r[r > 0]
    neg = r[r < 0]
    gp = pos.sum()
    gl = -neg.sum()
    pf = gp / gl if gl > 0 else None
    return {
        "n": int(len(df)),
        "win_rate_pct": round(float((r > 0).mean() * 100), 2),
        "avg_return_pct": round(float(r.mean()), 3),
        "median_return_pct": round(float(r.median()), 3),
        "profit_factor": round(float(pf), 3) if pf is not None else None,
    }


def main():
    df = load_archive()
    rows = []

    for code, group in df.groupby("Code", sort=True):
        g = prepare(group)
        for i in range(max(MA_LONG + SLOPE_LOOKBACK, ROLLING_HIGH), len(g)-HOLD_DAYS-1):
            if bool(g["GC"].iloc[i]):
                t = trade_from_signal(g, i)
                if t:
                    rows.append(t)

    tdf = pd.DataFrame(rows)
    if tdf.empty:
        raise RuntimeError("No GC signals found.")

    tdf["signal_date_dt"] = pd.to_datetime(tdf["signal_date"])
    archive_start = pd.Timestamp(df["Date"].min()).normalize()
    split = archive_start + pd.DateOffset(years=5)

    rising = tdf["ma25_slope5_pct"] > SLOPE_RISING_PCT
    dd15_20 = (tdf["dd60_pct"] >= TARGET_DD_MIN) & (tdf["dd60_pct"] < TARGET_DD_MAX)

    variants = {
        "all_gc": tdf.index == tdf.index,
        "rising_dd15_20": rising & dd15_20,
        "rising_dd15_20_seq2": rising & dd15_20 & (tdf["gc_seq"] == 2),
        "rising_dd15_20_seq3": rising & dd15_20 & (tdf["gc_seq"] == 3),
    }

    summary = {
        "strategy": "GC_10Y_frozen_2026-09-21",
        "universe": "current Prime listings in saved archive",
        "archive_start": str(pd.Timestamp(df["Date"].min()).date()),
        "archive_end": str(pd.Timestamp(df["Date"].max()).date()),
        "split_date": str(split.date()),
        "entry": "MA5 previous <= MA25 and current MA5 > MA25; buy next open",
        "exit": f"sell at open after {HOLD_DAYS} trading sessions",
        "ma25_rising": f"5-day MA25 slope > {SLOPE_RISING_PCT}%",
        "drawdown": "close vs rolling 60-day high",
        "episode": "starts at <= -15% drawdown, resets when drawdown > -5%; GC sequence increments within episode",
        "variants": {},
        "note": "These conditions were frozen from the earlier 5-year GC exploration. Older half is the main robustness check. Current-listing universe implies survivorship bias."
    }

    for name, mask in variants.items():
        part = tdf[mask].copy()
        older = part[part["signal_date_dt"] < split]
        recent = part[part["signal_date_dt"] >= split]
        yearly = {
            str(int(y)): metrics(p)
            for y, p in part.groupby(part["signal_date_dt"].dt.year)
        }
        summary["variants"][name] = {
            "overall_10y": metrics(part),
            "older_5y": metrics(older),
            "recent_5y": metrics(recent),
            "yearly": yearly,
        }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tdf.drop(columns=["signal_date_dt"]).to_csv(RESULT_DIR / "gc_10y_trades.csv", index=False)
    (RESULT_DIR / "gc_10y_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
