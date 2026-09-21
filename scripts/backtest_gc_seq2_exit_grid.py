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
HOLDS = [5, 7, 10, 12, 15, 20]


def load_archive():
    paths = sorted(BATCH_DIR.glob("batch_*.parquet"))
    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        cols = [c for c in ["Code","Date","O","H","L","C","AdjO","AdjH","AdjL","AdjC","ArchiveMarket"] if c in df.columns]
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
    g = g.dropna(subset=["O","C"]).sort_values("Date").reset_index(drop=True).copy()
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


def metrics(vals):
    s = pd.Series(vals, dtype=float).dropna()
    if s.empty:
        return {"n": 0}
    pos = s[s > 0]
    neg = s[s < 0]
    gp = pos.sum()
    gl = -neg.sum()
    pf = gp / gl if gl > 0 else None
    return {
        "n": int(len(s)),
        "win_rate_pct": round(float((s > 0).mean() * 100), 2),
        "avg_return_pct": round(float(s.mean()), 3),
        "median_return_pct": round(float(s.median()), 3),
        "profit_factor": round(float(pf), 3) if pf is not None else None,
    }


def main():
    df = load_archive()
    records = []

    for code, group in df.groupby("Code", sort=True):
        g = prepare(group)
        max_hold = max(HOLDS)
        for i in range(max(MA_LONG + SLOPE_LOOKBACK, ROLLING_HIGH), len(g)-max_hold-2):
            if not bool(g["GC"].iloc[i]):
                continue

            rising = g["MA25_SLOPE5_PCT"].iloc[i] > SLOPE_RISING_PCT
            dd = g["DD60_PCT"].iloc[i]
            in_dd = pd.notna(dd) and TARGET_DD_MIN <= dd < TARGET_DD_MAX
            seq2 = int(g["GC_SEQ"].iloc[i]) == 2
            if not (rising and in_dd and seq2):
                continue

            entry_i = i + 1
            entry = float(g["O"].iloc[entry_i])
            if not np.isfinite(entry) or entry <= 0:
                continue

            rec = {
                "code": str(code),
                "signal_date": str(pd.Timestamp(g["Date"].iloc[i]).date()),
                "entry_date": str(pd.Timestamp(g["Date"].iloc[entry_i]).date()),
            }
            for h in HOLDS:
                exit_i = entry_i + h
                if exit_i < len(g):
                    exitp = float(g["O"].iloc[exit_i])
                    rec[f"ret_{h}d"] = (exitp / entry - 1) * 100 if np.isfinite(exitp) else np.nan
                else:
                    rec[f"ret_{h}d"] = np.nan
            records.append(rec)

    tdf = pd.DataFrame(records)
    tdf["signal_date_dt"] = pd.to_datetime(tdf["signal_date"])

    archive_start = pd.Timestamp(df["Date"].min()).normalize()
    split = archive_start + pd.DateOffset(years=5)

    summary = {
        "strategy": "GC rising + DD15-20 + exact 2nd GC",
        "entry": "next open after GC",
        "tested_fixed_exits": HOLDS,
        "archive_start": str(pd.Timestamp(df["Date"].min()).date()),
        "archive_end": str(pd.Timestamp(df["Date"].max()).date()),
        "split_date": str(split.date()),
        "overall_10y": {},
        "older_5y": {},
        "recent_5y": {},
        "yearly": {},
    }

    for h in HOLDS:
        col = f"ret_{h}d"
        summary["overall_10y"][str(h)] = metrics(tdf[col])
        summary["older_5y"][str(h)] = metrics(tdf.loc[tdf["signal_date_dt"] < split, col])
        summary["recent_5y"][str(h)] = metrics(tdf.loc[tdf["signal_date_dt"] >= split, col])

    for year, part in tdf.groupby(tdf["signal_date_dt"].dt.year):
        summary["yearly"][str(int(year))] = {
            str(h): metrics(part[f"ret_{h}d"]) for h in HOLDS
        }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tdf.drop(columns=["signal_date_dt"]).to_csv(RESULT_DIR / "gc_seq2_exit_grid_trades.csv", index=False)
    (RESULT_DIR / "gc_seq2_exit_grid_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
