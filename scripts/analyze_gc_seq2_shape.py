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
        raise RuntimeError("No archive batches found")

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
    if i < 2:
        return False
    vals = [g["MA5"].iloc[i-2], g["MA5"].iloc[i-1], g["MA5"].iloc[i]]
    if any(pd.isna(v) for v in vals):
        return False
    return (
        g["MA5"].iloc[i-1] > g["MA5"].iloc[i-2]
        and g["MA5"].iloc[i] < g["MA5"].iloc[i-1]
        and g["C"].iloc[i] < g["O"].iloc[i]
    )


def qstats(s):
    s = pd.Series(s, dtype=float).dropna()
    if s.empty:
        return {}
    return {
        "mean": round(float(s.mean()), 3),
        "median": round(float(s.median()), 3),
        "p25": round(float(s.quantile(0.25)), 3),
        "p75": round(float(s.quantile(0.75)), 3),
        "positive_pct": round(float((s > 0).mean() * 100), 2),
    }


def main():
    df = load_archive()
    trades = []

    for code, group in df.groupby("Code", sort=True):
        g = prepare(group)
        for i in range(max(MA_LONG + SLOPE_LOOKBACK, ROLLING_HIGH), len(g) - HOLD_DAYS - 2):
            if not is_target_signal(g, i):
                continue

            entry_i = i + 1
            exit_i = entry_i + HOLD_DAYS
            entry = float(g["O"].iloc[entry_i])
            exit_open = float(g["O"].iloc[exit_i])
            if not (np.isfinite(entry) and np.isfinite(exit_open) and entry > 0):
                continue

            rec = {
                "code": str(code),
                "signal_date": str(pd.Timestamp(g["Date"].iloc[i]).date()),
                "entry_date": str(pd.Timestamp(g["Date"].iloc[entry_i]).date()),
                "final_10d_open_ret": (exit_open / entry - 1) * 100,
            }

            # Path: close of held sessions 1..10 relative to entry open.
            for d in range(1, HOLD_DAYS + 1):
                idx = entry_i + d - 1
                rec[f"close_ret_d{d}"] = (float(g["C"].iloc[idx]) / entry - 1) * 100

            lows = []
            highs = []
            for d in range(1, HOLD_DAYS + 1):
                idx = entry_i + d - 1
                lows.append((float(g["L"].iloc[idx]) / entry - 1) * 100)
                highs.append((float(g["H"].iloc[idx]) / entry - 1) * 100)

            rec["mae_pct"] = min(lows)
            rec["mfe_pct"] = max(highs)
            rec["mae_day"] = int(np.argmin(lows) + 1)
            rec["mfe_day"] = int(np.argmax(highs) + 1)

            for level in [-3.0, -5.0]:
                hit = any(x <= level for x in lows)
                rec[f"hit_m{int(abs(level))}"] = bool(hit)

            for level in [5.0, 7.0, 10.0]:
                hit = any(x >= level for x in highs)
                rec[f"hit_p{int(level)}"] = bool(hit)

            # First gakutto exit opportunity and its return.
            rec["gakutto_hit"] = False
            rec["gakutto_exit_ret"] = np.nan
            rec["gakutto_exit_day"] = np.nan
            for j in range(entry_i, exit_i):
                if is_gakutto(g, j):
                    nxt = j + 1
                    if nxt <= exit_i:
                        rec["gakutto_hit"] = True
                        rec["gakutto_exit_ret"] = (float(g["O"].iloc[nxt]) / entry - 1) * 100
                        rec["gakutto_exit_day"] = int(nxt - entry_i)
                        break

            trades.append(rec)

    tdf = pd.DataFrame(trades)
    if tdf.empty:
        raise RuntimeError("No target trades")

    tdf["signal_date_dt"] = pd.to_datetime(tdf["signal_date"])
    archive_start = pd.Timestamp(df["Date"].min()).normalize()
    split = archive_start + pd.DateOffset(years=5)

    def block(part):
        out = {
            "n": int(len(part)),
            "path_by_day": {},
            "mae": qstats(part["mae_pct"]),
            "mfe": qstats(part["mfe_pct"]),
            "mae_day_distribution": {
                str(int(k)): int(v) for k, v in part["mae_day"].value_counts().sort_index().to_dict().items()
            },
            "mfe_day_distribution": {
                str(int(k)): int(v) for k, v in part["mfe_day"].value_counts().sort_index().to_dict().items()
            },
        }

        for d in range(1, HOLD_DAYS + 1):
            out["path_by_day"][str(d)] = qstats(part[f"close_ret_d{d}"])

        for level in [3,5]:
            mask = part[f"hit_m{level}"]
            subset = part[mask]
            out[f"hit_minus_{level}"] = {
                "count": int(mask.sum()),
                "pct": round(float(mask.mean()*100), 2),
                "final10_positive_pct": round(float((subset["final_10d_open_ret"] > 0).mean()*100), 2) if len(subset) else None,
                "final10_avg_ret": round(float(subset["final_10d_open_ret"].mean()), 3) if len(subset) else None,
                "final10_median_ret": round(float(subset["final_10d_open_ret"].median()), 3) if len(subset) else None,
            }

        for level in [5,7,10]:
            mask = part[f"hit_p{level}"]
            subset = part[mask]
            out[f"hit_plus_{level}"] = {
                "count": int(mask.sum()),
                "pct": round(float(mask.mean()*100), 2),
                "final10_avg_ret": round(float(subset["final_10d_open_ret"].mean()), 3) if len(subset) else None,
                "final10_median_ret": round(float(subset["final_10d_open_ret"].median()), 3) if len(subset) else None,
                "final10_below_threshold_pct": round(float((subset["final_10d_open_ret"] < level).mean()*100), 2) if len(subset) else None,
            }

        g = part[part["gakutto_hit"]].copy()
        if len(g):
            diff = g["final_10d_open_ret"] - g["gakutto_exit_ret"]
            out["gakutto_counterfactual"] = {
                "count": int(len(g)),
                "pct": round(float(len(g)/len(part)*100), 2),
                "gakutto_exit_avg_ret": round(float(g["gakutto_exit_ret"].mean()), 3),
                "final10_avg_ret_same_trades": round(float(g["final_10d_open_ret"].mean()), 3),
                "final10_minus_gakutto_avg": round(float(diff.mean()), 3),
                "final10_better_pct": round(float((diff > 0).mean()*100), 2),
                "avg_gakutto_exit_day": round(float(g["gakutto_exit_day"].mean()), 2),
            }
        else:
            out["gakutto_counterfactual"] = {"count": 0}

        return out

    summary = {
        "strategy": "GC rising + DD15-20 + exact 2nd GC",
        "entry": "next open after GC",
        "shape_window": "10 held sessions; fixed strategy exit is open after 10 sessions",
        "overall_10y": block(tdf),
        "older_5y": block(tdf[tdf["signal_date_dt"] < split]),
        "recent_5y": block(tdf[tdf["signal_date_dt"] >= split]),
        "note": "Shape diagnostics only; no new selection thresholds are being optimized here."
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tdf.drop(columns=["signal_date_dt"]).to_csv(RESULT_DIR / "gc_seq2_shape_trades.csv", index=False)
    (RESULT_DIR / "gc_seq2_shape_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
