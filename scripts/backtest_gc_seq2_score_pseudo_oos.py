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
        d = pd.read_parquet(p)
        cols = [c for c in [
            "Code","Date","O","H","L","C","Vo",
            "AdjO","AdjH","AdjL","AdjC","AdjVo","ArchiveMarket"
        ] if c in d.columns]
        frames.append(d[cols].copy())
    df = pd.concat(frames, ignore_index=True)
    if "ArchiveMarket" in df.columns:
        df = df[df["ArchiveMarket"] == "プライム"].copy()
    for raw, adj in [("O","AdjO"),("H","AdjH"),("L","AdjL"),("C","AdjC"),("Vo","AdjVo")]:
        if adj in df.columns:
            df[raw] = df[adj].where(df[adj].notna(), df.get(raw))
    for c in ["O","H","L","C","Vo"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Code"] = df["Code"].astype(str)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.drop_duplicates(["Code","Date"], keep="last").sort_values(["Code","Date"]).reset_index(drop=True)


def prepare(g):
    g = g.dropna(subset=["O","H","L","C"]).sort_values("Date").reset_index(drop=True).copy()
    g["MA5"] = g["C"].rolling(MA_SHORT).mean()
    g["MA25"] = g["C"].rolling(MA_LONG).mean()
    g["MA25_SLOPE5_PCT"] = (g["MA25"] / g["MA25"].shift(SLOPE_LOOKBACK) - 1) * 100
    g["HIGH60"] = g["C"].rolling(ROLLING_HIGH, min_periods=ROLLING_HIGH).max()
    g["DD60_PCT"] = (g["C"] / g["HIGH60"] - 1) * 100
    g["GC_GAP_PCT"] = (g["MA5"] / g["MA25"] - 1) * 100
    prev_close = g["C"].shift(1)
    tr = pd.concat([
        g["H"] - g["L"],
        (g["H"] - prev_close).abs(),
        (g["L"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    g["ATR14_PCT"] = tr.rolling(14).mean() / g["C"] * 100
    g["TURNOVER"] = g["C"] * g["Vo"]
    g["AVG_TURNOVER20"] = g["TURNOVER"].rolling(20).mean()
    g["VOL_RATIO20"] = g["Vo"] / g["Vo"].rolling(20).mean()
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


def is_target(g, i):
    if not bool(g["GC"].iloc[i]):
        return False
    slope = g["MA25_SLOPE5_PCT"].iloc[i]
    dd = g["DD60_PCT"].iloc[i]
    if pd.isna(slope) or pd.isna(dd):
        return False
    return slope > SLOPE_RISING_PCT and TARGET_DD_MIN <= dd < TARGET_DD_MAX and int(g["GC_SEQ"].iloc[i]) == 2


def metrics(part):
    r = pd.Series(part["ret10"], dtype=float).dropna()
    if r.empty:
        return {"n": 0}
    gain = r[r > 0].sum()
    loss = -r[r < 0].sum()
    return {
        "n": int(len(r)),
        "win_rate_pct": round(float((r > 0).mean() * 100), 2),
        "avg_ret10_pct": round(float(r.mean()), 3),
        "median_ret10_pct": round(float(r.median()), 3),
        "p5_plus_pct": round(float((r >= 5).mean() * 100), 2),
        "loss5_or_worse_pct": round(float((r <= -5).mean() * 100), 2),
        "profit_factor": round(float(gain / loss), 3) if loss > 0 else None,
    }


def quartile_report(train, feature):
    s = train[[feature, "ret10"]].dropna().copy()
    q25, q75 = s[feature].quantile([0.25, 0.75])
    lo = s[s[feature] <= q25]
    hi = s[s[feature] >= q75]
    return float(q25), float(q75), metrics(lo), metrics(hi)


def main():
    df = load_archive()
    rows = []
    for code, g0 in df.groupby("Code", sort=True):
        g = prepare(g0)
        for i in range(max(MA_LONG + SLOPE_LOOKBACK, ROLLING_HIGH), len(g) - HOLD_DAYS - 2):
            if not is_target(g, i):
                continue
            ei = i + 1
            xi = ei + HOLD_DAYS
            entry = float(g["O"].iloc[ei])
            exitp = float(g["O"].iloc[xi])
            if not (np.isfinite(entry) and np.isfinite(exitp) and entry > 0):
                continue
            rows.append({
                "code": str(code),
                "signal_date": pd.Timestamp(g["Date"].iloc[i]),
                "entry_date": pd.Timestamp(g["Date"].iloc[ei]),
                "ret10": (exitp / entry - 1) * 100,
                "atr14_pct": g["ATR14_PCT"].iloc[i],
                "dd60_pct": g["DD60_PCT"].iloc[i],
                "ma25_slope5_pct": g["MA25_SLOPE5_PCT"].iloc[i],
                "avg_turnover20": g["AVG_TURNOVER20"].iloc[i],
                "vol_ratio20": g["VOL_RATIO20"].iloc[i],
                "gc_gap_pct": g["GC_GAP_PCT"].iloc[i],
            })

    t = pd.DataFrame(rows)
    if t.empty:
        raise RuntimeError("No GC seq2 trades found")

    split = pd.Timestamp(df["Date"].min()).normalize() + pd.DateOffset(years=5)
    train = t[t["signal_date"] < split].copy()
    test = t[t["signal_date"] >= split].copy()

    # Four pre-declared dimensions for a 0-4 score.
    # Direction and quartile threshold are learned from older 5y only.
    features = [
        "atr14_pct",
        "dd60_pct",
        "ma25_slope5_pct",
        "avg_turnover20",
    ]
    learned = {}
    for f in features:
        q25, q75, lo_m, hi_m = quartile_report(train, f)
        lo_avg = lo_m.get("avg_ret10_pct", -999)
        hi_avg = hi_m.get("avg_ret10_pct", -999)
        if hi_avg >= lo_avg:
            direction = "high"
            threshold = q75
        else:
            direction = "low"
            threshold = q25
        learned[f] = {
            "direction": direction,
            "threshold": threshold,
            "train_low_quartile": lo_m,
            "train_high_quartile": hi_m,
        }

    def add_score(part):
        part = part.copy()
        score = pd.Series(0, index=part.index, dtype=int)
        for f, rule in learned.items():
            if rule["direction"] == "high":
                hit = part[f] >= rule["threshold"]
            else:
                hit = part[f] <= rule["threshold"]
            score += hit.fillna(False).astype(int)
        part["gc_runner_score"] = score
        return part

    train_s = add_score(train)
    test_s = add_score(test)
    all_s = add_score(t)

    def score_block(part):
        return {str(score): metrics(group) for score, group in part.groupby("gc_runner_score")}

    out = {
        "strategy": "GC rising + DD15-20 + exact 2nd GC",
        "purpose": "Build a 0-4 ranking score without excluding candidates.",
        "entry_exit": "next-open entry, fixed 10-session exit",
        "split_date": str(split.date()),
        "method": "Four pre-declared features. For each feature, direction and q25/q75 threshold are learned only from older 5y by comparing low vs high quartile average 10d return; thresholds are then frozen for recent 5y.",
        "overall": metrics(all_s),
        "older5_train": metrics(train_s),
        "recent5_test": metrics(test_s),
        "learned_rules": learned,
        "score_results": {
            "older5_train": score_block(train_s),
            "recent5_test": score_block(test_s),
            "overall": score_block(all_s),
        },
        "caveat": "Current-Prime survivor universe. Small sample (283 total historically), so score bands should be treated as ranking aids rather than hard filters.",
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "gc_seq2_score_pseudo_oos_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    all_s.assign(
        signal_date=all_s["signal_date"].dt.date.astype(str),
        entry_date=all_s["entry_date"].dt.date.astype(str),
    ).to_csv(RESULT_DIR / "gc_seq2_score_pseudo_oos_trades.csv", index=False)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
