import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
BATCH_DIR = DATA_DIR / "batches"
RESULT_DIR = Path("results")

MA_SHORT = 5
MA_LONG = 25
TURN_LOOKBACK = 2
DECLINE_LOOKBACK = 5
DECLINE_MIN_PCT = -5.5
DECLINE_MAX_PCT = -2.0
MA5_VS_MA25_MIN_PCT = -5.0
MA5_VS_MA25_MAX_PCT = 0.0
CLOSE_VS_MA5_MAX_PCT = 4.0
BULL_MIN_PCT = 1.5
BULL_MAX_PCT = 3.5
VOLUME_RATIO_MIN = 1.25

HOLD_DAYS = 10


def load_archive():
    paths = sorted(BATCH_DIR.glob("batch_*.parquet"))
    if not paths:
        raise RuntimeError("No archive batches found")
    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        cols = [c for c in [
            "Code","Date","O","H","L","C","Vo",
            "AdjO","AdjH","AdjL","AdjC","AdjVo","ArchiveMarket"
        ] if c in df.columns]
        frames.append(df[cols].copy())
    df = pd.concat(frames, ignore_index=True)
    if "ArchiveMarket" in df.columns:
        df = df[df["ArchiveMarket"] == "プライム"].copy()

    for raw, adj in [("O","AdjO"),("H","AdjH"),("L","AdjL"),("C","AdjC"),("Vo","AdjVo")]:
        if adj in df.columns:
            if raw in df.columns:
                df[raw] = df[adj].where(df[adj].notna(), df[raw])
            else:
                df[raw] = df[adj]

    for c in ["O","H","L","C","Vo"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Code"] = df["Code"].astype(str)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.drop_duplicates(["Code","Date"], keep="last")
    return df.sort_values(["Code","Date"]).reset_index(drop=True)


def prepare(g):
    g = g.dropna(subset=["O","H","L","C"]).sort_values("Date").reset_index(drop=True).copy()
    g["MA5"] = g["C"].rolling(MA_SHORT).mean()
    g["MA25"] = g["C"].rolling(MA_LONG).mean()
    g["MA25_SLOPE5_PCT"] = (g["MA25"] / g["MA25"].shift(5) - 1) * 100
    g["HIGH20"] = g["C"].rolling(20).max()
    g["HIGH60"] = g["C"].rolling(60).max()
    g["DD20_PCT"] = (g["C"] / g["HIGH20"] - 1) * 100
    g["DD60_PCT"] = (g["C"] / g["HIGH60"] - 1) * 100
    g["TURNOVER"] = g["C"] * g["Vo"]
    g["AVG_TURNOVER20"] = g["TURNOVER"].rolling(20).mean()
    g["RET5"] = (g["C"] / g["C"].shift(5) - 1) * 100
    g["RET20"] = (g["C"] / g["C"].shift(20) - 1) * 100
    tr1 = g["H"] - g["L"]
    tr2 = (g["H"] - g["C"].shift(1)).abs()
    tr3 = (g["L"] - g["C"].shift(1)).abs()
    g["TR"] = pd.concat([tr1,tr2,tr3], axis=1).max(axis=1)
    g["ATR14_PCT"] = g["TR"].rolling(14).mean() / g["C"] * 100
    return g


def features_if_signal(g, i):
    if i < max(MA_LONG, DECLINE_LOOKBACK + 1, 20):
        return None
    r = g.iloc[i]
    if pd.isna(r["MA5"]) or pd.isna(r["MA25"]) or not r["O"] > 0:
        return None

    bull = (r["C"] / r["O"] - 1) * 100
    if not (BULL_MIN_PCT <= bull <= BULL_MAX_PCT):
        return None

    gap = (r["MA5"] / r["MA25"] - 1) * 100
    if not (MA5_VS_MA25_MIN_PCT <= gap <= MA5_VS_MA25_MAX_PCT):
        return None

    base = i - 1
    past = base - DECLINE_LOOKBACK
    if past < 0 or pd.isna(g["MA5"].iloc[past]) or not g["MA5"].iloc[past] > 0:
        return None
    decline = (g["MA5"].iloc[base] / g["MA5"].iloc[past] - 1) * 100
    if not (DECLINE_MIN_PCT <= decline <= DECLINE_MAX_PCT):
        return None

    for j in range(i - TURN_LOOKBACK, i):
        if g["MA5"].iloc[j] > g["MA5"].iloc[j - 1]:
            return None
    if not g["MA5"].iloc[i] > g["MA5"].iloc[i - 1]:
        return None

    close_ma5 = (r["C"] / r["MA5"] - 1) * 100
    if close_ma5 > CLOSE_VS_MA5_MAX_PCT:
        return None

    pv = g["Vo"].iloc[i-1]
    tv = r["Vo"]
    if pd.isna(pv) or pd.isna(tv) or pv <= 0:
        return None
    vr = float(tv) / float(pv)
    if vr < VOLUME_RATIO_MIN:
        return None

    return {
        "bull_pct": float(bull),
        "ma5_decline5_pct": float(decline),
        "ma5_vs_ma25_pct": float(gap),
        "close_vs_ma5_pct": float(close_ma5),
        "volume_ratio": float(vr),
        "ma25_slope5_pct": float(r["MA25_SLOPE5_PCT"]) if pd.notna(r["MA25_SLOPE5_PCT"]) else np.nan,
        "dd20_pct": float(r["DD20_PCT"]) if pd.notna(r["DD20_PCT"]) else np.nan,
        "dd60_pct": float(r["DD60_PCT"]) if pd.notna(r["DD60_PCT"]) else np.nan,
        "ret5_pct": float(r["RET5"]) if pd.notna(r["RET5"]) else np.nan,
        "ret20_pct": float(r["RET20"]) if pd.notna(r["RET20"]) else np.nan,
        "atr14_pct": float(r["ATR14_PCT"]) if pd.notna(r["ATR14_PCT"]) else np.nan,
        "avg_turnover20": float(r["AVG_TURNOVER20"]) if pd.notna(r["AVG_TURNOVER20"]) else np.nan,
    }


def metrics(part):
    r = part["ret10"].dropna()
    if r.empty:
        return {"n":0}
    gp = r[r>0].sum()
    gl = -r[r<0].sum()
    return {
        "n": int(len(r)),
        "win_rate_pct": round(float((r>0).mean()*100),2),
        "avg_ret10_pct": round(float(r.mean()),3),
        "median_ret10_pct": round(float(r.median()),3),
        "p10_plus_pct": round(float((r>=10).mean()*100),2),
        "p5_plus_pct": round(float((r>=5).mean()*100),2),
        "profit_factor": round(float(gp/gl),3) if gl>0 else None,
    }


def main():
    df = load_archive()
    recs = []
    for code, group in df.groupby("Code", sort=True):
        g = prepare(group)
        for i in range(25, len(g)-HOLD_DAYS-1):
            f = features_if_signal(g, i)
            if not f:
                continue
            entry_i = i+1
            exit_i = entry_i+HOLD_DAYS
            entry = float(g["O"].iloc[entry_i])
            exitp = float(g["O"].iloc[exit_i])
            if not (np.isfinite(entry) and np.isfinite(exitp) and entry>0):
                continue
            rec = {
                "code": str(code),
                "signal_date": str(pd.Timestamp(g["Date"].iloc[i]).date()),
                "entry_date": str(pd.Timestamp(g["Date"].iloc[entry_i]).date()),
                "ret10": (exitp/entry-1)*100,
                **f
            }
            recs.append(rec)

    t = pd.DataFrame(recs)
    t["signal_date_dt"] = pd.to_datetime(t["signal_date"])
    archive_start = pd.Timestamp(df["Date"].min()).normalize()
    split = archive_start + pd.DateOffset(years=5)

    feature_cols = [
        "bull_pct","ma5_decline5_pct","ma5_vs_ma25_pct","close_vs_ma5_pct",
        "volume_ratio","ma25_slope5_pct","dd20_pct","dd60_pct","ret5_pct",
        "ret20_pct","atr14_pct","avg_turnover20"
    ]

    def analyze_block(part):
        out = {"base": metrics(part), "features": {}}
        for c in feature_cols:
            x = part[[c,"ret10"]].dropna().copy()
            if len(x) < 50:
                continue
            try:
                x["bin"] = pd.qcut(x[c], 4, duplicates="drop")
            except Exception:
                continue
            bins = []
            for b, p in x.groupby("bin", observed=True):
                row = metrics(p)
                row["range"] = str(b)
                row["feature_median"] = round(float(p[c].median()),4)
                bins.append(row)
            out["features"][c] = bins
        return out

    # Focused diagnostics for the refined Kuitto subset.
    refined = t[(t["atr14_pct"] >= 3.4) & (t["dd20_pct"] <= -5.5)].copy()
    refined["year"] = refined["signal_date_dt"].dt.year
    compare_years = [2018, 2019, 2020, 2023, 2024, 2025, 2026]
    feature_compare = {}
    for y in compare_years:
        p = refined[refined["year"] == y]
        if p.empty:
            continue
        fd = {}
        for c in feature_cols:
            x = p[c].dropna()
            if len(x):
                fd[c] = {
                    "n": int(len(x)),
                    "mean": round(float(x.mean()), 4),
                    "median": round(float(x.median()), 4),
                    "q25": round(float(x.quantile(0.25)), 4),
                    "q75": round(float(x.quantile(0.75)), 4),
                }
        feature_compare[str(y)] = {
            "performance": metrics(p),
            "features": fd,
        }

    bad = refined[refined["year"].isin([2018, 2025])].copy()
    good = refined[refined["year"].isin([2019, 2020, 2023, 2024, 2026])].copy()
    bad_vs_good = {}
    for c in feature_cols:
        xb = bad[c].dropna(); xg = good[c].dropna()
        if len(xb) and len(xg):
            bad_vs_good[c] = {
                "bad_median": round(float(xb.median()), 4),
                "good_median": round(float(xg.median()), 4),
                "bad_mean": round(float(xb.mean()), 4),
                "good_mean": round(float(xg.mean()), 4),
            }

    summary = {
        "strategy": "kuitto_pullback_auto frozen signal, 10-day next-open return",
        "goal": "Find signal-day features associated with large winners without using post-entry information.",
        "labels": {
            "big_winner_5": "ret10 >= +5%",
            "big_winner_10": "ret10 >= +10%"
        },
        "overall_10y": analyze_block(t),
        "older_5y": analyze_block(t[t["signal_date_dt"] < split]),
        "recent_5y": analyze_block(t[t["signal_date_dt"] >= split]),
        "refined_rule_diagnostics": {"rule":"ATR14>=3.4 and DD20<=-5.5","year_compare":feature_compare,"bad_2018_2025_vs_good":bad_vs_good},
        "note": "Exploratory diagnostics only; any promising filter should be frozen and re-tested out of sample before production use."
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    t.drop(columns=["signal_date_dt"]).to_csv(RESULT_DIR/"kuitto_runner_features_trades.csv", index=False)
    (RESULT_DIR/"kuitto_runner_features_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
