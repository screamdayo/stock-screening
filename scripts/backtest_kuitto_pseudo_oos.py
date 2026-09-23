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
    g["HIGH20"] = g["C"].rolling(20).max()
    g["DD20_PCT"] = (g["C"] / g["HIGH20"] - 1) * 100
    g["TURNOVER"] = g["C"] * g["Vo"]
    g["AVG_TURNOVER20"] = g["TURNOVER"].rolling(20).mean()
    tr = pd.concat([
        g["H"] - g["L"],
        (g["H"] - g["C"].shift(1)).abs(),
        (g["L"] - g["C"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    g["ATR14_PCT"] = tr.rolling(14).mean() / g["C"] * 100
    return g


def is_kuitto(g, i):
    if i < max(MA_LONG, DECLINE_LOOKBACK + 1, 20):
        return False
    r = g.iloc[i]
    if any(pd.isna(r[x]) for x in ["MA5","MA25","O","C","Vo"]) or r["O"] <= 0:
        return False

    bull = (r["C"] / r["O"] - 1) * 100
    if not (BULL_MIN_PCT <= bull <= BULL_MAX_PCT):
        return False

    gap = (r["MA5"] / r["MA25"] - 1) * 100
    if not (MA5_VS_MA25_MIN_PCT <= gap <= MA5_VS_MA25_MAX_PCT):
        return False

    base = i - 1
    past = base - DECLINE_LOOKBACK
    if pd.isna(g["MA5"].iloc[past]) or g["MA5"].iloc[past] <= 0:
        return False
    decline = (g["MA5"].iloc[base] / g["MA5"].iloc[past] - 1) * 100
    if not (DECLINE_MIN_PCT <= decline <= DECLINE_MAX_PCT):
        return False

    for j in range(i - TURN_LOOKBACK, i):
        if g["MA5"].iloc[j] > g["MA5"].iloc[j - 1]:
            return False
    if not g["MA5"].iloc[i] > g["MA5"].iloc[i - 1]:
        return False

    if (r["C"] / r["MA5"] - 1) * 100 > CLOSE_VS_MA5_MAX_PCT:
        return False

    pv = g["Vo"].iloc[i-1]
    if pd.isna(pv) or pv <= 0:
        return False
    if float(r["Vo"]) / float(pv) < VOLUME_RATIO_MIN:
        return False
    return True


def metrics(part):
    if part.empty:
        return {"n": 0}
    r = part["ret10"].dropna()
    pos = r[r > 0].sum()
    neg = -r[r < 0].sum()
    return {
        "n": int(len(r)),
        "win_rate_pct": round(float((r > 0).mean() * 100), 2),
        "avg_ret10_pct": round(float(r.mean()), 3),
        "median_ret10_pct": round(float(r.median()), 3),
        "p5_plus_pct": round(float((r >= 5).mean() * 100), 2),
        "p10_plus_pct": round(float((r >= 10).mean() * 100), 2),
        "profit_factor": round(float(pos / neg), 3) if neg > 0 else None,
    }


def main():
    df = load_archive()
    rows = []
    for code, group in df.groupby("Code", sort=True):
        g = prepare(group)
        for i in range(25, len(g) - HOLD_DAYS - 1):
            if not is_kuitto(g, i):
                continue
            entry_i = i + 1
            exit_i = entry_i + HOLD_DAYS
            entry = float(g["O"].iloc[entry_i])
            exitp = float(g["O"].iloc[exit_i])
            if not (np.isfinite(entry) and np.isfinite(exitp) and entry > 0):
                continue
            r = g.iloc[i]
            rows.append({
                "code": str(code),
                "signal_date": str(pd.Timestamp(r["Date"]).date()),
                "ret10": (exitp / entry - 1) * 100,
                "atr14_pct": float(r["ATR14_PCT"]) if pd.notna(r["ATR14_PCT"]) else np.nan,
                "dd20_pct": float(r["DD20_PCT"]) if pd.notna(r["DD20_PCT"]) else np.nan,
                "avg_turnover20": float(r["AVG_TURNOVER20"]) if pd.notna(r["AVG_TURNOVER20"]) else np.nan,
            })

    t = pd.DataFrame(rows)
    t["signal_date_dt"] = pd.to_datetime(t["signal_date"])
    # Market-wide rebound proxy: how many current kuitto signals fire on the same day.
    # This uses same-day information only; no future data is involved.
    t["same_day_count"] = t.groupby("signal_date")["code"].transform("size")
    split = pd.Timestamp(df["Date"].min()).normalize() + pd.DateOffset(years=5)
    train = t[t["signal_date_dt"] < split].copy()
    test = t[t["signal_date_dt"] >= split].copy()

    # Thresholds are learned ONLY from the older 5y:
    # top quartile ATR, bottom quartile DD20, top quartile turnover.
    atr_q75 = float(train["atr14_pct"].quantile(0.75))
    dd20_q25 = float(train["dd20_pct"].quantile(0.25))
    turnover_q75 = float(train["avg_turnover20"].quantile(0.75))
    crowd_q75 = float(train["same_day_count"].quantile(0.75))

    rules = {
        "baseline": lambda x: pd.Series(True, index=x.index),
        "atr_only": lambda x: x["atr14_pct"] >= atr_q75,
        "atr_plus_dd20": lambda x: (x["atr14_pct"] >= atr_q75) & (x["dd20_pct"] <= dd20_q25),
        "atr_plus_turnover": lambda x: (x["atr14_pct"] >= atr_q75) & (x["avg_turnover20"] >= turnover_q75),
        "crowd_only": lambda x: x["same_day_count"] >= crowd_q75,
        "atr_plus_crowd": lambda x: (x["atr14_pct"] >= atr_q75) & (x["same_day_count"] >= crowd_q75),
        "atr_dd20_crowd": lambda x: (x["atr14_pct"] >= atr_q75) & (x["dd20_pct"] <= dd20_q25) & (x["same_day_count"] >= crowd_q75),
    }

    out = {
        "method": "Pseudo out-of-sample: derive thresholds from older 5y only, freeze them, then evaluate recent 5y untouched.",
        "split_date": str(split.date()),
        "learned_from_older5_only": {
            "atr14_pct_q75": round(atr_q75, 6),
            "dd20_pct_q25": round(dd20_q25, 6),
            "avg_turnover20_q75_yen": round(turnover_q75, 2),
            "same_day_count_q75": round(crowd_q75, 3),
        },
        "variants": {},
        "same_day_count_buckets": {},
        "caveat": "Historical universe is current-Prime survivors, so survivorship bias remains. This is pseudo-OOS, not a fully clean historical-universe OOS test."
    }

    for name, rule in rules.items():
        tr = train[rule(train).fillna(False)]
        te = test[rule(test).fillna(False)]
        out["variants"][name] = {
            "train_older5": metrics(tr),
            "test_recent5": metrics(te),
        }

    # Descriptive crowding buckets, shown separately for train/test.\n    bucket_defs = {\n        "1": lambda x: x["same_day_count"] == 1,\n        "2-3": lambda x: x["same_day_count"].between(2, 3),\n        "4-5": lambda x: x["same_day_count"].between(4, 5),\n        "6-10": lambda x: x["same_day_count"].between(6, 10),\n        "11+": lambda x: x["same_day_count"] >= 11,\n    }\n    for bname, brule in bucket_defs.items():\n        out["same_day_count_buckets"][bname] = {\n            "train_older5": metrics(train[brule(train)]),\n            "test_recent5": metrics(test[brule(test)]),\n        }\n\n    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "kuitto_pseudo_oos_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
