import json
from pathlib import Path
import pandas as pd

TRADES = Path("results/kuitto_runner_features_trades.csv")
TOPIX = Path("data/topix/topix_daily.parquet")
OUT = Path("results/kuitto_bad_market_duel.json")

def metrics(df):
    r = pd.to_numeric(df["ret10"], errors="coerce").dropna()
    if r.empty:
        return {"n": 0}
    gp = r[r > 0].sum()
    gl = -r[r < 0].sum()
    return {
        "n": int(len(r)),
        "win_rate_pct": round(float((r > 0).mean() * 100), 2),
        "avg_ret10_pct": round(float(r.mean()), 3),
        "median_ret10_pct": round(float(r.median()), 3),
        "p5_plus_pct": round(float((r >= 5).mean() * 100), 2),
        "p10_plus_pct": round(float((r >= 10).mean() * 100), 2),
        "loss3_pct": round(float((r <= -3).mean() * 100), 2),
        "loss5_pct": round(float((r <= -5).mean() * 100), 2),
        "profit_factor": round(float(gp / gl), 3) if gl > 0 else None,
    }

def main():
    t = pd.read_csv(TRADES, dtype={"code": str})
    t["signal_date"] = pd.to_datetime(t["signal_date"]).dt.normalize()
    t["ret10"] = pd.to_numeric(t["ret10"], errors="coerce")
    t["winner"] = (
        pd.to_numeric(t["atr14_pct"], errors="coerce") >= 3.0
    ) & (
        pd.to_numeric(t["dd20_pct"], errors="coerce") <= -5.5
    )

    x = pd.read_parquet(TOPIX).copy()
    x["date"] = pd.to_datetime(x["Date"]).dt.normalize()
    for c in ["O","C"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["date","C"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    x["ret1_pct"] = x["C"].pct_change() * 100
    x["MA5"] = x["C"].rolling(5).mean()
    x["MA25"] = x["C"].rolling(25).mean()
    x["ma5_slope1"] = x["MA5"].pct_change() * 100
    x["below_ma5"] = x["C"] < x["MA5"]
    x["below_ma25"] = x["C"] < x["MA25"]
    x["ma5_down"] = x["MA5"] < x["MA5"].shift(1)
    x["bearish"] = x["C"] < x["O"]

    cols = ["date","ret1_pct","below_ma5","below_ma25","ma5_down","bearish"]
    t = t.merge(x[cols], left_on="signal_date", right_on="date", how="left")

    regimes = {
        "topix_down": t["ret1_pct"] < 0,
        "topix_down_1pct": t["ret1_pct"] <= -1.0,
        "below_ma5": t["below_ma5"] == True,
        "below_ma25": t["below_ma25"] == True,
        "ma5_down": t["ma5_down"] == True,
        "bearish_day": t["bearish"] == True,
        "down_and_below_ma5": (t["ret1_pct"] < 0) & (t["below_ma5"] == True),
        "down_and_ma5_down": (t["ret1_pct"] < 0) & (t["ma5_down"] == True),
        "below_ma25_and_ma5_down": (t["below_ma25"] == True) & (t["ma5_down"] == True),
        "bad_triple": (t["ret1_pct"] < 0) & (t["below_ma25"] == True) & (t["ma5_down"] == True),
        "hard_bad": (t["ret1_pct"] <= -1.0) & (t["below_ma25"] == True) & (t["ma5_down"] == True),
    }

    split = pd.Timestamp("2021-11-01")
    out = {
        "period": {
            "start": str(t["signal_date"].min().date()),
            "end": str(t["signal_date"].max().date()),
            "split": str(split.date()),
        },
        "baseline": {
            "all": metrics(t),
            "winner": metrics(t[t["winner"]]),
        },
        "regimes": {},
    }

    for name, mask in regimes.items():
        bad = t[mask.fillna(False)].copy()
        other = t[~mask.fillna(False)].copy()
        out["regimes"][name] = {
            "all_bad": metrics(bad),
            "all_other": metrics(other),
            "winner_bad": metrics(bad[bad["winner"]]),
            "winner_other": metrics(other[other["winner"]]),
            "older_bad": metrics(bad[bad["signal_date"] < split]),
            "recent_bad": metrics(bad[bad["signal_date"] >= split]),
            "winner_older_bad": metrics(bad[(bad["signal_date"] < split) & bad["winner"]]),
            "winner_recent_bad": metrics(bad[(bad["signal_date"] >= split) & bad["winner"]]),
        }

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
