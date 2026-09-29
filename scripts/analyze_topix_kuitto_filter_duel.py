import json
from pathlib import Path
import pandas as pd

TRADES = Path("results/kuitto_runner_features_trades.csv")
TOPIX = Path("data/topix/topix_daily.parquet")
OUT = Path("results/topix_kuitto_filter_duel.json")

RUNNER_ATR14_MIN = 2.917505
RUNNER_DD20_MAX = -5.855856
RUNNER_TURNOVER20_MIN = 828_434_090

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

def add_topix_core(trades):
    tx = pd.read_parquet(TOPIX).copy()
    tx["date"] = pd.to_datetime(tx["Date"]).dt.normalize()
    for c in ["O", "C"]:
        tx[c] = pd.to_numeric(tx[c], errors="coerce")
    tx = tx.dropna(subset=["date", "O", "C"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    tx["MA5"] = tx["C"].rolling(5).mean()
    prev1 = tx["MA5"].shift(1) <= tx["MA5"].shift(2)
    prev2 = tx["MA5"].shift(2) <= tx["MA5"].shift(3)
    turn = tx["MA5"] > tx["MA5"].shift(1)
    bullish = tx["C"] > tx["O"]
    tx["topix_kuitto"] = prev1 & prev2 & turn & bullish
    event_dates = set(tx.loc[tx["topix_kuitto"], "date"])
    trades["topix_kuitto"] = trades["signal_date"].isin(event_dates)
    return trades

def add_filters(t):
    t = t.copy()
    t["winner"] = (t["atr14_pct"] >= 3.0) & (t["dd20_pct"] <= -5.5)

    score = pd.Series(0, index=t.index, dtype=int)
    score += (t["atr14_pct"] >= RUNNER_ATR14_MIN).astype(int) * 2
    score += (t["dd20_pct"] <= RUNNER_DD20_MAX).astype(int)
    score += (t["avg_turnover20"] >= RUNNER_TURNOVER20_MIN).astype(int)
    t["runner_score"] = score

    t["score3plus"] = t["runner_score"] >= 3
    t["score4"] = t["runner_score"] >= 4

    t["composite204"] = (
        (t["atr14_pct"] >= 3.4)
        & (
            ((t["dd20_pct"] > -7.0) & (t["dd20_pct"] <= -5.5))
            | ((t["dd20_pct"] > -9.0) & (t["dd20_pct"] <= -7.0) & (t["ma25_slope5_pct"] >= -1.5))
        )
    )
    return t

def yearly(df):
    if df.empty:
        return {}
    x = df.copy()
    x["year"] = x["signal_date"].dt.year
    return {str(int(y)): metrics(g) for y, g in x.groupby("year")}

def split_metrics(df, split):
    return {
        "older": metrics(df[df["signal_date"] < split]),
        "recent": metrics(df[df["signal_date"] >= split]),
    }

def main():
    t = pd.read_csv(TRADES, dtype={"code": str})
    t["signal_date"] = pd.to_datetime(t["signal_date"]).dt.normalize()
    for c in ["ret10","atr14_pct","dd20_pct","avg_turnover20","ma25_slope5_pct"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t = add_filters(add_topix_core(t))

    start = t["signal_date"].min().normalize()
    split = start + pd.DateOffset(years=5)

    variants = {
        "all": pd.Series(True, index=t.index),
        "winner_atr3_dd20m5_5": t["winner"],
        "runner_score3plus": t["score3plus"],
        "runner_score4": t["score4"],
        "composite204": t["composite204"],
    }

    out = {
        "period": {
            "start": str(start.date()),
            "end": str(t["signal_date"].max().date()),
            "split": str(split.date()),
        },
        "definitions": {
            "topix_kuitto": "TOPIX MA5 prior two slopes down/flat, today first upturn, bullish candle",
            "winner_atr3_dd20m5_5": "ATR14>=3.0 and DD20<=-5.5",
            "runner_score3plus": "score>=3; ATR14>=2.9175 gives 2pt, DD20<=-5.8559 gives 1pt, turnover20>=828.4m gives 1pt",
            "runner_score4": "all runner-score conditions satisfied",
            "composite204": "ATR14>=3.4; -7<DD20<=-5.5 OR (-9<DD20<=-7 and MA25 slope5>=-1.5)",
        },
        "overall": {},
        "topix_event": {},
        "topix_non_event": {},
        "event_vs_non_event_same_filter": {},
    }

    for name, mask in variants.items():
        allp = t[mask].copy()
        ev = t[mask & t["topix_kuitto"]].copy()
        non = t[mask & ~t["topix_kuitto"]].copy()

        out["overall"][name] = {
            "metrics": metrics(allp),
            "split": split_metrics(allp, split),
        }
        out["topix_event"][name] = {
            "metrics": metrics(ev),
            "split": split_metrics(ev, split),
            "yearly": yearly(ev),
        }
        out["topix_non_event"][name] = {
            "metrics": metrics(non),
            "split": split_metrics(non, split),
            "yearly": yearly(non),
        }

        em = metrics(ev)
        nm = metrics(non)
        out["event_vs_non_event_same_filter"][name] = {
            "event_n": em.get("n", 0),
            "non_event_n": nm.get("n", 0),
            "avg_ret_diff_pctpt": round(em.get("avg_ret10_pct", 0) - nm.get("avg_ret10_pct", 0), 3)
                if em.get("n",0) and nm.get("n",0) else None,
            "win_rate_diff_pctpt": round(em.get("win_rate_pct", 0) - nm.get("win_rate_pct", 0), 2)
                if em.get("n",0) and nm.get("n",0) else None,
            "pf_event": em.get("profit_factor"),
            "pf_non_event": nm.get("profit_factor"),
        }

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
