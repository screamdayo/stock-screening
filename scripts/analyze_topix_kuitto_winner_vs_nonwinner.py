import json
from pathlib import Path
import pandas as pd

TRADES = Path("results/kuitto_runner_features_trades.csv")
TOPIX = Path("data/topix/topix_daily.parquet")
OUT = Path("results/topix_kuitto_winner_vs_nonwinner.json")

def metrics(df):
    r = pd.to_numeric(df["ret10"], errors="coerce").dropna()
    if r.empty:
        return {"n":0}
    gp = r[r>0].sum()
    gl = -r[r<0].sum()
    return {
        "n": int(len(r)),
        "win_rate_pct": round(float((r>0).mean()*100),2),
        "avg_ret10_pct": round(float(r.mean()),3),
        "median_ret10_pct": round(float(r.median()),3),
        "p5_plus_pct": round(float((r>=5).mean()*100),2),
        "p10_plus_pct": round(float((r>=10).mean()*100),2),
        "loss3_pct": round(float((r<=-3).mean()*100),2),
        "loss5_pct": round(float((r<=-5).mean()*100),2),
        "loss10_pct": round(float((r<=-10).mean()*100),2),
        "profit_factor": round(float(gp/gl),3) if gl>0 else None,
    }

def main():
    t = pd.read_csv(TRADES, dtype={"code":str})
    t["signal_date"] = pd.to_datetime(t["signal_date"]).dt.normalize()
    for c in ["ret10","atr14_pct","dd20_pct","ma25_slope5_pct"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")

    tx = pd.read_parquet(TOPIX).copy()
    tx["date"] = pd.to_datetime(tx["Date"]).dt.normalize()
    for c in ["O","C"]:
        tx[c] = pd.to_numeric(tx[c], errors="coerce")
    tx = tx.dropna(subset=["date","O","C"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    tx["MA5"] = tx["C"].rolling(5).mean()
    core = (
        (tx["MA5"].shift(1) <= tx["MA5"].shift(2))
        & (tx["MA5"].shift(2) <= tx["MA5"].shift(3))
        & (tx["MA5"] > tx["MA5"].shift(1))
        & (tx["C"] > tx["O"])
    )
    event_dates = set(tx.loc[core, "date"])
    e = t[t["signal_date"].isin(event_dates)].copy()

    e["winner"] = (e["atr14_pct"] >= 3.0) & (e["dd20_pct"] <= -5.5)
    e["composite204"] = (
        (e["atr14_pct"] >= 3.4)
        & (
            ((e["dd20_pct"] > -7.0) & (e["dd20_pct"] <= -5.5))
            | ((e["dd20_pct"] > -9.0) & (e["dd20_pct"] <= -7.0) & (e["ma25_slope5_pct"] >= -1.5))
        )
    )

    groups = {
        "all_topix_kuitto": e,
        "winner": e[e["winner"]],
        "nonwinner": e[~e["winner"]],
        "composite204": e[e["composite204"]],
        "winner_excluding_204": e[e["winner"] & ~e["composite204"]],
        "nonwinner_but_204": e[~e["winner"] & e["composite204"]],
    }

    split = e["signal_date"].min() + pd.DateOffset(years=5)
    out = {
        "period": {
            "start": str(e["signal_date"].min().date()),
            "end": str(e["signal_date"].max().date()),
            "split": str(split.date()),
        },
        "definitions": {
            "topix_kuitto": "TOPIX MA5 prior two slopes down/flat, today first upturn, bullish candle",
            "winner": "ATR14>=3.0 and DD20<=-5.5",
            "composite204": "ATR14>=3.4; -7<DD20<=-5.5 OR (-9<DD20<=-7 and MA25 slope5>=-1.5)",
        },
        "groups": {},
        "direct_differences": {},
    }

    for name, g in groups.items():
        out["groups"][name] = {
            "metrics": metrics(g),
            "older": metrics(g[g["signal_date"] < split]),
            "recent": metrics(g[g["signal_date"] >= split]),
            "yearly": {
                str(int(y)): metrics(p)
                for y,p in g.groupby(g["signal_date"].dt.year)
            },
        }

    w = out["groups"]["winner"]["metrics"]
    nw = out["groups"]["nonwinner"]["metrics"]
    c = out["groups"]["composite204"]["metrics"]
    wx = out["groups"]["winner_excluding_204"]["metrics"]

    out["direct_differences"] = {
        "winner_minus_nonwinner": {
            "avg_ret_pctpt": round(w["avg_ret10_pct"] - nw["avg_ret10_pct"],3),
            "win_rate_pctpt": round(w["win_rate_pct"] - nw["win_rate_pct"],2),
            "p5_plus_pctpt": round(w["p5_plus_pct"] - nw["p5_plus_pct"],2),
            "loss5_pctpt": round(w["loss5_pct"] - nw["loss5_pct"],2),
            "pf_winner": w["profit_factor"],
            "pf_nonwinner": nw["profit_factor"],
        },
        "composite204_minus_winner_excluding_204": {
            "avg_ret_pctpt": round(c["avg_ret10_pct"] - wx["avg_ret10_pct"],3),
            "win_rate_pctpt": round(c["win_rate_pct"] - wx["win_rate_pct"],2),
            "p5_plus_pctpt": round(c["p5_plus_pct"] - wx["p5_plus_pct"],2),
            "loss5_pctpt": round(c["loss5_pct"] - wx["loss5_pct"],2),
            "pf_204": c["profit_factor"],
            "pf_winner_ex204": wx["profit_factor"],
        }
    }

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
