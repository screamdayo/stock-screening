import json
from pathlib import Path
import pandas as pd
import numpy as np

SRC = Path("results/kuitto_runner_features_trades.csv")
OUT = Path("results/kuitto_winner_final_duel.json")

def metrics(df):
    s = pd.to_numeric(df["ret10"], errors="coerce").dropna()
    if s.empty:
        return {"n": 0}
    pos=s[s>0]; neg=s[s<0]; gl=-neg.sum()
    return {
        "n": int(len(s)),
        "win_rate_pct": round((s>0).mean()*100,2),
        "avg_ret10_pct": round(s.mean(),3),
        "median_ret10_pct": round(s.median(),3),
        "p5_plus_pct": round((s>=5).mean()*100,2),
        "p10_plus_pct": round((s>=10).mean()*100,2),
        "profit_factor": round(pos.sum()/gl,3) if gl>0 else None,
    }

def yearly(df):
    z=df.copy()
    z["year"]=z["signal_date"].dt.year
    return {str(int(y)):metrics(g) for y,g in z.groupby("year")}

def main():
    t=pd.read_csv(SRC,dtype={"code":str})
    t["signal_date"]=pd.to_datetime(t["signal_date"])
    for c in ["atr14_pct","dd20_pct","ma25_slope5_pct","ret10"]:
        t[c]=pd.to_numeric(t[c],errors="coerce")
    t=t.dropna(subset=["signal_date","atr14_pct","dd20_pct","ma25_slope5_pct","ret10"]).sort_values("signal_date")

    # Frozen finalists from prior exploration.
    masks={
      "aggressive_atr3_dd20m5.5": (t.atr14_pct>=3.0)&(t.dd20_pct<=-5.5),
      "trend_atr3_ma25slope0": (t.atr14_pct>=3.0)&(t.ma25_slope5_pct>=0.0),
    }
    split=pd.Timestamp("2021-09-20")
    out={"source_n":int(len(t)),"split_date":str(split.date()),"rules":{}}
    for name,m in masks.items():
        p=t[m].copy()
        out["rules"][name]={
          "overall":metrics(p),
          "older":metrics(p[p.signal_date<split]),
          "recent":metrics(p[p.signal_date>=split]),
          "yearly":yearly(p),
        }

    a=masks["aggressive_atr3_dd20m5.5"]
    b=masks["trend_atr3_ma25slope0"]
    both=t[a&b]; aonly=t[a&~b]; bonly=t[b&~a]
    out["overlap"]={
      "both":metrics(both),
      "aggressive_only":metrics(aonly),
      "trend_only":metrics(bonly),
    }
    A=out["rules"]["aggressive_atr3_dd20m5.5"]
    B=out["rules"]["trend_atr3_ma25slope0"]
    out["differences_aggressive_minus_trend"]={
      "overall_avg":round(A["overall"]["avg_ret10_pct"]-B["overall"]["avg_ret10_pct"],3),
      "overall_pf":round(A["overall"]["profit_factor"]-B["overall"]["profit_factor"],3),
      "overall_p5":round(A["overall"]["p5_plus_pct"]-B["overall"]["p5_plus_pct"],2),
      "overall_p10":round(A["overall"]["p10_plus_pct"]-B["overall"]["p10_plus_pct"],2),
      "recent_avg":round(A["recent"]["avg_ret10_pct"]-B["recent"]["avg_ret10_pct"],3),
      "recent_pf":round(A["recent"]["profit_factor"]-B["recent"]["profit_factor"],3),
      "recent_p5":round(A["recent"]["p5_plus_pct"]-B["recent"]["p5_plus_pct"],2),
      "recent_p10":round(A["recent"]["p10_plus_pct"]-B["recent"]["p10_plus_pct"],2),
    }
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
