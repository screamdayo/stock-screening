import json
from pathlib import Path
import pandas as pd

SRC=Path("results/kuitto_runner_features_trades.csv")
OUT=Path("results/kuitto_winner_trend_overlay_scan.json")

def met(df):
    s=pd.to_numeric(df["ret10"],errors="coerce").dropna()
    if s.empty:return {"n":0}
    pos=s[s>0]; neg=s[s<0]; gl=-neg.sum()
    return {
      "n":int(len(s)),
      "win_rate_pct":round((s>0).mean()*100,2),
      "avg_ret10_pct":round(s.mean(),3),
      "median_ret10_pct":round(s.median(),3),
      "p5_plus_pct":round((s>=5).mean()*100,2),
      "p10_plus_pct":round((s>=10).mean()*100,2),
      "profit_factor":round(pos.sum()/gl,3) if gl>0 else None,
    }

def yearly(df):
    z=df.copy(); z["year"]=z.signal_date.dt.year
    return {str(int(y)):met(g) for y,g in z.groupby("year")}

def main():
    t=pd.read_csv(SRC,dtype={"code":str})
    t["signal_date"]=pd.to_datetime(t["signal_date"])
    for c in ["atr14_pct","dd20_pct","ma25_slope5_pct","ret10"]:
        t[c]=pd.to_numeric(t[c],errors="coerce")
    t=t.dropna(subset=["signal_date","atr14_pct","dd20_pct","ma25_slope5_pct","ret10"])
    base=t[(t.atr14_pct>=3.0)&(t.dd20_pct<=-5.5)].copy()
    split=pd.Timestamp("2021-09-20")
    thresholds=[-1.5,-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5,0.75,1.0]
    rows=[]
    for th in thresholds:
        p=base[base.ma25_slope5_pct>=th].copy()
        rows.append({
          "ma25_slope5_min":th,
          "overall":met(p),
          "older":met(p[p.signal_date<split]),
          "recent":met(p[p.signal_date>=split]),
          "yearly":yearly(p),
        })
    out={
      "base_rule":"ATR14>=3.0 and DD20<=-5.5",
      "base":{"overall":met(base),"older":met(base[base.signal_date<split]),"recent":met(base[base.signal_date>=split]),"yearly":yearly(base)},
      "scan":rows,
      "note":"Trend overlay scan with ATR/DD20 frozen. Prefer broad plateau and pseudo-OOS stability over a single best threshold."
    }
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
