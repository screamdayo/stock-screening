import json
from pathlib import Path
import pandas as pd
import numpy as np

TR=Path("results/reversal_signal_baseline_trades.csv")
OUT=Path("results/sharp_drop_reversal_regime_2024_2026.json")

def q(s):
 s=pd.to_numeric(s,errors="coerce").dropna()
 return {"n":int(len(s)),"mean":round(s.mean(),3),"median":round(s.median(),3)} if len(s) else {"n":0}

def main():
 t=pd.read_csv(TR,dtype={"code":str}); t=t[t.type=="sharp_drop_reversal"].copy()
 t["signal_date"]=pd.to_datetime(t.signal_date); t["year"]=t.signal_date.dt.year
 m=(t.ret5_before_pct<=-12)&(t.dd20_pct<=-20)&(t.candle_pct>=2)&(t.vol_ratio20>=1.5)
 p=t[m].copy()
 # Daily signal breadth/crowding from all sharp-drop baseline signals, no future data.
 daily=t.groupby("signal_date").agg(all_reversal_count=("code","size"),
     med_ret5=("ret5_before_pct","median"),med_dd20=("dd20_pct","median"),
     med_atr14=("atr14_pct","median"),med_vol_ratio=("vol_ratio20","median")).reset_index()
 p=p.merge(daily,on="signal_date",how="left")
 features=["ret5_before_pct","dd20_pct","candle_pct","vol_ratio20","atr14_pct","all_reversal_count",
           "med_ret5","med_dd20","med_atr14","med_vol_ratio"]
 out={"rule":"ret5<=-12, dd20<=-20, candle>=2, vol_ratio20>=1.5; fixed15 open exit",
      "years":{},"daily_crowding_buckets":{}}
 for y in [2024,2025,2026]:
  d=p[p.year==y]
  r=pd.to_numeric(d.ret_15d_pct,errors="coerce")
  out["years"][str(y)]={"n":len(d),"win_rate_pct":round((r>0).mean()*100,2),
    "avg_return_pct":round(r.mean(),3),"features":{f:q(d[f]) for f in features}}
 # Compare signal crowding directly.
 bins=[0,5,10,20,50,100,10**9]; labels=["1-5","6-10","11-20","21-50","51-100","101+"]
 p["crowd_bucket"]=pd.cut(p.all_reversal_count,bins=bins,labels=labels,include_lowest=True)
 for b,g in p.groupby("crowd_bucket",observed=True):
  r=pd.to_numeric(g.ret_15d_pct,errors="coerce")
  out["daily_crowding_buckets"][str(b)]={"n":len(g),"win_rate_pct":round((r>0).mean()*100,2),
    "avg_return_pct":round(r.mean(),3),"median_return_pct":round(r.median(),3),
    "years":{str(y):int((g.year==y).sum()) for y in [2024,2025,2026]}}
 OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
