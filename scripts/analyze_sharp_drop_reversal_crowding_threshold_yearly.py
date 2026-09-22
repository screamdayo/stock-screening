import json
from pathlib import Path
import pandas as pd

TR=Path("results/reversal_signal_baseline_trades.csv")
OUT=Path("results/sharp_drop_reversal_crowding_threshold_yearly.json")
TH=[1,5,10,20,30,50,75,100,150,200,300,500]

def met(d):
 s=pd.to_numeric(d.ret_15d_pct,errors="coerce").dropna()
 if not len(s): return {"n":0}
 pos=s[s>0]; neg=s[s<0]; gl=-neg.sum()
 return {"n":int(len(s)),"win_rate_pct":round((s>0).mean()*100,2),
 "avg_return_pct":round(s.mean(),3),"median_return_pct":round(s.median(),3),
 "profit_factor":round(pos.sum()/gl,3) if gl>0 else None}

def main():
 t=pd.read_csv(TR,dtype={"code":str}); t=t[t.type=="sharp_drop_reversal"].copy()
 t["signal_date"]=pd.to_datetime(t.signal_date); t["year"]=t.signal_date.dt.year
 # count all coarse reversal signals per day; this is known on signal close
 cnt=t.groupby("signal_date").size().rename("daily_reversal_count")
 t=t.join(cnt,on="signal_date")
 p=t[(t.ret5_before_pct<=-12)&(t.dd20_pct<=-20)&(t.candle_pct>=2)&(t.vol_ratio20>=1.5)].copy()
 out={"rule":"ret5<=-12, dd20<=-20, candle>=2, vol_ratio20>=1.5; fixed15 open exit",
 "count_definition":"all coarse sharp_drop_reversal signals on same signal date",
 "thresholds":{}}
 years=sorted(p.year.unique())
 for th in TH:
  d=p[p.daily_reversal_count>=th]
  out["thresholds"][str(th)]={"overall":met(d),"yearly":{str(int(y)):met(d[d.year==y]) for y in years}}
 OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
