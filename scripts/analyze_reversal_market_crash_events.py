import json
from pathlib import Path
import pandas as pd
TR=Path("results/reversal_signal_baseline_trades.csv")
OUT=Path("results/reversal_market_crash_events.json")
def main():
 t=pd.read_csv(TR,dtype={"code":str}); t=t[t.type=="sharp_drop_reversal"].copy()
 t.signal_date=pd.to_datetime(t.signal_date); t["year"]=t.signal_date.dt.year
 cnt=t.groupby("signal_date").size().rename("crowd"); t=t.join(cnt,on="signal_date")
 p=t[(t.ret5_before_pct<=-12)&(t.dd20_pct<=-20)&(t.candle_pct>=2)&(t.vol_ratio20>=1.5)].copy()
 # Market-wide stress dates: >=20 coarse reversal signals. Cluster dates separated by <=5 trading-day observations
 daily=t.groupby("signal_date").agg(crowd=("code","size"),med_ret5=("ret5_before_pct","median"),med_dd20=("dd20_pct","median"),med_atr=("atr14_pct","median")).reset_index().sort_values("signal_date")
 stress=daily[daily.crowd>=20].copy()
 # cluster by calendar gap <=10 days to represent one crash/rebound episode
 stress["new_event"]=(stress.signal_date.diff().dt.days.fillna(999)>10).astype(int)
 stress["event_id"]=stress.new_event.cumsum()
 events=[]
 for eid,g in stress.groupby("event_id"):
  start,end=g.signal_date.min(),g.signal_date.max()
  # include anchor trades on stress dates in cluster
  d=p[p.signal_date.isin(g.signal_date)]
  s=pd.to_numeric(d.ret_15d_pct,errors="coerce").dropna()
  events.append({"event_id":int(eid),"start":str(start.date()),"end":str(end.date()),
    "stress_days":int(len(g)),"peak_crowd":int(g.crowd.max()),"sum_crowd":int(g.crowd.sum()),
    "anchor_trades":int(len(s)),"win_rate_pct":round((s>0).mean()*100,2) if len(s) else None,
    "avg_return_pct":round(s.mean(),3) if len(s) else None,"median_return_pct":round(s.median(),3) if len(s) else None,
    "median_market_ret5":round(g.med_ret5.median(),3),"median_market_dd20":round(g.med_dd20.median(),3),
    "median_market_atr":round(g.med_atr.median(),3)})
 OUT.write_text(json.dumps({"definition":"coarse reversal count >=20; stress dates within 10 calendar days clustered as one event","events":events},ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps(events,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
