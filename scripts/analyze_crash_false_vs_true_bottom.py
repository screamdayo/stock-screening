import json
from pathlib import Path
import pandas as pd
TR=Path("results/reversal_signal_baseline_trades.csv")
OUT=Path("results/crash_false_vs_true_bottom.json")
DATES=["2020-03-02","2020-03-10","2020-03-13","2020-03-17","2024-08-06","2024-08-07","2025-04-07","2025-04-08"]
def main():
 t=pd.read_csv(TR,dtype={"code":str}); t=t[t.type=="sharp_drop_reversal"].copy()
 t.signal_date=pd.to_datetime(t.signal_date)
 # all coarse reversal signals = market cross-section observable at close
 daily=t.groupby("signal_date").agg(
  crowd=("code","size"),
  ret5_mean=("ret5_before_pct","mean"),ret5_med=("ret5_before_pct","median"),
  ret20_mean=("ret20_before_pct","mean"),ret20_med=("ret20_before_pct","median"),
  dd20_mean=("dd20_pct","mean"),dd20_med=("dd20_pct","median"),
  atr_mean=("atr14_pct","mean"),atr_med=("atr14_pct","median"),
  candle_mean=("candle_pct","mean"),candle_med=("candle_pct","median"),
  vol_mean=("vol_ratio20","mean"),vol_med=("vol_ratio20","median")).sort_index()
 # derive stress-history features using only current/past signal counts
 daily["crowd_max_5obs"]=daily.crowd.rolling(5,min_periods=1).max()
 daily["crowd_max_10obs"]=daily.crowd.rolling(10,min_periods=1).max()
 daily["crowd_sum_5obs"]=daily.crowd.rolling(5,min_periods=1).sum()
 daily["crowd_prev_max_5obs"]=daily.crowd.shift(1).rolling(5,min_periods=1).max()
 daily["crowd_vs_prev5max"]=daily.crowd/daily.crowd_prev_max_5obs
 daily["dd_change_vs_prev_obs"]=daily.dd20_med-daily.dd20_med.shift(1)
 daily["ret20_change_vs_prev_obs"]=daily.ret20_med-daily.ret20_med.shift(1)
 p=t[(t.ret5_before_pct<=-12)&(t.dd20_pct<=-20)&(t.candle_pct>=2)&(t.vol_ratio20>=1.5)].copy()
 out=[]
 for ds in DATES:
  d=pd.Timestamp(ds)
  if d not in daily.index: continue
  r=daily.loc[d]
  a=p[p.signal_date==d]; perf=pd.to_numeric(a.ret_15d_pct,errors="coerce").dropna()
  out.append({"date":ds,"crowd":int(r.crowd),
   "ret5_med":round(r.ret5_med,3),"ret20_med":round(r.ret20_med,3),"dd20_med":round(r.dd20_med,3),
   "atr_med":round(r.atr_med,3),"candle_med":round(r.candle_med,3),"vol_med":round(r.vol_med,3),
   "crowd_prev5max":None if pd.isna(r.crowd_prev_max_5obs) else int(r.crowd_prev_max_5obs),
   "crowd_vs_prev5max":None if pd.isna(r.crowd_vs_prev5max) else round(r.crowd_vs_prev5max,3),
   "crowd_sum5":int(r.crowd_sum_5obs),"dd_change_prev":None if pd.isna(r.dd_change_vs_prev_obs) else round(r.dd_change_vs_prev_obs,3),
   "anchor_n":int(len(perf)),"win15":round((perf>0).mean()*100,2) if len(perf) else None,
   "avg15":round(perf.mean(),3) if len(perf) else None})
 OUT.write_text(json.dumps({"note":"Cross-sectional coarse-reversal features known by signal close; rolling obs are prior reversal-signal dates, not every market day.","dates":out},ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
