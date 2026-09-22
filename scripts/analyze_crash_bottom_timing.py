import json
from pathlib import Path
import pandas as pd
TR=Path("results/reversal_signal_baseline_trades.csv")
OUT=Path("results/crash_bottom_timing_diagnostic.json")
def met(s):
 s=pd.to_numeric(s,errors="coerce").dropna()
 if not len(s): return {"n":0}
 return {"n":int(len(s)),"win":round((s>0).mean()*100,2),"avg":round(s.mean(),3),"med":round(s.median(),3)}
def main():
 t=pd.read_csv(TR,dtype={"code":str}); t=t[t.type=="sharp_drop_reversal"].copy()
 t.signal_date=pd.to_datetime(t.signal_date)
 daily=t.groupby("signal_date").agg(crowd=("code","size"),med_ret5=("ret5_before_pct","median"),med_ret20=("ret20_before_pct","median"),med_dd20=("dd20_pct","median"),med_atr=("atr14_pct","median"),med_candle=("candle_pct","median"),med_vol=("vol_ratio20","median")).reset_index().sort_values("signal_date")
 stress=daily[daily.crowd>=20].copy(); stress["new"]=(stress.signal_date.diff().dt.days.fillna(999)>10).astype(int); stress["event"]=stress.new.cumsum()
 p=t[(t.ret5_before_pct<=-12)&(t.dd20_pct<=-20)&(t.candle_pct>=2)&(t.vol_ratio20>=1.5)].copy()
 rows=[]
 for eid,g in stress.groupby("event"):
  dates=list(g.signal_date)
  for i,(_,r) in enumerate(g.iterrows()):
   d=p[p.signal_date==r.signal_date]
   if not len(d): continue
   rows.append({"event":int(eid),"date":str(r.signal_date.date()),"event_day":i+1,"stress_days":len(g),
    "crowd":int(r.crowd),"crowd_vs_prev":None if i==0 else round(r.crowd/g.iloc[i-1].crowd,3),
    "med_ret5":round(r.med_ret5,3),"med_ret20":round(r.med_ret20,3),"med_dd20":round(r.med_dd20,3),
    "med_atr":round(r.med_atr,3),"med_candle":round(r.med_candle,3),"med_vol":round(r.med_vol,3),
    "anchor_n":len(d),"perf":met(d.ret_15d_pct)})
 # compare first vs later stress day and crowd expansion/contraction
 df=pd.DataFrame(rows)
 groups={}
 if len(df):
  groups["first_stress_day"]=met([x for r in rows if r["event_day"]==1 for x in p[p.signal_date==pd.Timestamp(r["date"])].ret_15d_pct])
  groups["later_stress_day"]=met([x for r in rows if r["event_day"]>1 for x in p[p.signal_date==pd.Timestamp(r["date"])].ret_15d_pct])
  groups["crowd_expanding"]=met([x for r in rows if r["crowd_vs_prev"] is not None and r["crowd_vs_prev"]>1 for x in p[p.signal_date==pd.Timestamp(r["date"])].ret_15d_pct])
  groups["crowd_contracting"]=met([x for r in rows if r["crowd_vs_prev"] is not None and r["crowd_vs_prev"]<1 for x in p[p.signal_date==pd.Timestamp(r["date"])].ret_15d_pct])
 OUT.write_text(json.dumps({"rows":rows,"groups":groups},ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps({"groups":groups,"rows":rows},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
