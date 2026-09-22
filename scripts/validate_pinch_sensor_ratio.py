import json,pandas as pd
from pathlib import Path
TR=Path("results/reversal_signal_baseline_trades.csv"); OUT=Path("results/pinch_sensor_ratio_validation.json")
def met(x):
 x=pd.to_numeric(pd.Series(x),errors="coerce").dropna()
 if not len(x): return {"n":0}
 neg=x[x<0].sum()
 return {"n":int(len(x)),"win":round(float((x>0).mean()*100),2),"avg":round(float(x.mean()),3),"med":round(float(x.median()),3),"pf":None if neg==0 else round(float(x[x>0].sum()/abs(neg)),3)}
def main():
 t=pd.read_csv(TR,dtype={"code":str});t=t[t.type=="sharp_drop_reversal"].copy();t.signal_date=pd.to_datetime(t.signal_date)
 # eligible universe = codes with a baseline-computable observation that day, approximated by all rows' source universe availability is not stored here.
 # Use daily distinct codes represented in source signal export only if an explicit eligible-universe column exists; otherwise derive denominator from archive batch daily coverage is required.
 cols=set(t.columns)
 if "eligible_universe" in cols:
  denom=t.groupby("signal_date").eligible_universe.max()
 elif "universe_n" in cols:
  denom=t.groupby("signal_date").universe_n.max()
 else:
  raise RuntimeError("Need true daily eligible universe denominator; do not ratio-normalize using signal rows only.")
 daily=t.groupby("signal_date").size().rename("crowd").to_frame().join(denom.rename("universe"))
 daily["ratio_pct"]=daily.crowd/daily.universe*100
 # TOPIX-confirmed dates from prior frozen validation
 confirmed={"2018-12-27","2020-04-06","2022-03-10","2024-08-06","2025-04-08"}
 p=t[(t.ret5_before_pct<=-12)&(t.dd20_pct<=-20)&(t.candle_pct>=2)&(t.vol_ratio20>=1.5)].copy()
 grid=[]
 for th in [0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0]:
  ds=daily[daily.ratio_pct>=th].index
  ds=[d for d in ds if str(d.date()) in confirmed]
  a=p[p.signal_date.isin(ds)]
  grid.append({"ratio_threshold_pct":th,"days":len(ds),"dates":[str(d.date()) for d in ds],"r15":met(a.ret_15d_pct)})
 OUT.write_text(json.dumps({"note":"ratio denominator is true daily eligible universe","grid":grid,"confirmed_dates":[{"date":str(d.date()),"crowd":int(r.crowd),"universe":int(r.universe),"ratio_pct":round(float(r.ratio_pct),3)} for d,r in daily.iterrows() if str(d.date()) in confirmed]},ensure_ascii=False,indent=2),encoding="utf-8")
if __name__=="__main__":main()
