import os,json,requests,pandas as pd
from pathlib import Path
TR=Path("results/reversal_signal_baseline_trades.csv"); OUT=Path("results/crash_bottom_sensor_grid.json")
def topix():
 h={"x-api-key":os.environ["JQUANTS_API_KEY"]}
 r=requests.get("https://api.jquants.com/v2/indices/bars/daily/topix",params={"from":"2017-01-01","to":"2026-09-20"},headers=h,timeout=60); r.raise_for_status()
 d=pd.DataFrame(r.json().get("data",[])).rename(columns={"Date":"date","O":"open","H":"high","L":"low","C":"close"})
 d.date=pd.to_datetime(d.date)
 for c in ["open","high","low","close"]: d[c]=pd.to_numeric(d[c],errors="coerce")
 d["ret1"]=d.close.pct_change()*100; d["ret2"]=d.close.pct_change(2)*100; d["ret5"]=d.close.pct_change(5)*100
 d["dd20"]=(d.close/d.close.rolling(20).max()-1)*100; d["body"]=(d.close/d.open-1)*100
 d["from_low20"]=(d.close/d.low.rolling(20).min()-1)*100
 d["low_up"]=d.low>d.low.shift(1); d["reclaim"]=d.close>d.close.shift(1); d["above_prev_high"]=d.close>d.high.shift(1)
 return d.set_index("date")
def stats(x):
 x=pd.to_numeric(pd.Series(x),errors="coerce").dropna()
 if len(x)==0:return {"n":0}
 neg=x[x<0].sum(); return {"n":int(len(x)),"win":round(float((x>0).mean()*100),2),"avg":round(float(x.mean()),3),"med":round(float(x.median()),3),"pf":None if neg==0 else round(float(x[x>0].sum()/abs(neg)),3)}
def main():
 t=pd.read_csv(TR,dtype={"code":str}); t=t[t.type=="sharp_drop_reversal"].copy(); t.signal_date=pd.to_datetime(t.signal_date)
 daily=t.groupby("signal_date").size().rename("crowd")
 stress=daily[daily>=20].reset_index(name="crowd"); stress["new"]=(stress.signal_date.diff().dt.days.fillna(999)>10).astype(int); stress["event"]=stress.new.cumsum()
 p=t[(t.ret5_before_pct<=-12)&(t.dd20_pct<=-20)&(t.candle_pct>=2)&(t.vol_ratio20>=1.5)].copy()
 ix=topix(); rows=[]
 for _,r in stress.iterrows():
  d=r.signal_date
  if d not in ix.index:continue
  a=p[p.signal_date==d]
  if a.empty:continue
  z=ix.loc[d]
  rows.append({"date":str(d.date()),"event":int(r.event),"crowd":int(r.crowd),"n":len(a),"ret15":a.ret_15d_pct.tolist(),
   "ret1":float(z.ret1),"ret2":float(z.ret2),"ret5":float(z.ret5),"dd20":float(z.dd20),"body":float(z.body),"from_low20":float(z.from_low20),
   "low_up":bool(z.low_up),"reclaim":bool(z.reclaim),"above_prev_high":bool(z.above_prev_high)})
 df=pd.DataFrame(rows)
 rules=[]
 # deliberately coarse, interpretable confirmation rules; no winner selection by one metric
 specs=[
 ("A_ret3_lowup",lambda r:r.ret1>=3 and r.low_up),
 ("B_ret5_lowup",lambda r:r.ret1>=5 and r.low_up),
 ("C_ret3_fromlow5",lambda r:r.ret1>=3 and r.from_low20>=5),
 ("D_ret5_fromlow5",lambda r:r.ret1>=5 and r.from_low20>=5),
 ("E_body3_lowup",lambda r:r.body>=3 and r.low_up),
 ("F_reclaim_lowup_ret2",lambda r:r.reclaim and r.low_up and r.ret1>=2),
 ("G_prevhigh",lambda r:r.above_prev_high),
 ("H_ret3_lowup_crowd100",lambda r:r.ret1>=3 and r.low_up and r.crowd>=100)]
 for name,fn in specs:
  sel=[r for r in rows if fn(pd.Series(r))]
  trades=[x for r in sel for x in r["ret15"]]
  ev={}
  for r in sel: ev.setdefault(r["event"],[]).extend(r["ret15"])
  event_returns=[pd.Series(v).median() for v in ev.values()]
  rules.append({"rule":name,"days":len(sel),"events":len(ev),"trade":stats(trades),"event_median":stats(event_returns),"dates":[r["date"] for r in sel]})
 OUT.write_text(json.dumps({"rules":rules,"rows":[{k:v for k,v in r.items() if k!="ret15"} for r in rows]},ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps(rules,ensure_ascii=False,indent=2))
if __name__=="__main__":main()

# trigger run
