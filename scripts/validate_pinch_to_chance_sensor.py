import os,json,requests,pandas as pd
from pathlib import Path
TR=Path("results/reversal_signal_baseline_trades.csv"); OUT=Path("results/pinch_to_chance_sensor_validation.json")
def get_ix():
 h={"x-api-key":os.environ["JQUANTS_API_KEY"]}
 r=requests.get("https://api.jquants.com/v2/indices/bars/daily/topix",params={"from":"2017-01-01","to":"2026-09-20"},headers=h,timeout=60);r.raise_for_status()
 d=pd.DataFrame(r.json().get("data",[])).rename(columns={"Date":"date","O":"open","H":"high","L":"low","C":"close"});d.date=pd.to_datetime(d.date)
 for c in ["open","high","low","close"]:d[c]=pd.to_numeric(d[c],errors="coerce")
 d["ret1"]=d.close.pct_change()*100;d["body"]=(d.close/d.open-1)*100;d["low_up"]=d.low>d.low.shift(1);d["from_low20"]=(d.close/d.low.rolling(20).min()-1)*100
 return d.set_index("date")
def met(x):
 x=pd.to_numeric(pd.Series(x),errors="coerce").dropna()
 if not len(x):return {"n":0}
 neg=x[x<0].sum()
 return {"n":int(len(x)),"win":round(float((x>0).mean()*100),2),"avg":round(float(x.mean()),3),"med":round(float(x.median()),3),"pf":None if neg==0 else round(float(x[x>0].sum()/abs(neg)),3)}
def main():
 t=pd.read_csv(TR,dtype={"code":str});t=t[t.type=="sharp_drop_reversal"].copy();t.signal_date=pd.to_datetime(t.signal_date)
 daily=t.groupby("signal_date").size().rename("crowd")
 # fixed individual strength rule from prior study
 p=t[(t.ret5_before_pct<=-12)&(t.dd20_pct<=-20)&(t.candle_pct>=2)&(t.vol_ratio20>=1.5)].copy()
 ix=get_ix(); candidates=[]
 for d,crowd in daily.items():
  if crowd<20 or d not in ix.index:continue
  z=ix.loc[d]
  if not (z.ret1>=3 and z.low_up):continue
  a=p[p.signal_date==d]
  if a.empty:continue
  candidates.append({"date":str(d.date()),"year":d.year,"crowd":int(crowd),"topix_ret1":round(float(z.ret1),3),"body":round(float(z.body),3),"from_low20":round(float(z.from_low20),3),"anchor_n":int(len(a)),"r5":a.ret_5d_pct.tolist(),"r10":a.ret_10d_pct.tolist(),"r15":a.ret_15d_pct.tolist()})
 # cluster sensor dates <=10 calendar days; evaluate event medians, not just trades
 ev=[]; eid=0;prev=None
 for r in candidates:
  d=pd.Timestamp(r["date"])
  if prev is None or (d-prev).days>10:eid+=1
  r["event"]=eid;prev=d
 for e,g in pd.DataFrame(candidates).groupby("event"):
  dates=g.date.tolist(); vals5=[x for arr in g.r5 for x in arr];vals10=[x for arr in g.r10 for x in arr];vals15=[x for arr in g.r15 for x in arr]
  ev.append({"event":int(e),"dates":dates,"days":len(g),"trades":int(sum(g.anchor_n)),"med5":round(float(pd.Series(vals5).median()),3),"med10":round(float(pd.Series(vals10).median()),3),"med15":round(float(pd.Series(vals15).median()),3),"avg15":round(float(pd.Series(vals15).mean()),3)})
 # leave-one-event-out descriptive stability: stats after removing each event
 loo=[]
 for e in [x["event"] for x in ev]:
  rows=[r for r in candidates if r["event"]!=e]; vals=[x for r in rows for x in r["r15"]]
  loo.append({"left_out_event":e,"trade15":met(vals)})
 # yearly + horizons
 yearly={}
 for y in sorted(set(r["year"] for r in candidates)):
  rs=[r for r in candidates if r["year"]==y];yearly[str(y)]={"events":len(set(r["event"] for r in rs)),"days":len(rs),"r5":met([x for r in rs for x in r["r5"]]),"r10":met([x for r in rs for x in r["r10"]]),"r15":met([x for r in rs for x in r["r15"]])}
 OUT.write_text(json.dumps({"sensor":"crowd>=20 + TOPIX ret1>=3% + TOPIX low>prior low; individual anchor ret5<=-12, dd20<=-20, candle>=2, volratio>=1.5","candidates":[{k:v for k,v in r.items() if k not in ["r5","r10","r15"]} for r in candidates],"overall":{"r5":met([x for r in candidates for x in r["r5"]]),"r10":met([x for r in candidates for x in r["r10"]]),"r15":met([x for r in candidates for x in r["r15"]])},"events":ev,"yearly":yearly,"leave_one_event_out":loo},ensure_ascii=False,indent=2),encoding="utf-8")
if __name__=="__main__":main()

# trigger validation
