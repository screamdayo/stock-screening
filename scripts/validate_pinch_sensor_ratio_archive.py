import os,json,requests,pandas as pd
from pathlib import Path
B=Path("data/batches"); TR=Path("results/reversal_signal_baseline_trades.csv"); OUT=Path("results/pinch_sensor_ratio_validation.json")
def met(x):
 x=pd.to_numeric(pd.Series(x),errors="coerce").dropna()
 if not len(x):return {"n":0}
 neg=x[x<0].sum();return {"n":int(len(x)),"win":round(float((x>0).mean()*100),2),"avg":round(float(x.mean()),3),"med":round(float(x.median()),3),"pf":None if neg==0 else round(float(x[x>0].sum()/abs(neg)),3)}
def universe():
 fs=[]
 for p in sorted(B.glob("batch_*.parquet")):
  d=pd.read_parquet(p,columns=None)
  if "ArchiveMarket" in d.columns:d=d[d.ArchiveMarket=="プライム"]
  d["Date"]=pd.to_datetime(d.Date);d["Code"]=d.Code.astype(str)
  ok=pd.Series(True,index=d.index)
  for raw,adj in [("O","AdjO"),("H","AdjH"),("L","AdjL"),("C","AdjC")]:
   v=d[adj].where(d[adj].notna(),d[raw]) if adj in d.columns else d[raw]
   ok &= pd.to_numeric(v,errors="coerce").notna()
  fs.append(d.loc[ok,["Date","Code"]])
 x=pd.concat(fs).drop_duplicates(["Date","Code"])
 return x.groupby("Date").Code.nunique()
def topix():
 h={"x-api-key":os.environ["JQUANTS_API_KEY"]};r=requests.get("https://api.jquants.com/v2/indices/bars/daily/topix",params={"from":"2017-01-01","to":"2026-09-20"},headers=h,timeout=60);r.raise_for_status()
 d=pd.DataFrame(r.json().get("data",[])).rename(columns={"Date":"date","L":"low","C":"close"});d.date=pd.to_datetime(d.date)
 d["low"]=pd.to_numeric(d.low);d["close"]=pd.to_numeric(d.close);d["ret1"]=d.close.pct_change()*100;d["low_up"]=d.low>d.low.shift(1)
 return d.set_index("date")
def main():
 t=pd.read_csv(TR,dtype={"code":str});t=t[t.type=="sharp_drop_reversal"].copy();t.signal_date=pd.to_datetime(t.signal_date)
 u=universe(); crowd=t.groupby("signal_date").size(); ix=topix()
 daily=pd.concat([crowd.rename("crowd"),u.rename("universe")],axis=1).dropna();daily["ratio"]=daily.crowd/daily.universe*100
 p=t[(t.ret5_before_pct<=-12)&(t.dd20_pct<=-20)&(t.candle_pct>=2)&(t.vol_ratio20>=1.5)]
 rows=[]
 for th in [.5,.75,1,1.25,1.5,2,2.5,3]:
  dates=[]
  for d,r in daily.iterrows():
   if r.ratio<th or d not in ix.index:continue
   z=ix.loc[d]
   if z.ret1>=3 and bool(z.low_up) and not p[p.signal_date==d].empty:dates.append(d)
  a=p[p.signal_date.isin(dates)]
  rows.append({"threshold_pct":th,"days":len(dates),"events":len(dates),"dates":[str(x.date()) for x in dates],"r15":met(a.ret_15d_pct)})
 key=[]
 for d in sorted(set(sum([[pd.Timestamp(x) for x in r["dates"]] for r in rows],[]))):
  if d in daily.index:key.append({"date":str(d.date()),"crowd":int(daily.loc[d].crowd),"universe":int(daily.loc[d].universe),"ratio_pct":round(float(daily.loc[d].ratio),3)})
 OUT.write_text(json.dumps({"definition":"coarse reversal count / current-Prime archive codes with valid OHLC that day","grid":rows,"detected_date_ratios":key,"caveat":"Current-Prime survivor-biased archive denominator; ratio is internally consistent but not historical TOPIX membership."},ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps(rows,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
