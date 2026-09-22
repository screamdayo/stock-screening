import os,json,requests,pandas as pd
from pathlib import Path
DATES=["2018-02-06","2018-12-25","2019-05-14","2019-08-06","2020-03-02","2020-03-10","2020-03-13","2020-03-17","2024-08-06","2024-08-07","2025-04-07","2025-04-08","2026-03-06"]
OUT=Path("results/crash_market_bottom_confirmation.json")
def get_topix():
 h={"x-api-key":os.environ["JQUANTS_API_KEY"]}
 r=requests.get("https://api.jquants.com/v2/indices/bars/daily/topix",params={"from":"2017-01-01","to":"2026-09-20"},headers=h,timeout=60); r.raise_for_status()
 j=r.json(); rows=j.get("data",j.get("topix",[]))
 d=pd.DataFrame(rows); d=d.rename(columns={"Date":"date","O":"open","H":"high","L":"low","C":"close"})
 d["date"]=pd.to_datetime(d.date)
 for c in ["open","high","low","close"]: d[c]=pd.to_numeric(d[c],errors="coerce")
 return d.sort_values("date").reset_index(drop=True)
def main():
 d=get_topix()
 d["ret1"]=d.close.pct_change()*100; d["ret2"]=d.close.pct_change(2)*100; d["ret5"]=d.close.pct_change(5)*100
 d["dd20"]=(d.close/d.close.rolling(20).max()-1)*100
 d["range"]=(d.high/d.low-1)*100
 d["body"]=(d.close/d.open-1)*100
 d["prev_high"]=d.high.shift(1); d["prev_close"]=d.close.shift(1); d["prev_low"]=d.low.shift(1)
 d["close_above_prev_high"]=d.close>d.prev_high
 d["low_above_prev_low"]=d.low>d.prev_low
 d["reclaim_prev_close"]=d.close>d.prev_close
 d["from_low20"]=(d.close/d.low.rolling(20).min()-1)*100
 out=[]
 for ds in DATES:
  z=d[d.date==pd.Timestamp(ds)]
  if z.empty: continue
  i=z.index[0]; r=d.loc[i]
  def q(x): return None if pd.isna(x) else round(float(x),3)
  out.append({"date":ds,"topix_ret1":q(r.ret1),"ret2":q(r.ret2),"ret5":q(r.ret5),"dd20":q(r.dd20),
   "intraday_body":q(r.body),"range":q(r.range),"from_low20":q(r.from_low20),
   "close_above_prev_high":bool(r.close_above_prev_high),"low_above_prev_low":bool(r.low_above_prev_low),
   "reclaim_prev_close":bool(r.reclaim_prev_close)})
 OUT.write_text(json.dumps({"rows":out},ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
