import os,json,requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd

TR=Path("results/reversal_signal_baseline_trades.csv")
OUT=Path("results/pinch_candidate_sector_cap.json")
SENSOR_DATES=pd.to_datetime(["2018-12-27","2020-04-06","2022-03-10","2024-08-06","2025-04-08"])
TOPKS=[3,5]
CAPS=[None,2,1]

def norm(code):
    s=str(code)
    return s[:-1] if len(s)==5 and s.endswith("0") else s

def met(x):
    s=pd.to_numeric(pd.Series(x),errors="coerce").dropna()
    if s.empty:return {"n":0}
    neg=s[s<0].sum()
    return {"n":int(len(s)),"win":round(float((s>0).mean()*100),2),"avg":round(float(s.mean()),3),"med":round(float(s.median()),3),"pf":None if neg==0 else round(float(s[s>0].sum()/abs(neg)),3)}

def fetch_sector_map():
    h={"x-api-key":os.environ["JQUANTS_API_KEY"]}
    r=requests.get("https://api.jquants.com/v2/equities/master",headers=h,timeout=60)
    r.raise_for_status()
    d=pd.DataFrame(r.json().get("data",[]))
    if d.empty: raise RuntimeError("empty equities master")
    name_col=None
    for c in ["S33Nm","S33Name","Sector33CodeName","Sector33Name"]:
        if c in d.columns: name_col=c;break
    code_col="Code" if "Code" in d.columns else None
    if not code_col or not name_col:
        raise RuntimeError(f"sector fields not found: {list(d.columns)}")
    return {norm(r[code_col]):str(r[name_col]) for _,r in d.iterrows() if pd.notna(r[name_col])}

def prepare():
    t=pd.read_csv(TR,dtype={"code":str})
    t=t[t.type=="sharp_drop_reversal"].copy()
    t.signal_date=pd.to_datetime(t.signal_date)
    t=t[t.signal_date.isin(SENSOR_DATES)]
    t=t[(t.ret5_before_pct<=-12)&(t.dd20_pct<=-20)&(t.candle_pct>=2)&(t.vol_ratio20>=1.5)].copy()
    sm=fetch_sector_map()
    t["sector33"]=t.code.map(sm).fillna("不明")
    return t

def pick(g,k,cap):
    g=g.sort_values(["dd20_pct","code"]).copy()
    if cap is None:return g.head(k)
    out=[]; counts={}
    for _,r in g.iterrows():
        sec=r["sector33"]; n=counts.get(sec,0)
        if n>=cap:continue
        out.append(r);counts[sec]=n+1
        if len(out)>=k:break
    return pd.DataFrame(out,columns=g.columns)

def evaluate(args):
    k,cap=args
    picks=[];events=[]
    for d,g in DATA.groupby("signal_date"):
        p=pick(g,k,cap);picks.append(p)
        events.append({"date":str(d.date()),"available":int(len(g)),"picked":int(len(p)),
                       "sectors":p.sector33.value_counts().to_dict(),
                       "win15":round(float((p.ret_15d_pct>0).mean()*100),2) if len(p) else None,
                       "avg15":round(float(p.ret_15d_pct.mean()),3) if len(p) else None,
                       "med15":round(float(p.ret_15d_pct.median()),3) if len(p) else None})
    p=pd.concat(picks,ignore_index=True) if picks else DATA.iloc[0:0].copy()
    meds=[x["med15"] for x in events if x["med15"] is not None]
    avgs=[x["avg15"] for x in events if x["avg15"] is not None]
    return {"top_k":k,"sector_cap":"none" if cap is None else cap,
            "r5":met(p.ret_5d_pct),"r10":met(p.ret_10d_pct),"r15":met(p.ret_15d_pct),
            "event_median_of_medians":round(float(pd.Series(meds).median()),3) if meds else None,
            "worst_event_median15":round(float(min(meds)),3) if meds else None,
            "worst_event_avg15":round(float(min(avgs)),3) if avgs else None,
            "events":events}

DATA=prepare()

def main():
    jobs=[(k,c) for k in TOPKS for c in CAPS]
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex: rows=list(ex.map(evaluate,jobs))
    OUT.write_text(json.dumps({"study":"DD20 ranking x sector cap","sector_source":"current J-Quants equities master 33-sector classification","results":rows,
      "caveat":"Current sector classification is applied to historical candidate dates; classification history is not reconstructed."},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(rows,ensure_ascii=False,indent=2))
if __name__=="__main__":main()

# trigger
