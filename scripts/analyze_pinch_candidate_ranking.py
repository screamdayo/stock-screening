import json,math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd

TR=Path("results/reversal_signal_baseline_trades.csv")
OUT=Path("results/pinch_candidate_ranking.json")
SENSOR_DATES=pd.to_datetime(["2018-12-27","2020-04-06","2022-03-10","2024-08-06","2025-04-08"])
TOPKS=[3,5,10]

def met(x):
    s=pd.to_numeric(pd.Series(x),errors="coerce").dropna()
    if s.empty:return {"n":0}
    neg=s[s<0].sum()
    return {"n":int(len(s)),"win":round(float((s>0).mean()*100),2),"avg":round(float(s.mean()),3),"med":round(float(s.median()),3),"pf":None if neg==0 else round(float(s[s>0].sum()/abs(neg)),3)}

def prepare():
    t=pd.read_csv(TR,dtype={"code":str})
    t=t[t.type=="sharp_drop_reversal"].copy()
    t.signal_date=pd.to_datetime(t.signal_date)
    t=t[t.signal_date.isin(SENSOR_DATES)]
    t=t[(t.ret5_before_pct<=-12)&(t.dd20_pct<=-20)&(t.candle_pct>=2)&(t.vol_ratio20>=1.5)].copy()
    return t

def rank_group(g,mode):
    g=g.copy()
    if mode=="deep_ret5":
        return g.sort_values(["ret5_before_pct","code"])
    if mode=="deep_dd20":
        return g.sort_values(["dd20_pct","code"])
    if mode=="high_volume":
        return g.sort_values(["vol_ratio20","code"],ascending=[False,True])
    if mode=="strong_candle":
        return g.sort_values(["candle_pct","code"],ascending=[False,True])
    if mode=="composite":
        # equal-weight within-day percentile score; higher = stronger
        g["score"]=(
            g["ret5_before_pct"].rank(pct=True,ascending=False)+
            g["dd20_pct"].rank(pct=True,ascending=False)+
            g["vol_ratio20"].rank(pct=True,ascending=True)+
            g["candle_pct"].rank(pct=True,ascending=True)
        )
        return g.sort_values(["score","code"],ascending=[False,True])
    raise ValueError(mode)

def evaluate(args):
    mode,k=args
    picks=[]
    event_rows=[]
    for d,g in DATA.groupby("signal_date"):
        rg=rank_group(g,mode).head(k).copy()
        picks.append(rg)
        event_rows.append({
            "date":str(d.date()),"available":int(len(g)),"picked":int(len(rg)),
            "win15":round(float((rg.ret_15d_pct>0).mean()*100),2),
            "avg15":round(float(rg.ret_15d_pct.mean()),3),
            "med15":round(float(rg.ret_15d_pct.median()),3),
        })
    p=pd.concat(picks,ignore_index=True) if picks else DATA.iloc[0:0].copy()
    ev_meds=[x["med15"] for x in event_rows]
    ev_avgs=[x["avg15"] for x in event_rows]
    return {
        "ranker":mode,"top_k":k,
        "r5":met(p.ret_5d_pct),"r10":met(p.ret_10d_pct),"r15":met(p.ret_15d_pct),
        "events":event_rows,
        "event_median_of_medians":round(float(pd.Series(ev_meds).median()),3) if ev_meds else None,
        "worst_event_median15":round(float(min(ev_meds)),3) if ev_meds else None,
        "worst_event_avg15":round(float(min(ev_avgs)),3) if ev_avgs else None,
    }

DATA=prepare()

def main():
    modes=["deep_ret5","deep_dd20","high_volume","strong_candle","composite"]
    jobs=[(m,k) for m in modes for k in TOPKS]
    with ThreadPoolExecutor(max_workers=min(12,len(jobs))) as ex:
        rows=list(ex.map(evaluate,jobs))

    # baseline: all candidates
    event_base=[]
    for d,g in DATA.groupby("signal_date"):
        event_base.append({"date":str(d.date()),"available":int(len(g)),"win15":round(float((g.ret_15d_pct>0).mean()*100),2),"avg15":round(float(g.ret_15d_pct.mean()),3),"med15":round(float(g.ret_15d_pct.median()),3)})
    out={
      "study":"pinch sensor candidate ranking",
      "sensor_dates":[str(x.date()) for x in SENSOR_DATES],
      "candidate_rule":"ret5<=-12, dd20<=-20, candle>=2, vol_ratio20>=1.5",
      "baseline_all":{"r5":met(DATA.ret_5d_pct),"r10":met(DATA.ret_10d_pct),"r15":met(DATA.ret_15d_pct),"events":event_base},
      "rankings":sorted(rows,key=lambda x:(x["top_k"],x["ranker"])),
      "note":"Rankers and top-k variants evaluated in parallel. No sector constraint yet; next step can add industry caps if ranking remains strong."
    }
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
