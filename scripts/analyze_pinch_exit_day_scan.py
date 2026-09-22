import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

BATCH=Path("data/batches")
TR=Path("results/reversal_signal_baseline_trades.csv")
OUT=Path("results/pinch_exit_day_scan.json")
DATES=pd.to_datetime(["2018-12-27","2020-04-06","2022-03-10","2024-08-06","2025-04-08"])
HOLDS=list(range(8,31))
SLOTS=3

def load_candidates():
    t=pd.read_csv(TR,dtype={"code":str})
    t=t[t.type=="sharp_drop_reversal"].copy()
    t["signal_date"]=pd.to_datetime(t.signal_date)
    t=t[t.signal_date.isin(DATES)]
    t=t[(t.ret5_before_pct<=-12)&(t.dd20_pct<=-20)&(t.candle_pct>=2)&(t.vol_ratio20>=1.5)].copy()
    t=t.sort_values(["signal_date","dd20_pct","code"]).reset_index(drop=True)
    t["dd20_rank"]=t.groupby("signal_date").cumcount()+1
    return t[t.dd20_rank<=3].copy()

CANDS=load_candidates()
CODESET=set(CANDS.code.astype(str))

def load_prices():
    fs=[]
    for p in sorted(BATCH.glob("batch_*.parquet")):
        d=pd.read_parquet(p)
        if "ArchiveMarket" in d.columns:
            d=d[d.ArchiveMarket=="プライム"].copy()
        d["Code"]=d.Code.astype(str)
        d=d[d.Code.isin(CODESET)].copy()
        if d.empty: continue
        for raw,adj in [("O","AdjO"),("H","AdjH"),("L","AdjL"),("C","AdjC")]:
            if adj in d.columns:
                rawv=pd.to_numeric(d[raw],errors="coerce") if raw in d.columns else pd.Series(np.nan,index=d.index)
                adjv=pd.to_numeric(d[adj],errors="coerce")
                d[raw]=adjv.where(adjv.notna(),rawv)
            else:
                d[raw]=pd.to_numeric(d[raw],errors="coerce")
        d["Date"]=pd.to_datetime(d.Date)
        fs.append(d[["Code","Date","O","H","L","C"]])
    if not fs: raise RuntimeError("No archive prices")
    x=pd.concat(fs,ignore_index=True).dropna(subset=["O","H","L","C"])
    return x.drop_duplicates(["Code","Date"],keep="last").sort_values(["Code","Date"])

PX=load_prices()
BYCODE={c:g.sort_values("Date").reset_index(drop=True) for c,g in PX.groupby("Code")}

def one_trade(code,signal_date,hold):
    g=BYCODE.get(str(code))
    if g is None:return None
    ix=g.index[g.Date==pd.Timestamp(signal_date)]
    if len(ix)==0:return None
    si=int(ix[0]); ei=si+1; xi=ei+hold
    if xi>=len(g):return None
    entry=float(g.O.iloc[ei]); exitp=float(g.O.iloc[xi])
    if not(np.isfinite(entry) and np.isfinite(exitp) and entry>0):return None
    return (exitp/entry-1)*100

def evaluate(hold):
    events=[]; allrets=[]
    for d,g in CANDS.groupby("signal_date"):
        rs=[]
        for _,r in g.sort_values("dd20_rank").iterrows():
            v=one_trade(r.code,d,hold)
            if v is not None: rs.append(v)
        allrets += rs
        port=sum(rs)/SLOTS  # missing slots stay cash, e.g. 2018 only 2 names
        events.append({"date":str(pd.Timestamp(d).date()),"executed":len(rs),"portfolio_return_pct":round(port,3)})
    ports=pd.Series([x["portfolio_return_pct"] for x in events],dtype=float)
    rs=pd.Series(allrets,dtype=float)
    return {
      "hold_days":hold,
      "event_avg_return_pct":round(float(ports.mean()),3),
      "event_median_return_pct":round(float(ports.median()),3),
      "worst_event_return_pct":round(float(ports.min()),3),
      "best_event_return_pct":round(float(ports.max()),3),
      "event_stdev_pct":round(float(ports.std(ddof=0)),3),
      "positive_events":int((ports>0).sum()),
      "trade_win_pct":round(float((rs>0).mean()*100),2),
      "trade_avg_return_pct":round(float(rs.mean()),3),
      "trade_median_return_pct":round(float(rs.median()),3),
      "events":events
    }

def main():
    with ThreadPoolExecutor(max_workers=16) as ex:
        rows=list(ex.map(evaluate,HOLDS))
    ranked=sorted(rows,key=lambda x:(-x["event_avg_return_pct"],-x["worst_event_return_pct"],x["event_stdev_pct"]))
    # Robust plateau candidates: within 10% of best average and all 5 events positive.
    best=ranked[0]["event_avg_return_pct"]
    plateau=[r for r in sorted(rows,key=lambda x:x["hold_days"]) if r["positive_events"]==5 and r["event_avg_return_pct"]>=best*0.90]
    out={
      "study":"Pinch TOP3 fixed exit day scan",
      "entry":"Next business day open, no gap limit, no fixed stop",
      "exit":"Open on Nth business day after entry",
      "scan_days":HOLDS,
      "best_by_event_average":ranked[:10],
      "robust_plateau_within_90pct_of_best":plateau,
      "all_results":sorted(rows,key=lambda x:x["hold_days"]),
      "caveat":"Only five independent crash events; choose a broad plateau rather than overfitting a single exact day."
    }
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"best":ranked[:10],"plateau":plateau},ensure_ascii=False,indent=2))
if __name__=="__main__":main()

# trigger
