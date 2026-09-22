import json
from pathlib import Path
import numpy as np
import pandas as pd

BATCH=Path("data/batches")
TR=Path("results/reversal_signal_baseline_trades.csv")
OUT=Path("results/pinch_exit_final_robustness.json")
DATES=pd.to_datetime(["2018-12-27","2020-04-06","2022-03-10","2024-08-06","2025-04-08"])
DAYS=[40,41,42,43]
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

def ret_at(code,signal_date,hold):
    g=BYCODE.get(str(code))
    if g is None:return None
    ix=g.index[g.Date==pd.Timestamp(signal_date)]
    if len(ix)==0:return None
    si=int(ix[0]); ei=si+1; xi=ei+hold
    if xi>=len(g):return None
    entry=float(g.O.iloc[ei]); exitp=float(g.O.iloc[xi])
    if not(np.isfinite(entry) and np.isfinite(exitp) and entry>0):return None
    return (exitp/entry-1)*100

# Precompute event x day trade returns.
RET={}
for d,g in CANDS.groupby("signal_date"):
    for h in DAYS:
        rows=[]
        for _,r in g.sort_values("dd20_rank").iterrows():
            v=ret_at(r.code,d,h)
            if v is not None: rows.append({"code":r.code,"rank":int(r.dd20_rank),"ret":v})
        RET[(pd.Timestamp(d),h)]=rows

def event_port(rows):
    return sum(x["ret"] for x in rows)/SLOTS

def metrics(vals):
    s=pd.Series(vals,dtype=float)
    return {
      "n_events":int(len(s)),
      "avg_pct":round(float(s.mean()),3),
      "median_pct":round(float(s.median()),3),
      "worst_pct":round(float(s.min()),3),
      "best_pct":round(float(s.max()),3),
      "stdev_pct":round(float(s.std(ddof=0)),3),
      "positive_events":int((s>0).sum())
    }

def main():
    loo=[]
    for excluded in DATES:
        for h in DAYS:
            vals=[]
            detail=[]
            for d in DATES:
                if d==excluded: continue
                rows=RET[(pd.Timestamp(d),h)]
                p=event_port(rows); vals.append(p)
                detail.append({"date":str(pd.Timestamp(d).date()),"portfolio_return_pct":round(p,3)})
            loo.append({"excluded_event":str(excluded.date()),"hold_days":h,**metrics(vals),"events":detail})

    trimmed=[]
    for h in DAYS:
        vals=[];details=[]
        for d in DATES:
            rows=list(RET[(pd.Timestamp(d),h)])
            if rows:
                removed=max(rows,key=lambda x:x["ret"])
                kept=[x for x in rows if x is not removed]
            else:
                removed=None;kept=[]
            # Keep original 3-slot capital denominator: removed winner becomes cash.
            p=event_port(kept); vals.append(p)
            details.append({
              "date":str(pd.Timestamp(d).date()),
              "removed":None if removed is None else {"code":removed["code"],"rank":removed["rank"],"return_pct":round(removed["ret"],3)},
              "portfolio_return_pct_after_remove_best":round(p,3)
            })
        trimmed.append({"hold_days":h,**metrics(vals),"events":details})

    full=[]
    for h in DAYS:
        vals=[event_port(RET[(pd.Timestamp(d),h)]) for d in DATES]
        full.append({"hold_days":h,**metrics(vals)})

    # Robustness summary per day: worst LOO average + trimmed average.
    summary=[]
    for h in DAYS:
        l=[x for x in loo if x["hold_days"]==h]
        t=next(x for x in trimmed if x["hold_days"]==h)
        f=next(x for x in full if x["hold_days"]==h)
        summary.append({
          "hold_days":h,
          "full_avg_pct":f["avg_pct"],
          "full_worst_event_pct":f["worst_pct"],
          "worst_leave_one_out_avg_pct":round(min(x["avg_pct"] for x in l),3),
          "best_leave_one_out_avg_pct":round(max(x["avg_pct"] for x in l),3),
          "trimmed_avg_pct":t["avg_pct"],
          "trimmed_worst_event_pct":t["worst_pct"],
          "trimmed_positive_events":t["positive_events"]
        })

    out={
      "study":"Pinch exit final robustness: 40-43d",
      "entry":"Next business day open, no gap cap, no fixed stop, DD20 top3",
      "tests":[
        "Leave one crash event out at a time",
        "Remove the single best-returning stock from each event and leave that capital slot in cash"
      ],
      "summary":summary,
      "leave_one_event_out":loo,
      "remove_best_stock_each_event":trimmed,
      "full":full,
      "caveat":"Only five independent crash events; this is a robustness check, not new independent evidence."
    }
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"summary":summary},ensure_ascii=False,indent=2))

if __name__=="__main__":main()

# trigger
