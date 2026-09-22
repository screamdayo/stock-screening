import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

BATCH=Path("data/batches")
TR=Path("results/reversal_signal_baseline_trades.csv")
OUT=Path("results/pinch_entry_exit_grid.json")
DATES=pd.to_datetime(["2018-12-27","2020-04-06","2022-03-10","2024-08-06","2025-04-08"])
GAPS=[None,3.0,5.0,7.0]
STOPS=[None,3.0,5.0,7.0]
HOLDS=[10,15,20]
POLICIES=["fixed_top3","backfill_to3"]
SLOTS=3

def load_candidates():
    t=pd.read_csv(TR,dtype={"code":str})
    t=t[(t.type=="sharp_drop_reversal")].copy()
    t["signal_date"]=pd.to_datetime(t.signal_date)
    t=t[t.signal_date.isin(DATES)]
    t=t[(t.ret5_before_pct<=-12)&(t.dd20_pct<=-20)&(t.candle_pct>=2)&(t.vol_ratio20>=1.5)].copy()
    t=t.sort_values(["signal_date","dd20_pct","code"]).reset_index(drop=True)
    t["dd20_rank"]=t.groupby("signal_date").cumcount()+1
    return t

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
                d[raw]=pd.to_numeric(d[adj],errors="coerce").where(d[adj].notna(),pd.to_numeric(d.get(raw),errors="coerce"))
            else:
                d[raw]=pd.to_numeric(d[raw],errors="coerce")
        d["Date"]=pd.to_datetime(d.Date)
        fs.append(d[["Code","Date","O","H","L","C"]])
    if not fs: raise RuntimeError("No archive prices")
    x=pd.concat(fs,ignore_index=True).dropna(subset=["O","H","L","C"])
    return x.drop_duplicates(["Code","Date"],keep="last").sort_values(["Code","Date"])

PX=load_prices()
BYCODE={c:g.sort_values("Date").reset_index(drop=True) for c,g in PX.groupby("Code")}

def trade_path(code,signal_date,gap_max,stop_pct,hold):
    g=BYCODE.get(str(code))
    if g is None:return None
    ix=g.index[g.Date==pd.Timestamp(signal_date)]
    if len(ix)==0:return None
    si=int(ix[0]); ei=si+1; xi=ei+hold
    if xi>=len(g):return None
    signal_close=float(g.C.iloc[si]); entry=float(g.O.iloc[ei])
    if not(np.isfinite(signal_close) and np.isfinite(entry) and signal_close>0 and entry>0):return None
    gap=(entry/signal_close-1)*100
    if gap_max is not None and gap>gap_max:
        return {"eligible":False,"gap_pct":gap}
    stop_price=None if stop_pct is None else entry*(1-stop_pct/100)
    exit_price=None;exit_date=None;reason=None;mae=0.0
    # Stop checks from entry day through day before fixed exit-open.
    for i in range(ei,xi):
        r=g.iloc[i]
        low=float(r.L); op=float(r.O)
        if np.isfinite(low):
            mae=min(mae,(low/entry-1)*100)
        if stop_price is not None:
            if op<=stop_price:
                exit_price=op;exit_date=r.Date;reason="gap_stop";break
            if low<=stop_price:
                exit_price=stop_price;exit_date=r.Date;reason="stop";break
    if exit_price is None:
        exit_price=float(g.O.iloc[xi]);exit_date=g.Date.iloc[xi];reason=f"fixed_{hold}d"
    ret=(exit_price/entry-1)*100
    return {"eligible":True,"entry_date":str(pd.Timestamp(g.Date.iloc[ei]).date()),"gap_pct":gap,
            "return_pct":ret,"mae_pct":mae,"exit_date":str(pd.Timestamp(exit_date).date()),"reason":reason}

def evaluate(job):
    gap,stop,hold,policy=job
    events=[];allrets=[];allmae=[];stop_count=0;gap_skips=0
    for d,grp in CANDS.groupby("signal_date"):
        ranked=grp.sort_values(["dd20_pct","code"])
        pool=ranked.head(SLOTS) if policy=="fixed_top3" else ranked
        selected=[]
        for _,r in pool.iterrows():
            z=trade_path(r.code,d,gap,stop,hold)
            if z is None: continue
            if not z["eligible"]:
                gap_skips+=1
                continue
            z.update({"code":r.code,"rank":int(r.dd20_rank)})
            selected.append(z)
            if len(selected)>=SLOTS:break
        # fixed_top3 leaves skipped slot as cash; backfill can refill from lower ranks.
        slot_returns=[x["return_pct"] for x in selected] + [0.0]*(SLOTS-len(selected))
        port=sum(slot_returns)/SLOTS
        allrets += [x["return_pct"] for x in selected]
        allmae += [x["mae_pct"] for x in selected]
        stop_count += sum(1 for x in selected if x["reason"] in ("stop","gap_stop"))
        events.append({"date":str(pd.Timestamp(d).date()),"executed":len(selected),
                       "portfolio_return_pct":round(port,3),
                       "selected":[{"code":x["code"],"rank":x["rank"],"gap_pct":round(x["gap_pct"],3),
                                    "return_pct":round(x["return_pct"],3),"mae_pct":round(x["mae_pct"],3),"exit_reason":x["reason"]} for x in selected]})
    ports=pd.Series([x["portfolio_return_pct"] for x in events],dtype=float)
    rs=pd.Series(allrets,dtype=float)
    maes=pd.Series(allmae,dtype=float)
    return {
      "gap_max_pct":"none" if gap is None else gap,
      "stop_loss_pct":"none" if stop is None else stop,
      "hold_days":hold,"entry_policy":policy,
      "event_avg_return_pct":round(float(ports.mean()),3),
      "event_median_return_pct":round(float(ports.median()),3),
      "worst_event_return_pct":round(float(ports.min()),3),
      "best_event_return_pct":round(float(ports.max()),3),
      "event_stdev_pct":round(float(ports.std(ddof=0)),3),
      "positive_events":int((ports>0).sum()),
      "executed_trades":int(len(rs)),"trade_win_pct":round(float((rs>0).mean()*100),2) if len(rs) else None,
      "trade_avg_return_pct":round(float(rs.mean()),3) if len(rs) else None,
      "trade_median_return_pct":round(float(rs.median()),3) if len(rs) else None,
      "avg_mae_pct":round(float(maes.mean()),3) if len(maes) else None,
      "worst_mae_pct":round(float(maes.min()),3) if len(maes) else None,
      "stop_count":int(stop_count),"gap_skip_count":int(gap_skips),
      "events":events
    }

def main():
    jobs=[(g,s,h,p) for g in GAPS for s in STOPS for h in HOLDS for p in POLICIES]
    with ThreadPoolExecutor(max_workers=16) as ex:
        rows=list(ex.map(evaluate,jobs))
    # Stable shortlist: event average first, then worst event, then lower stdev.
    ranked=sorted(rows,key=lambda x:(-x["event_avg_return_pct"],-x["worst_event_return_pct"],x["event_stdev_pct"]))
    out={
      "study":"Pinch TOP3 entry/exit completion grid",
      "assumption":"Signal close -> next open entry. Same total capital divided into 3 fixed slots. If fixed_top3 gap-filters a name, that slot remains cash; backfill_to3 promotes lower DD20-ranked candidates. Stops use adjusted intraday lows; open below stop exits at open. Fixed exit uses the open hold_days after entry.",
      "grid_size":len(rows),
      "top10":ranked[:10],
      "all_results":rows,
      "candidate_dates":[str(x.date()) for x in DATES],
      "caveat":"Only five independent crash events; current-Prime survivor-biased archive and no fees/slippage."
    }
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"top10":ranked[:10]},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
