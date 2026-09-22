import json
from pathlib import Path
import numpy as np
import pandas as pd

BATCH=Path("data/batches")
TR=Path("results/reversal_signal_baseline_trades.csv")
OUT=Path("results/pinch_real_capital_allocation.json")
DATES=pd.to_datetime(["2018-12-27","2020-04-06","2022-03-10","2024-08-06","2025-04-08"])
CAPITALS=[300_000,500_000,1_000_000,1_500_000,2_000_000]
HOLD=41
LOT=100
MAX_RANK=5

def load_candidates():
    t=pd.read_csv(TR,dtype={"code":str})
    t=t[t.type=="sharp_drop_reversal"].copy()
    t["signal_date"]=pd.to_datetime(t.signal_date)
    t=t[t.signal_date.isin(DATES)]
    t=t[(t.ret5_before_pct<=-12)&(t.dd20_pct<=-20)&(t.candle_pct>=2)&(t.vol_ratio20>=1.5)].copy()
    t=t.sort_values(["signal_date","dd20_pct","code"]).reset_index(drop=True)
    t["dd20_rank"]=t.groupby("signal_date").cumcount()+1
    return t[t.dd20_rank<=MAX_RANK].copy()

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
        d["Date"]=pd.to_datetime(d.Date)
        # Raw open is required for actual historical 100-share affordability.
        d["RAW_O"]=pd.to_numeric(d["O"],errors="coerce")
        # Adjusted opens are required for split-adjusted holding return.
        if "AdjO" in d.columns:
            adj=pd.to_numeric(d["AdjO"],errors="coerce")
            d["RET_O"]=adj.where(adj.notna(),d["RAW_O"])
        else:
            d["RET_O"]=d["RAW_O"]
        fs.append(d[["Code","Date","RAW_O","RET_O"]])
    if not fs: raise RuntimeError("No archive prices")
    x=pd.concat(fs,ignore_index=True).dropna(subset=["RAW_O","RET_O"])
    return x.drop_duplicates(["Code","Date"],keep="last").sort_values(["Code","Date"])

PX=load_prices()
BYCODE={c:g.sort_values("Date").reset_index(drop=True) for c,g in PX.groupby("Code")}

def trade_info(code,signal_date):
    g=BYCODE.get(str(code))
    if g is None:return None
    ix=g.index[g.Date==pd.Timestamp(signal_date)]
    if len(ix)==0:return None
    si=int(ix[0]); ei=si+1; xi=ei+HOLD
    if xi>=len(g):return None
    raw_entry=float(g.RAW_O.iloc[ei])
    adj_entry=float(g.RET_O.iloc[ei])
    adj_exit=float(g.RET_O.iloc[xi])
    if not(raw_entry>0 and adj_entry>0 and np.isfinite(adj_exit)):return None
    return {
      "entry_date":str(pd.Timestamp(g.Date.iloc[ei]).date()),
      "exit_date":str(pd.Timestamp(g.Date.iloc[xi]).date()),
      "raw_entry":raw_entry,
      "lot_cost":raw_entry*LOT,
      "return_pct":(adj_exit/adj_entry-1)*100,
    }

EVENTS={}
for d,g in CANDS.groupby("signal_date"):
    rows=[]
    for _,r in g.sort_values("dd20_rank").iterrows():
        z=trade_info(r.code,d)
        if z:
            rows.append({"code":str(r.code),"rank":int(r.dd20_rank),"dd20_pct":float(r.dd20_pct),**z})
    EVENTS[pd.Timestamp(d)]=rows

def finalize(capital,allocs):
    invested=sum(a["cost"] for a in allocs)
    pnl=sum(a["cost"]*a["return_pct"]/100 for a in allocs)
    end=capital+pnl
    return {
      "invested_yen":round(invested),
      "cash_yen":round(capital-invested),
      "utilization_pct":round(invested/capital*100,2),
      "portfolio_return_pct":round((end/capital-1)*100,3),
      "end_value_yen":round(end),
      "holdings":allocs,
    }

def buy_lots(row,lots):
    return {
      "code":row["code"],"rank":row["rank"],"entry_price":round(row["raw_entry"],2),
      "lots":int(lots),"shares":int(lots*LOT),"cost":round(row["lot_cost"]*lots),
      "return_pct":round(row["return_pct"],3)
    }

def equal_target_top3(rows,capital):
    target=capital/3
    out=[]
    for r in rows:
        if r["rank"]>3:continue
        lots=int(target//r["lot_cost"])
        if lots>0:out.append(buy_lots(r,lots))
    return finalize(capital,out)

def rank_priority_top3(rows,capital):
    # Buy as much as possible from rank 1, then rank 2, then rank 3.
    remain=capital;out=[]
    for r in rows:
        if r["rank"]>3:continue
        lots=int(remain//r["lot_cost"])
        if lots>0:
            a=buy_lots(r,lots);out.append(a);remain-=a["cost"]
    return finalize(capital,out)

def one_lot_each_then_rank(rows,capital):
    top=[r for r in rows if r["rank"]<=3]
    remain=capital;lots={r["code"]:0 for r in top}
    # First pass: one lot each, preserving rank, but skip unaffordable names.
    for r in top:
        if r["lot_cost"]<=remain:
            lots[r["code"]]+=1;remain-=r["lot_cost"]
    # Then recycle leftover in rank order until no additional lot fits.
    while True:
        bought=False
        for r in top:
            if r["lot_cost"]<=remain:
                lots[r["code"]]+=1;remain-=r["lot_cost"];bought=True
        if not bought:break
    out=[buy_lots(r,lots[r["code"]]) for r in top if lots[r["code"]]>0]
    return finalize(capital,out)

def cap50_backfill_top5(rows,capital):
    # Max 50% of starting capital per stock; ranks 4-5 may backfill if top3 cannot use capital.
    remain=capital;out=[]
    per_cap=capital*0.50
    for r in rows:
        lots=int(min(remain,per_cap)//r["lot_cost"])
        if lots>0:
            a=buy_lots(r,lots);out.append(a);remain-=a["cost"]
    return finalize(capital,out)

POLICIES={
  "equal_target_top3":equal_target_top3,
  "rank_priority_top3":rank_priority_top3,
  "one_lot_each_then_rank_top3":one_lot_each_then_rank,
  "cap50_backfill_top5":cap50_backfill_top5,
}

def summarize(policy,capital):
    ev=[]
    for d,rows in EVENTS.items():
        z=POLICIES[policy](rows,capital)
        ev.append({"date":str(d.date()),"available":len(rows),**z})
    s=pd.Series([x["portfolio_return_pct"] for x in ev],dtype=float)
    util=pd.Series([x["utilization_pct"] for x in ev],dtype=float)
    return {
      "policy":policy,"capital_yen":capital,
      "event_avg_return_pct":round(float(s.mean()),3),
      "event_median_return_pct":round(float(s.median()),3),
      "worst_event_return_pct":round(float(s.min()),3),
      "best_event_return_pct":round(float(s.max()),3),
      "event_stdev_pct":round(float(s.std(ddof=0)),3),
      "positive_events":int((s>0).sum()),
      "avg_utilization_pct":round(float(util.mean()),2),
      "events":ev,
    }

def main():
    rows=[summarize(p,c) for c in CAPITALS for p in POLICIES]
    by_cap={}
    for c in CAPITALS:
        x=[r for r in rows if r["capital_yen"]==c]
        by_cap[str(c)]=sorted(x,key=lambda r:(-r["event_avg_return_pct"],-r["worst_event_return_pct"],r["event_stdev_pct"]))
    out={
      "study":"Pinch sensor real-capital allocation with 100-share lots",
      "entry":"DD20-ranked candidates, next business-day open",
      "exit":"41 business days after entry, open",
      "lot_size":LOT,
      "capital_levels_yen":CAPITALS,
      "policies":{
        "equal_target_top3":"Capital/3 target for each of ranks 1-3; buy whole 100-share lots; unused target stays cash.",
        "rank_priority_top3":"Spend on rank 1 first, then rank 2, then rank 3.",
        "one_lot_each_then_rank_top3":"Try to secure one lot in ranks 1-3 in rank order, then add extra lots in rank order while cash allows.",
        "cap50_backfill_top5":"No stock may use more than 50% of starting capital; scan ranks 1-5 so ranks 4-5 can backfill."
      },
      "price_method":"Historical raw next-open price determines 100-share affordability. Split-adjusted open-to-open return determines P/L.",
      "results":rows,
      "ranking_by_capital":by_cap,
      "caveat":"Only five independent crash events. Fees/slippage/taxes excluded; 100-share standard unit is appropriate for these post-Oct-2018 events."
    }
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    compact={k:[{a:r[a] for a in ["policy","event_avg_return_pct","worst_event_return_pct","event_stdev_pct","avg_utilization_pct"]} for r in v] for k,v in by_cap.items()}
    print(json.dumps(compact,ensure_ascii=False,indent=2))

if __name__=="__main__":main()

# trigger
