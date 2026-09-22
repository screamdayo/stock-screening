import json
from pathlib import Path
import numpy as np
import pandas as pd

BATCH=Path("data/batches")
TR=Path("results/reversal_signal_baseline_trades.csv")
OUT=Path("results/selling_climax_position_cap_sweep_fine.json")
DATES=pd.to_datetime(["2018-12-25","2020-03-13","2024-08-05","2025-04-07"])
CAPITALS=[300_000,500_000,1_000_000,1_500_000,2_000_000]
CAP_PCTS=[60,65,70,75,80,85]
HOLD=41
LOT=100
MAX_RANK=5

def load_candidates():
    t=pd.read_csv(TR,dtype={"code":str})
    t=t[t.type=="sharp_drop_reversal"].copy()
    t["signal_date"]=pd.to_datetime(t.signal_date)
    for c in ["ret5_before_pct","dd20_pct","candle_pct","vol_ratio20"]:
        t[c]=pd.to_numeric(t[c],errors="coerce")
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
        d["RAW_O"]=pd.to_numeric(d["O"],errors="coerce")
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
    raw=float(g.RAW_O.iloc[ei]); ao=float(g.RET_O.iloc[ei]); xo=float(g.RET_O.iloc[xi])
    if not(raw>0 and ao>0 and np.isfinite(xo)):return None
    return {
        "entry_date":str(pd.Timestamp(g.Date.iloc[ei]).date()),
        "exit_date":str(pd.Timestamp(g.Date.iloc[xi]).date()),
        "entry_price":raw,
        "lot_cost":raw*LOT,
        "return_pct":(xo/ao-1)*100,
    }

EVENTS={}
for d,g in CANDS.groupby("signal_date"):
    rows=[]
    for _,r in g.sort_values("dd20_rank").iterrows():
        z=trade_info(r.code,d)
        if z:
            rows.append({
                "code":str(r.code),"rank":int(r.dd20_rank),
                "dd20_pct":float(r.dd20_pct),"ret5_pct":float(r.ret5_before_pct),**z
            })
    EVENTS[pd.Timestamp(d)]=rows

def allocate(rows,capital,cap_pct,max_rank):
    remain=float(capital)
    per_name=capital*cap_pct/100
    alloc=[]
    for r in rows:
        if r["rank"]>max_rank: continue
        lots=int(min(remain,per_name)//r["lot_cost"])
        if lots<=0: continue
        cost=lots*r["lot_cost"]
        alloc.append({
            "code":r["code"],"rank":r["rank"],
            "lots":lots,"shares":lots*LOT,
            "entry_price":round(r["entry_price"],2),
            "cost_yen":round(cost),
            "capital_share_pct":round(cost/capital*100,2),
            "return_pct":round(r["return_pct"],3),
        })
        remain-=cost
        if remain<=0: break
    invested=sum(a["cost_yen"] for a in alloc)
    pnl=sum(a["cost_yen"]*a["return_pct"]/100 for a in alloc)
    return {
        "invested_yen":round(invested),
        "cash_yen":round(capital-invested),
        "utilization_pct":round(invested/capital*100,2),
        "portfolio_return_pct":round(pnl/capital*100,3),
        "holdings":alloc,
    }

def summarize(capital,cap_pct,max_rank):
    ev=[]
    for d,rows in EVENTS.items():
        z=allocate(rows,capital,cap_pct,max_rank)
        ev.append({"date":str(d.date()),"available":len(rows),**z})
    s=pd.Series([e["portfolio_return_pct"] for e in ev],dtype=float)
    util=pd.Series([e["utilization_pct"] for e in ev],dtype=float)
    conc=[max([h["capital_share_pct"] for h in e["holdings"]] or [0]) for e in ev]
    return {
        "capital_yen":capital,
        "single_stock_cap_pct":cap_pct,
        "candidate_pool":f"top{max_rank}",
        "event_avg_return_pct":round(float(s.mean()),3),
        "event_median_return_pct":round(float(s.median()),3),
        "worst_event_return_pct":round(float(s.min()),3),
        "best_event_return_pct":round(float(s.max()),3),
        "event_stdev_pct":round(float(s.std(ddof=0)),3),
        "positive_events":int((s>0).sum()),
        "avg_utilization_pct":round(float(util.mean()),2),
        "avg_max_single_stock_share_pct":round(float(pd.Series(conc).mean()),2),
        "events":ev,
    }

def main():
    rows=[]
    for capital in CAPITALS:
        for max_rank in [1,2,3,5]:
            for cap in CAP_PCTS:
                rows.append(summarize(capital,cap,max_rank))

    by_cap={}
    for capital in CAPITALS:
        grp=[r for r in rows if r["capital_yen"]==capital]
        by_cap[str(capital)]=sorted(grp,key=lambda r:(-r["event_avg_return_pct"],-r["worst_event_return_pct"],r["event_stdev_pct"]))

    stable=[]
    for pool in ["top1","top2","top3","top5"]:
        for cap in CAP_PCTS:
            xs=[r for r in rows if r["candidate_pool"]==pool and r["single_stock_cap_pct"]==cap]
            stable.append({
                "candidate_pool":pool,
                "single_stock_cap_pct":cap,
                "mean_of_capital_level_avg_returns_pct":round(float(pd.Series([x["event_avg_return_pct"] for x in xs]).mean()),3),
                "worst_of_all_event_returns_pct":round(float(min(x["worst_event_return_pct"] for x in xs)),3),
                "mean_utilization_pct":round(float(pd.Series([x["avg_utilization_pct"] for x in xs]).mean()),2),
                "mean_stdev_pct":round(float(pd.Series([x["event_stdev_pct"] for x in xs]).mean()),3),
            })
    stable.sort(key=lambda r:(-r["mean_of_capital_level_avg_returns_pct"],-r["worst_of_all_event_returns_pct"],r["mean_stdev_pct"]))

    out={
        "study":"Selling-climax fine DD20-ranked allocation sweep 60-85%",
        "sensor_dates":[str(x.date()) for x in DATES],
        "entry":"Next business-day open",
        "exit":"41 business days after entry, open",
        "ranking":"DD20 deepest first",
        "candidate_rule":"ret5<=-12%, dd20<=-20%, candle>=+2%, volume ratio20>=1.5",
        "capital_levels_yen":CAPITALS,
        "single_stock_caps_pct":CAP_PCTS,
        "candidate_pools":["top1","top2","top3","top5"],
        "lot_size":LOT,
        "allocation":"Scan DD20 rank order. Each stock may use at most X% of starting capital. 100-share lots only; unused cash stays cash.",
        "results":rows,
        "ranking_by_capital":by_cap,
        "cross_capital_stability":stable,
        "caveat":"Only four independent selling-climax events. Fees/slippage/taxes excluded; current-Prime survivor-biased archive."
    }
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({
        "cross_capital_stability":stable[:20],
        "capital_1m":by_cap["1000000"][:12]
    },ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()

# trigger

# fine trigger

# fine trigger 2
