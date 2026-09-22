import json
from pathlib import Path
import numpy as np
import pandas as pd

BATCH=Path("data/batches")
TR=Path("results/reversal_signal_baseline_trades.csv")
OUT=Path("results/pinch_position_cap_sweep.json")
DATES=pd.to_datetime(["2018-12-27","2020-04-06","2022-03-10","2024-08-06","2025-04-08"])
CAPITALS=[300_000,500_000,1_000_000,1_500_000,2_000_000]
CAP_PCTS=[40,50,60,70,80,90,100]
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
    return {"entry_price":raw,"lot_cost":raw*LOT,"return_pct":(xo/ao-1)*100}

EVENTS={}
for d,g in CANDS.groupby("signal_date"):
    rows=[]
    for _,r in g.sort_values("dd20_rank").iterrows():
        z=trade_info(r.code,d)
        if z: rows.append({"code":str(r.code),"rank":int(r.dd20_rank),**z})
    EVENTS[pd.Timestamp(d)]=rows

def allocate(rows,capital,cap_pct,max_rank):
    remain=float(capital)
    cap_yen=capital*cap_pct/100
    alloc=[]
    for r in rows:
        if r["rank"]>max_rank: continue
        budget=min(remain,cap_yen)
        lots=int(budget//r["lot_cost"])
        if lots<=0: continue
        cost=lots*r["lot_cost"]
        alloc.append({
            "code":r["code"],"rank":r["rank"],"lots":lots,"shares":lots*LOT,
            "entry_price":round(r["entry_price"],2),"cost_yen":round(cost),
            "capital_share_pct":round(cost/capital*100,2),
            "return_pct":round(r["return_pct"],3),
        })
        remain-=cost
        if remain<=0: break
    invested=sum(x["cost_yen"] for x in alloc)
    pnl=sum(x["cost_yen"]*x["return_pct"]/100 for x in alloc)
    return {
        "portfolio_return_pct":round(pnl/capital*100,3),
        "invested_yen":round(invested),
        "cash_yen":round(capital-invested),
        "utilization_pct":round(invested/capital*100,2),
        "holdings":alloc,
    }

def summarize(capital,cap_pct,max_rank):
    ev=[]
    for d,rows in EVENTS.items():
        z=allocate(rows,capital,cap_pct,max_rank)
        ev.append({"date":str(d.date()),**z})
    s=pd.Series([x["portfolio_return_pct"] for x in ev],dtype=float)
    util=pd.Series([x["utilization_pct"] for x in ev],dtype=float)
    concentration=[]
    for e in ev:
        concentration.append(max([h["capital_share_pct"] for h in e["holdings"]] or [0]))
    return {
        "capital_yen":capital,"single_stock_cap_pct":cap_pct,"candidate_pool":f"top{max_rank}",
        "event_avg_return_pct":round(float(s.mean()),3),
        "event_median_return_pct":round(float(s.median()),3),
        "worst_event_return_pct":round(float(s.min()),3),
        "event_stdev_pct":round(float(s.std(ddof=0)),3),
        "positive_events":int((s>0).sum()),
        "avg_utilization_pct":round(float(util.mean()),2),
        "avg_max_single_stock_share_pct":round(float(pd.Series(concentration).mean()),2),
        "events":ev,
    }

def main():
    rows=[]
    for capital in CAPITALS:
        for max_rank in [3,5]:
            for cap in CAP_PCTS:
                rows.append(summarize(capital,cap,max_rank))

    by_capital={}
    for capital in CAPITALS:
        group=[r for r in rows if r["capital_yen"]==capital]
        by_capital[str(capital)]=sorted(
            group,key=lambda r:(-r["event_avg_return_pct"],-r["worst_event_return_pct"],r["event_stdev_pct"])
        )

    # Cross-cap stability: average each cap/pool across the five capital levels.
    stable=[]
    for pool in ["top3","top5"]:
        for cap in CAP_PCTS:
            xs=[r for r in rows if r["candidate_pool"]==pool and r["single_stock_cap_pct"]==cap]
            stable.append({
                "candidate_pool":pool,"single_stock_cap_pct":cap,
                "mean_of_capital_level_avg_returns_pct":round(float(pd.Series([x["event_avg_return_pct"] for x in xs]).mean()),3),
                "worst_of_all_event_returns_pct":round(float(min(x["worst_event_return_pct"] for x in xs)),3),
                "mean_utilization_pct":round(float(pd.Series([x["avg_utilization_pct"] for x in xs]).mean()),2),
                "mean_stdev_pct":round(float(pd.Series([x["event_stdev_pct"] for x in xs]).mean()),3),
            })
    stable.sort(key=lambda r:(-r["mean_of_capital_level_avg_returns_pct"],-r["worst_of_all_event_returns_pct"],r["mean_stdev_pct"]))

    out={
        "study":"Pinch real-capital single-stock cap sweep",
        "entry":"DD20 rank order, next business-day open, 100-share lots",
        "exit":"41 business days after entry, open",
        "capital_levels_yen":CAPITALS,
        "single_stock_caps_pct":CAP_PCTS,
        "candidate_pools":["top3","top5"],
        "allocation":"Scan DD20 rank order; each stock may use at most X% of starting capital. Unused cash remains cash.",
        "results":rows,
        "ranking_by_capital":by_capital,
        "cross_capital_stability":stable,
        "caveat":"Only five independent crash events. Fees/slippage/taxes excluded."
    }
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"cross_capital_stability":stable},ensure_ascii=False,indent=2))

if __name__=="__main__":main()

# trigger
