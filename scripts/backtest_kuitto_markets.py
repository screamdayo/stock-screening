"""Research current Kuitto rules across markets; current-master survivor universe."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from strategies import kuitto_pullback_auto as strategy
from kuitto_order_policy import SCORE_GAP_CAP_PCT,preset_stop
from forward_test import _strict_gakutto

def summarize(d):
    if d.empty: return {"n":0}
    r=d.pnl_pct;loss=-r[r<0].sum()
    return {"n":len(d),"mean_pct":float(r.mean()),"median_pct":float(r.median()),
            "win_pct":float((r>0).mean()*100),"pf":float(r[r>0].sum()/loss) if loss else None,
            "worst_pct":float(r.min()),"best_pct":float(r.max()),
            "mean_hold_days":float(d.hold_days.mean())}

def signal_mask(g):
    m=g.MA_SHORT
    decline=(m.shift(1)/m.shift(6)-1)*100
    gap=(m/g.MA_LONG-1)*100
    bull=(g.C/g.O-1)*100
    return (decline.between(-5.5,-2)&gap.between(-5,0)&bull.between(1.5,3.5)
            &(m>m.shift(1))&(m.shift(1)<=m.shift(2))&(m.shift(2)<=m.shift(3))
            &((g.C/m-1)*100<=4)&(g.Vo.shift(1)>0)&(g.Vo/g.Vo.shift(1)>=1.25)
            &(g.AVG_TURNOVER_20>=500_000_000)&(g.O>0)&(g.index>=25))

def replay(g,idx,f):
    ent=idx+1
    # Fixed maturity: require the full maximum-holding observation window for every signal.
    if ent+15>=len(g): return None
    entry=float(g.O.iloc[ent]);close=float(g.C.iloc[idx])
    if not entry>0 or not close>0:return None
    score=int(f["runner_score"]);cap=SCORE_GAP_CAP_PCT[score]
    gap=round((entry/close-1)*100,3)
    row={"entry_date":str(g.Date.iloc[ent].date()),"entry_open":entry,"runner_score":score,
         "next_open_gap_pct":gap,"gap_pass":bool(cap is None or gap<=cap)}
    if not row["gap_pass"]:return row
    # Apply notifier tick rounding at the historical unadjusted price, then convert back.
    raw_close=float(g.raw_C.iloc[idx])
    scale=close/raw_close
    stop=preset_stop(raw_close,score)[2]*scale
    pending=False
    for j in range(ent,ent+15):
        r=g.iloc[j]
        if pending: px=float(r.O);reason="strict_gakutto_next_open";break
        if r.O<=stop: px=float(r.O);reason="stop_gap_open";break
        if r.L<=stop: px=stop;reason="stop_loss";break
        if j<ent+14 and _strict_gakutto(g,j):pending=True
    else:
        j=ent+15;px=float(g.O.iloc[j]);reason="max15_next_open"
    row.update(exit_date=str(g.Date.iloc[j].date()),pnl_pct=(px/entry-1)*100,
               hold_days=j-ent+1,exit_reason=reason,effective_stop_pct=(stop/entry-1)*100)
    return row

def main(root):
    master=pd.read_parquet(root/"master.parquet");master.Code=master.Code.astype(str)
    meta=master.set_index("Code").to_dict("index")
    updates=pd.read_parquet(root/"daily.parquet");updates.Code=updates.Code.astype(str)
    rows=[];counts={};validated={};periods=[];stale=[]
    for path in sorted((root/"archive_batches").glob("*.parquet")):
        raw=pd.read_parquet(path);raw.Code=raw.Code.astype(str)
        codes=set(raw.Code)
        raw=pd.concat([raw,updates[updates.Code.isin(codes)]],ignore_index=True)
        raw.Date=pd.to_datetime(raw.Date)
        raw=raw.drop_duplicates(["Code","Date"],keep="last")
        raw["raw_C"]=pd.to_numeric(raw.C,errors="coerce")
        for c in ["O","H","L","C","Vo"]:
            adj="Adj"+c
            if adj in raw:raw[c]=raw[adj].where(raw[adj].notna(),raw[c])
        for code,part in raw.groupby("Code"):
            if code not in meta:continue
            market=meta[code]["MktNm"]
            g=strategy._prepare(part);g["MA5"]=g.MA_SHORT
            periods.append((str(g.Date.min().date()),str(g.Date.max().date())))
            if g.Date.iloc[-1]<pd.Timestamp("2026-10-22"):stale.append(code)
            indexes=list(g.index[signal_mask(g)])
            if market not in validated:
                full=[i for i in range(25,len(g)) if (f:=strategy._features(g,i,apply_liquidity=False)) and f["avg_turnover_20"]>=500_000_000]
                assert indexes==full,(market,code,indexes,full)
                validated[market]=code
            for idx in indexes:
                f=strategy._features(g,idx,apply_liquidity=False)
                assert f is not None and f["avg_turnover_20"]>=500_000_000
                c=counts.setdefault(market,{"signals":0,"immature":0,"gap_rejected":0,"accepted":0})
                c["signals"]+=1
                z=replay(g,idx,f)
                if z is None:c["immature"]+=1;continue
                if not z["gap_pass"]:c["gap_rejected"]+=1;continue
                c["accepted"]+=1
                rows.append({"code":code,"name":meta[code]["CoName"],"market":market,
                             "signal_date":str(g.Date.iloc[idx].date()),**f,**z})
        print(path.name,"completed trades",len(rows),flush=True)
    d=pd.DataFrame(rows);d["year"]=d.entry_date.str[:4]
    markets=["プライム","スタンダード","グロース"]
    result={"method":{
        "as_of":"2026-10-22","master_date":str(pd.to_datetime(master.Date).max().date()),
        "membership":"Current 2026-09-24 master applied to all history; survivor and historical-market-classification bias.",
        "signal":"Production kuitto_pullback_auto unchanged; volume >=1.25x, turnover20>=500m throughout history.",
        "entry":"Next session open with existing score caps; gap rounded to 3 decimals as forward_test.",
        "exit":"Score0 signal-close -5%; score1-4 tick-floored cap -7%, historical raw-price tick rounding. Strict gakutto next open, max15 next open.",
        "maturity":"Require 16 available sessions from entry for every signal, including early exits.",
        "scope":"Independent equal-weight signal returns; overlapping trades allowed; no capital or daily rank constraints.",
        "costs":"No fees, taxes, spread or slippage deducted; stop threshold execution assumed except opening gaps.",
        "validation":"Vector mask matches full production _features scan for one full-history code per market; every detected signal rechecked.",
        "validation_codes":validated},
        "coverage":{"min_bar_date":min(a for a,b in periods),"max_bar_date":max(b for a,b in periods),
                    "first_entry":d.entry_date.min(),"last_entry":d.entry_date.max(),
                    "latest_data_missing_codes":len(stale),"counts":counts},
        "overall":{m:summarize(d[d.market==m]) for m in markets},
        "since_2021":{m:summarize(d[(d.market==m)&(d.entry_date>="2021-01-01")]) for m in markets},
        "yearly":{y:{m:summarize(d[(d.year==y)&(d.market==m)]) for m in markets} for y in sorted(d.year.unique())},
        "by_score":{m:{str(s):summarize(d[(d.market==m)&(d.runner_score==s)]) for s in range(5)} for m in markets},
        "winner":{m:summarize(d[(d.market==m)&d.winner_filter]) for m in markets},
        "combined_standard_growth":summarize(d[d.market!="プライム"])}
    out=Path(__file__).resolve().parents[1]/"results"
    d.to_csv(out/"kuitto_market_performance_20261022_trades.csv",index=False)
    (out/"kuitto_market_performance_20261022.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data-root",type=Path,required=True);main(p.parse_args().data_root)
