import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

BATCH=Path("data/batches")
TR=Path("results/reversal_signal_baseline_trades.csv")
OUT=Path("results/pinch_exit_fixed_vs_trailing.json")
DATES=pd.to_datetime(["2018-12-27","2020-04-06","2022-03-10","2024-08-06","2025-04-08"])
FIXED_DAYS=list(range(35,46))
ACTIVATE=[15.0,20.0,25.0]
TRAIL=[8.0,10.0,12.0]
MAX_HOLD=60
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

def locate(code,signal_date):
    g=BYCODE.get(str(code))
    if g is None:return None,None,None
    ix=g.index[g.Date==pd.Timestamp(signal_date)]
    if len(ix)==0:return None,None,None
    si=int(ix[0]); ei=si+1
    if ei>=len(g):return None,None,None
    entry=float(g.O.iloc[ei])
    if not(np.isfinite(entry) and entry>0):return None,None,None
    return g,ei,entry

def fixed_trade(code,signal_date,hold):
    g,ei,entry=locate(code,signal_date)
    if g is None:return None
    xi=ei+hold
    if xi>=len(g):return None
    exitp=float(g.O.iloc[xi])
    return {"return_pct":(exitp/entry-1)*100,"exit_reason":f"fixed_{hold}d",
            "exit_day":hold,"exit_date":str(pd.Timestamp(g.Date.iloc[xi]).date())}

def trailing_trade(code,signal_date,activate_pct,trail_pct):
    g,ei,entry=locate(code,signal_date)
    if g is None:return None
    max_i=min(ei+MAX_HOLD,len(g)-1)
    if max_i<=ei:return None
    peak_close=-np.inf
    active=False
    for i in range(ei,max_i):
        close=float(g.C.iloc[i])
        if not np.isfinite(close):continue
        peak_close=max(peak_close,close)
        if not active and (peak_close/entry-1)*100>=activate_pct:
            active=True
        if active and close<=peak_close*(1-trail_pct/100):
            xi=i+1
            if xi>max_i:break
            exitp=float(g.O.iloc[xi])
            return {"return_pct":(exitp/entry-1)*100,
                    "exit_reason":f"trail_a{activate_pct:g}_t{trail_pct:g}",
                    "exit_day":xi-ei,
                    "exit_date":str(pd.Timestamp(g.Date.iloc[xi]).date())}
    exitp=float(g.O.iloc[max_i])
    return {"return_pct":(exitp/entry-1)*100,"exit_reason":f"max_{MAX_HOLD}d",
            "exit_day":max_i-ei,"exit_date":str(pd.Timestamp(g.Date.iloc[max_i]).date())}

def summarize(name,kind,runner,params):
    events=[];allrets=[];days=[]
    for d,g in CANDS.groupby("signal_date"):
        rs=[];detail=[]
        for _,r in g.sort_values("dd20_rank").iterrows():
            z=runner(r.code,d)
            if z is None:continue
            rs.append(z["return_pct"]);days.append(z["exit_day"])
            detail.append({"code":r.code,"rank":int(r.dd20_rank),"return_pct":round(z["return_pct"],3),
                           "exit_day":int(z["exit_day"]),"exit_reason":z["exit_reason"]})
        allrets+=rs
        port=sum(rs)/SLOTS
        events.append({"date":str(pd.Timestamp(d).date()),"executed":len(rs),
                       "portfolio_return_pct":round(port,3),"trades":detail})
    ports=pd.Series([x["portfolio_return_pct"] for x in events],dtype=float)
    rs=pd.Series(allrets,dtype=float)
    return {"name":name,"kind":kind,**params,
            "event_avg_return_pct":round(float(ports.mean()),3),
            "event_median_return_pct":round(float(ports.median()),3),
            "worst_event_return_pct":round(float(ports.min()),3),
            "best_event_return_pct":round(float(ports.max()),3),
            "event_stdev_pct":round(float(ports.std(ddof=0)),3),
            "positive_events":int((ports>0).sum()),
            "trade_win_pct":round(float((rs>0).mean()*100),2),
            "trade_avg_return_pct":round(float(rs.mean()),3),
            "trade_median_return_pct":round(float(rs.median()),3),
            "avg_exit_day":round(float(pd.Series(days).mean()),2),
            "events":events}

def job_fixed(h):
    return summarize(f"fixed_{h}d","fixed",lambda c,d:fixed_trade(c,d,h),{"hold_days":h})

def job_trailing(pair):
    a,t=pair
    return summarize(f"trail_a{a:g}_t{t:g}","trailing",
                     lambda c,d:trailing_trade(c,d,a,t),
                     {"activate_profit_pct":a,"trail_drawdown_pct":t,"max_hold_days":MAX_HOLD})

def main():
    with ThreadPoolExecutor(max_workers=16) as ex:
        fixed=list(ex.map(job_fixed,FIXED_DAYS))
        trailing=list(ex.map(job_trailing,[(a,t) for a in ACTIVATE for t in TRAIL]))
    rows=fixed+trailing
    ranked=sorted(rows,key=lambda x:(-x["event_avg_return_pct"],-x["worst_event_return_pct"],x["event_stdev_pct"]))
    # Balanced view favors a strong worst event first, then average return.
    balanced=sorted(rows,key=lambda x:(-x["worst_event_return_pct"],-x["event_avg_return_pct"],x["event_stdev_pct"]))
    out={
      "study":"Pinch TOP3 exit: fixed 35-45d vs close-based profit-protection trailing",
      "entry":"Next business day open, no gap cap, no fixed stop, DD20 top3",
      "trailing_rule":"Activate after highest close reaches +15/+20/+25% from entry. Thereafter, if a daily close falls 8/10/12% from the highest close, exit at next business day open. Otherwise force exit at day 60 open.",
      "fixed_days":FIXED_DAYS,
      "trailing_activation_pct":ACTIVATE,
      "trailing_drawdown_pct":TRAIL,
      "best_by_event_average":ranked[:10],
      "best_by_worst_event":balanced[:10],
      "all_results":rows,
      "caveat":"Only five independent crash events. Close-based trailing avoids intraday sequencing assumptions; no fees/slippage."
    }
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"best_avg":ranked[:10],"best_worst":balanced[:10]},ensure_ascii=False,indent=2))
if __name__=="__main__":main()

# trigger
