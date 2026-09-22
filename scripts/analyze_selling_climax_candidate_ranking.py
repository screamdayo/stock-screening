import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

BATCH=Path("data/batches")
TR=Path("results/reversal_signal_baseline_trades.csv")
OUT=Path("results/selling_climax_candidate_ranking.json")
DATES=pd.to_datetime(["2018-12-25","2020-03-13","2024-08-05","2025-04-07"])
TOPKS=[1,3,5]
HOLDS=[5,10,15,20,41]

def met(vals):
    s=pd.to_numeric(pd.Series(vals),errors="coerce").dropna()
    if s.empty:return {"n":0}
    neg=s[s<0].sum()
    return {
        "n":int(len(s)),
        "win":round(float((s>0).mean()*100),2),
        "avg":round(float(s.mean()),3),
        "med":round(float(s.median()),3),
        "pf":None if neg==0 else round(float(s[s>0].sum()/abs(neg)),3),
    }

def load_candidates():
    t=pd.read_csv(TR,dtype={"code":str})
    t=t[t.type=="sharp_drop_reversal"].copy()
    t["signal_date"]=pd.to_datetime(t.signal_date)
    for c in ["ret5_before_pct","dd20_pct","candle_pct","vol_ratio20"]:
        t[c]=pd.to_numeric(t[c],errors="coerce")
    t=t[t.signal_date.isin(DATES)]
    t=t[(t.ret5_before_pct<=-12)&(t.dd20_pct<=-20)&(t.candle_pct>=2)&(t.vol_ratio20>=1.5)].copy()
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
        d["Date"]=pd.to_datetime(d.Date)
        if "AdjO" in d.columns:
            d["O2"]=pd.to_numeric(d["AdjO"],errors="coerce").where(d["AdjO"].notna(),pd.to_numeric(d["O"],errors="coerce"))
        else:
            d["O2"]=pd.to_numeric(d["O"],errors="coerce")
        fs.append(d[["Code","Date","O2"]])
    if not fs: raise RuntimeError("No archive prices")
    x=pd.concat(fs,ignore_index=True).dropna(subset=["O2"])
    return x.drop_duplicates(["Code","Date"],keep="last").sort_values(["Code","Date"])

PX=load_prices()
BYCODE={c:g.sort_values("Date").reset_index(drop=True) for c,g in PX.groupby("Code")}

def returns_for(code,signal_date):
    g=BYCODE.get(str(code))
    if g is None:return None
    ix=g.index[g.Date==pd.Timestamp(signal_date)]
    if len(ix)==0:return None
    si=int(ix[0]); ei=si+1
    if ei>=len(g):return None
    entry=float(g.O2.iloc[ei])
    if not(np.isfinite(entry) and entry>0):return None
    out={"entry_date":str(pd.Timestamp(g.Date.iloc[ei]).date())}
    for h in HOLDS:
        xi=ei+h
        if xi>=len(g):
            out[f"r{h}"]=np.nan
        else:
            xo=float(g.O2.iloc[xi])
            out[f"r{h}"]=(xo/entry-1)*100 if np.isfinite(xo) else np.nan
    return out

def rank_group(g,mode):
    g=g.copy()
    if mode=="deep_dd20":
        return g.sort_values(["dd20_pct","code"])
    if mode=="deep_ret5":
        return g.sort_values(["ret5_before_pct","code"])
    if mode=="high_volume":
        return g.sort_values(["vol_ratio20","code"],ascending=[False,True])
    if mode=="strong_candle":
        return g.sort_values(["candle_pct","code"],ascending=[False,True])
    if mode=="composite":
        # higher score = stronger crash/reversal combination
        g["score"]=(
            g["dd20_pct"].rank(pct=True,ascending=False)+
            g["ret5_before_pct"].rank(pct=True,ascending=False)+
            g["vol_ratio20"].rank(pct=True,ascending=True)+
            g["candle_pct"].rank(pct=True,ascending=True)
        )
        return g.sort_values(["score","code"],ascending=[False,True])
    raise ValueError(mode)

def evaluate(job):
    mode,k=job
    picked=[]
    events=[]
    for d,g in CANDS.groupby("signal_date"):
        rg=rank_group(g,mode).head(k)
        erows=[]
        for _,r in rg.iterrows():
            z=returns_for(r.code,d)
            if z is None: continue
            row={
                "date":d,"code":str(r.code),"ret5":float(r.ret5_before_pct),
                "dd20":float(r.dd20_pct),"candle":float(r.candle_pct),"vol":float(r.vol_ratio20),
                **z
            }
            picked.append(row); erows.append(row)
        ev={"date":str(pd.Timestamp(d).date()),"available":int(len(g)),"picked":len(erows)}
        for h in HOLDS:
            s=pd.Series([x[f"r{h}"] for x in erows],dtype=float).dropna()
            ev[f"avg{h}"]=round(float(s.mean()),3) if len(s) else None
        events.append(ev)
    df=pd.DataFrame(picked)
    out={"ranker":mode,"top_k":k,"events":events}
    for h in HOLDS:
        out[f"r{h}"]=met(df[f"r{h}"]) if not df.empty else {"n":0}
        vals=[e[f"avg{h}"] for e in events if e[f"avg{h}"] is not None]
        out[f"event_avg{h}"]=round(float(pd.Series(vals).mean()),3) if vals else None
        out[f"worst_event{h}"]=round(float(min(vals)),3) if vals else None
        out[f"positive_events{h}"]=int(sum(v>0 for v in vals))
    return out

def main():
    modes=["deep_dd20","deep_ret5","high_volume","strong_candle","composite"]
    jobs=[(m,k) for m in modes for k in TOPKS]
    with ThreadPoolExecutor(max_workers=min(15,len(jobs))) as ex:
        rows=list(ex.map(evaluate,jobs))
    ranked15=sorted(rows,key=lambda x:(-(x["event_avg15"] or -999),-(x["worst_event15"] or -999)))
    ranked41=sorted(rows,key=lambda x:(-(x["event_avg41"] or -999),-(x["worst_event41"] or -999)))
    out={
        "study":"Selling-climax candidate ranking",
        "sensor_dates":[str(x.date()) for x in DATES],
        "sensor_rule":"TOPIX ret1<=-2%, DD20<=-15%, strong-reversal share>=10% (observed production-candidate dates)",
        "candidate_rule":"ret5<=-12%, dd20<=-20%, candle>=+2%, volume ratio20>=1.5",
        "entry":"next business-day open",
        "rankers":modes,
        "topks":TOPKS,
        "holds_business_days":HOLDS,
        "ranked_by_event_avg15":ranked15,
        "ranked_by_event_avg41":ranked41,
        "all_results":rows,
        "caveat":"Only four independent selling-climax events. No fees/slippage; current-Prime survivor-biased archive."
    }
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({
        "top15":[{k:r[k] for k in ["ranker","top_k","event_avg15","worst_event15","positive_events15","r15"]} for r in ranked15[:10]],
        "top41":[{k:r[k] for k in ["ranker","top_k","event_avg41","worst_event41","positive_events41","r41"]} for r in ranked41[:10]],
    },ensure_ascii=False,indent=2))

if __name__=="__main__":main()

# trigger
