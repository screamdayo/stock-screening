import json
from pathlib import Path
import numpy as np
import pandas as pd

CACHE = Path("results/kuitto_composite_signal_cache.parquet")
TOPIX = Path("data/topix/topix_daily.parquet")
OUT = Path("results/gakutto_topix_cache_sanity.json")
STOP_PCT = -5.0
MAX_HOLD = 15
DELAYS = [1,3,5,10]
THRESHOLDS = [-0.5,-1.0,-1.5]

def strict_gakutto(g, pos):
    if pos < 3: return False
    m=g["MA5"].to_numpy()
    if any(pd.isna(m[k]) for k in [pos-3,pos-2,pos-1,pos]): return False
    return (m[pos-2]>=m[pos-3] and m[pos-1]>=m[pos-2] and m[pos]<m[pos-1]
            and float(g["C"].iloc[pos]) < float(g["O"].iloc[pos]))

def simulate(g, topix, threshold=None, delay=0):
    g=g.sort_values("offset").reset_index(drop=True)
    idx={int(o):i for i,o in enumerate(g["offset"])}
    if 0 not in idx: return None
    p0=idx[0]; entry=float(g["O"].iloc[p0])
    if not np.isfinite(entry) or entry<=0: return None
    stop=entry*(1+STOP_PCT/100)
    scheduled=None; rescued=False; gd=None
    for off in range(0,MAX_HOLD):
        if off not in idx: return None
        i=idx[off]
        o=float(g["O"].iloc[i]); l=float(g["L"].iloc[i])
        if scheduled is not None and off>=scheduled:
            return {"ret":(o/entry-1)*100,"reason":f"delay_{delay}","rescued":True,"gakutto_date":gd}
        if o<=stop:
            return {"ret":(o/entry-1)*100,"reason":"stop_gap","rescued":rescued,"gakutto_date":gd}
        if l<=stop:
            return {"ret":STOP_PCT,"reason":"stop","rescued":rescued,"gakutto_date":gd}
        if strict_gakutto(g,i):
            d=pd.Timestamp(g["date"].iloc[i]).normalize()
            tx=topix.get(d,np.nan)
            if threshold is not None and pd.notna(tx) and tx<=threshold:
                scheduled=off+1+delay
                rescued=True; gd=str(d.date())
                continue
            if off+1 in idx:
                ep=float(g["O"].iloc[idx[off+1]])
                return {"ret":(ep/entry-1)*100,"reason":"gakutto","rescued":False,"gakutto_date":str(d.date())}
            return None
    if MAX_HOLD in idx:
        ep=float(g["O"].iloc[idx[MAX_HOLD]])
        return {"ret":(ep/entry-1)*100,"reason":"max15","rescued":rescued,"gakutto_date":gd}
    return None

def metrics(rows):
    x=pd.Series([r["ret"] for r in rows if r is not None],dtype=float)
    if x.empty:return {"n":0}
    gp=x[x>0].sum(); gl=-x[x<0].sum()
    return {"n":int(len(x)),"win":round((x>0).mean()*100,2),"avg":round(x.mean(),3),
            "median":round(x.median(),3),"pf":round(gp/gl,3) if gl>0 else None,
            "loss5":round((x<=-5).mean()*100,2)}

def main():
    c=pd.read_parquet(CACHE)
    c["signal_date"]=pd.to_datetime(c["signal_date"])
    c["date"]=pd.to_datetime(c["date"])
    tx=pd.read_parquet(TOPIX)
    tx["date"]=pd.to_datetime(tx["Date"]).dt.normalize()
    tx["close"]=pd.to_numeric(tx["C"],errors="coerce")
    tx=tx.sort_values("date")
    tx["ret1"]=tx["close"].pct_change()*100
    topix=dict(zip(tx["date"],tx["ret1"]))

    groups=[g.copy() for _,g in c.groupby(["code","signal_date"],sort=False)]
    base=[simulate(g,topix) for g in groups]
    base=[r for r in base if r]
    out={"signals":len(groups),"baseline":{"metrics":metrics(base),
          "gakutto":sum(r["reason"]=="gakutto" for r in base)},"tests":{}}
    for th in THRESHOLDS:
        key=str(th)
        out["tests"][key]={}
        for d in DELAYS:
            rows=[simulate(g,topix,th,d) for g in groups]
            rows=[r for r in rows if r]
            # paired by group ordering: all cache groups normally produce one row; record overall delta approx.
            rescued=sum(bool(r.get("rescued")) for r in rows)
            out["tests"][key][str(d)]={"metrics":metrics(rows),"rescued":rescued}
        # event-level: baseline gakutto dates that meet TOPIX threshold
        ev=[]
        for r in base:
            if r["reason"]!="gakutto" or not r["gakutto_date"]: continue
            dt=pd.Timestamp(r["gakutto_date"]).normalize()
            if pd.notna(topix.get(dt,np.nan)) and topix[dt]<=th:
                ev.append({"date":r["gakutto_date"],"topix_ret1":float(topix[dt])})
        out["tests"][key]["event_count"]=len(ev)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(OUT.read_text())

if __name__=="__main__": main()
