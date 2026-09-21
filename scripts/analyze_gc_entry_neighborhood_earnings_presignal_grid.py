import json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

BASE_URL="https://api.jquants.com/v2"
API_KEY=os.environ["JQUANTS_API_KEY"]
HEADERS={"x-api-key":API_KEY}
BATCH_DIR=Path("data/batches")
OUT=Path("results/gc_entry_neighborhood_earnings_presignal_grid.json")
CACHE_DIR=Path(".cache/gc_earnings_presignal_grid")

SLOPE_RISING_PCT=0.25
EPISODE_START_DD=-15.0; EPISODE_RESET_DD=-5.0
TARGET_DD_MIN=-20.0; TARGET_DD_MAX=-15.0

VOL_GRID=[0.85,0.88,0.95]
GAP_GRID=[0.10,0.13,0.16]
CANDLE_GRID=[2.00,2.14,2.30]

def req(path,params=None):
    last=None
    for attempt in range(1,6):
        try:
            r=requests.get(BASE_URL+path,headers=HEADERS,params=params,timeout=60)
            if r.status_code in (429,502,503,504):
                last=RuntimeError(f"{r.status_code}: {r.text[:200]}")
                time.sleep(min(20,2**attempt)); continue
            r.raise_for_status(); return r
        except (requests.Timeout,requests.ConnectionError) as e:
            last=e; time.sleep(min(20,2**attempt))
    raise last or RuntimeError("request failed")

def api_code(code):
    s=str(code); return s if len(s)==5 else s+"0"

def earnings_records(code):
    CACHE_DIR.mkdir(parents=True,exist_ok=True)
    p=CACHE_DIR/f"{code}.json"
    if p.exists():
        try:return json.loads(p.read_text())
        except:pass
    out=[]; pk=None
    while True:
        params={"code":api_code(code)}
        if pk:params["pagination_key"]=pk
        b=req("/fins/earnings-date",params).json()
        out.extend(b.get("data") or [])
        pk=b.get("pagination_key")
        if not pk:break
    p.write_text(json.dumps(out,ensure_ascii=False))
    return out

def load_archive():
    fs=[]
    for p in sorted(BATCH_DIR.glob("batch_*.parquet")):
        d=pd.read_parquet(p)
        cols=[c for c in ["Code","Date","O","H","L","C","Vo","AdjO","AdjH","AdjL","AdjC","AdjVo","ArchiveMarket"] if c in d.columns]
        fs.append(d[cols].copy())
    if not fs: raise RuntimeError("archive missing")
    d=pd.concat(fs,ignore_index=True)
    if "ArchiveMarket" in d.columns:d=d[d.ArchiveMarket=="プライム"].copy()
    for raw,adj in [("O","AdjO"),("H","AdjH"),("L","AdjL"),("C","AdjC"),("Vo","AdjVo")]:
        if adj in d.columns:d[raw]=d[adj].where(d[adj].notna(),d.get(raw))
    for c in ["O","H","L","C","Vo"]:d[c]=pd.to_numeric(d[c],errors="coerce")
    d["Code"]=d.Code.astype(str); d["Date"]=pd.to_datetime(d.Date)
    return d.drop_duplicates(["Code","Date"],keep="last").sort_values(["Code","Date"]).reset_index(drop=True)

def prep(g):
    g=g.dropna(subset=["O","H","L","C"]).sort_values("Date").reset_index(drop=True).copy()
    g["MA5"]=g.C.rolling(5).mean(); g["MA25"]=g.C.rolling(25).mean()
    g["SLOPE25"]=(g.MA25/g.MA25.shift(5)-1)*100
    g["HIGH60"]=g.C.rolling(60,min_periods=60).max()
    g["DD60"]=(g.C/g.HIGH60-1)*100
    g["GAP"]=(g.MA5/g.MA25-1)*100
    g["VOLR"]=g.Vo/g.Vo.rolling(20).mean()
    g["CANDLE"]=(g.C/g.O-1)*100
    g["GC"]=(g.MA5.shift(1)<=g.MA25.shift(1))&(g.MA5>g.MA25)
    active=False;seq=0;arr=[]
    for i in range(len(g)):
        dd=g.DD60.iloc[i]
        if pd.isna(dd):arr.append(0);continue
        if not active and dd<=EPISODE_START_DD:active=True;seq=0
        elif active and dd>EPISODE_RESET_DD:active=False;seq=0
        if active and bool(g.GC.iloc[i]):seq+=1
        arr.append(seq if active else 0)
    g["SEQ"]=arr
    return g

def is_base(g,i):
    if not bool(g.GC.iloc[i]):return False
    s=g.SLOPE25.iloc[i];dd=g.DD60.iloc[i]
    return pd.notna(s) and pd.notna(dd) and s>SLOPE_RISING_PCT and TARGET_DD_MIN<=dd<TARGET_DD_MAX and int(g.SEQ.iloc[i])==2

def latest_known_schedule_dates(records, signal_date):
    sig=pd.Timestamp(signal_date).normalize()
    rows=[]
    for x in records:
        if not x.get("PubDate") or not x.get("SchDate"):continue
        try:
            pub=pd.Timestamp(x["PubDate"]).normalize(); sch=pd.Timestamp(x["SchDate"]).normalize()
        except:continue
        if pub>sig:continue
        rows.append({"pub":pub,"sch":sch,"fy":str(x.get("FYE") or ""),"fq":str(x.get("FQName") or "")})
    if not rows:return []
    d=pd.DataFrame(rows).sort_values("pub")
    latest=d.groupby(["fy","fq"],dropna=False,as_index=False).tail(1)
    return sorted(pd.to_datetime(latest.sch.unique()))

def event_gap(g,sch):
    idx=g.index[g.Date==sch].tolist()
    if not idx:return None
    i=idx[0]
    if i+1>=len(g):return None
    c=float(g.loc[i,"C"]); no=float(g.loc[i+1,"O"])
    if not(np.isfinite(c) and c>0 and np.isfinite(no)):return None
    return (no/c-1)*100

def stats(vals):
    s=pd.Series(vals,dtype=float).dropna()
    if s.empty:return {"n":0}
    return {
      "n":int(len(s)),
      "gap_up_rate_pct":round(float((s>0).mean()*100),2),
      "avg_next_open_gap_pct":round(float(s.mean()),3),
      "median_next_open_gap_pct":round(float(s.median()),3),
      "gap_5pct_plus_rate_pct":round(float((s>=5).mean()*100),2),
      "gap_minus5pct_or_worse_rate_pct":round(float((s<=-5).mean()*100),2),
    }

def main():
    df=load_archive()
    gs={}; sigrows=[]
    for code,g0 in df.groupby("Code",sort=True):
        g=prep(g0); gs[code]=g
        for i in range(60,len(g)-1):
            if is_base(g,i):
                sigrows.append({
                  "code":code,"signal_date":g.Date.iloc[i],
                  "vol":float(g.VOLR.iloc[i]),"gap":float(g.GAP.iloc[i]),"candle":float(g.CANDLE.iloc[i])
                })
    sig=pd.DataFrame(sigrows).dropna()
    envelope=sig[(sig.vol>min(VOL_GRID))&(sig.gap>min(GAP_GRID))&(sig.candle>min(CANDLE_GRID))].copy()
    codes=sorted(envelope.code.unique())
    ev={}
    for n,c in enumerate(codes,1):
        ev[c]=earnings_records(c)
        if n%25==0:print("earnings",n,"/",len(codes))

    events=[]
    for _,r in envelope.iterrows():
        g=gs[r.code]
        idx=g.index[g.Date==r.signal_date].tolist()
        if not idx:continue
        i=idx[0]
        future=set(g.loc[i+1:min(i+10,len(g)-1),"Date"].dt.normalize())
        inside=[d for d in latest_known_schedule_dates(ev.get(r.code,[]),r.signal_date) if d in future]
        if not inside:continue
        sch=min(inside)
        gp=event_gap(g,sch)
        if gp is None:continue
        events.append({
          "code":r.code,"signal_date":r.signal_date,"earnings_date":sch,
          "vol":r.vol,"gap":r.gap,"candle":r.candle,"next_open_gap_pct":gp
        })
    e=pd.DataFrame(events)

    combos=[]
    for v in VOL_GRID:
      for gap in GAP_GRID:
        for c in CANDLE_GRID:
          s=e[(e.vol>v)&(e.gap>gap)&(e.candle>c)]
          combos.append({
            "vol_gt":v,"gap_gt":gap,"candle_gt":c,
            **stats(s.next_open_gap_pct.tolist())
          })

    # descriptive range across the 27 nearby threshold cells
    gap_up=[x["gap_up_rate_pct"] for x in combos if x["n"]]
    avg_gap=[x["avg_next_open_gap_pct"] for x in combos if x["n"]]
    med_gap=[x["median_next_open_gap_pct"] for x in combos if x["n"]]
    plus5=[x["gap_5pct_plus_rate_pct"] for x in combos if x["n"]]
    minus5=[x["gap_minus5pct_or_worse_rate_pct"] for x in combos if x["n"]]
    ns=[x["n"] for x in combos if x["n"]]

    out={
      "definition":"After-close assumption: earnings reaction = scheduled earnings date close to next trading-session open. Each cell uses the same GC seq2 base and varies only the strong-breakout neighborhood thresholds.",
      "grid_values":{"vol":VOL_GRID,"gap":GAP_GRID,"candle":CANDLE_GRID},
      "grid_27":combos,
      "summary_ranges":{
        "n_min":min(ns) if ns else 0,"n_max":max(ns) if ns else 0,
        "gap_up_rate_pct_min":min(gap_up) if gap_up else None,"gap_up_rate_pct_max":max(gap_up) if gap_up else None,
        "avg_next_open_gap_pct_min":min(avg_gap) if avg_gap else None,"avg_next_open_gap_pct_max":max(avg_gap) if avg_gap else None,
        "median_next_open_gap_pct_min":min(med_gap) if med_gap else None,"median_next_open_gap_pct_max":max(med_gap) if med_gap else None,
        "gap_5pct_plus_rate_pct_min":min(plus5) if plus5 else None,"gap_5pct_plus_rate_pct_max":max(plus5) if plus5 else None,
        "gap_minus5pct_or_worse_rate_pct_min":min(minus5) if minus5 else None,"gap_minus5pct_or_worse_rate_pct_max":max(minus5) if minus5 else None,
      },
      "caveats":[
        "Thresholds are post-hoc and samples are very small.",
        "Historical universe is current-Prime survivors.",
        "Actual announcement times are not verified; after-close is a simplifying assumption."
      ]
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=="__main__":main()

# trigger run 2026-09-22
