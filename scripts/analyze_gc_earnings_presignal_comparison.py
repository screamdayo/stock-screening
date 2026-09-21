import json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

BASE_URL="https://api.jquants.com/v2"
API_KEY=os.environ["JQUANTS_API_KEY"]
HEADERS={"x-api-key":API_KEY}
BATCH_DIR=Path("data/batches")
OUT=Path("results/gc_earnings_presignal_comparison.json")
CACHE_DIR=Path(".cache/gc_earnings_presignal")

SLOPE_RISING_PCT=0.25
ROLLING_HIGH=60; EPISODE_START_DD=-15.0; EPISODE_RESET_DD=-5.0
TARGET_DD_MIN=-20.0; TARGET_DD_MAX=-15.0
ANCHOR={"vol":0.88,"gap":0.13,"candle":2.14}

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
    g["HIGH60"]=g.C.rolling(60,min_periods=60).max(); g["DD60"]=(g.C/g.HIGH60-1)*100
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
    # after-close assumption: SchDate close -> next session open
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
    gs={}; base_signals=[]
    for code,g0 in df.groupby("Code",sort=True):
        g=prep(g0); gs[code]=g
        for i in range(60,len(g)-1):
            if not is_base(g,i):continue
            base_signals.append({
              "code":code,"signal_date":g.Date.iloc[i],
              "vol":float(g.VOLR.iloc[i]),"gap":float(g.GAP.iloc[i]),"candle":float(g.CANDLE.iloc[i])
            })
    sig=pd.DataFrame(base_signals).dropna()
    codes=sorted(sig.code.unique())
    ev={c:earnings_records(c) for c in codes}

    # classify base GC seq2 signals that have a known scheduled earnings date in next 10 sessions
    base_cross=[]; strong_cross=[]
    for _,r in sig.iterrows():
        g=gs[r.code]
        si=g.index[g.Date==r.signal_date].tolist()
        if not si:continue
        i=si[0]
        future_dates=set(g.loc[i+1:min(i+10,len(g)-1),"Date"].dt.normalize())
        schs=latest_known_schedule_dates(ev.get(r.code,[]),r.signal_date)
        inside=[d for d in schs if d in future_dates]
        if not inside:continue
        sch=min(inside)
        gp=event_gap(g,sch)
        if gp is None:continue
        rec={"code":r.code,"signal_date":str(pd.Timestamp(r.signal_date).date()),"earnings_date":str(pd.Timestamp(sch).date()),"next_open_gap_pct":gp}
        base_cross.append(rec)
        if r.vol>ANCHOR["vol"] and r.gap>ANCHOR["gap"] and r.candle>ANCHOR["candle"]:
            strong_cross.append(rec)

    # same-company baseline: all scheduled earnings events for companies in strong_cross,
    # restricted to dates with market data in archive
    strong_codes=sorted({x["code"] for x in strong_cross})
    same_company=[]
    min_date=df.Date.min().normalize(); max_date=df.Date.max().normalize()
    for c in strong_codes:
        g=gs[c]
        # unique latest-known schedule snapshots can duplicate actual event dates; de-dupe SchDate
        schs=set()
        for x in ev.get(c,[]):
            sd=x.get("SchDate")
            if not sd:continue
            try:d=pd.Timestamp(sd).normalize()
            except:continue
            if min_date<=d<=max_date:schs.add(d)
        for sch in sorted(schs):
            gp=event_gap(g,sch)
            if gp is not None:
                same_company.append({"code":c,"earnings_date":str(sch.date()),"next_open_gap_pct":gp})

    # remove the 8 strong events from the same-company baseline to show "other earnings"
    strong_keys={(x["code"],x["earnings_date"]) for x in strong_cross}
    same_company_other=[x for x in same_company if (x["code"],x["earnings_date"]) not in strong_keys]

    out={
      "definition":"After-close assumption: earnings reaction = SchDate close to next trading session open. Signal-cross means a schedule known on signal day falls within the next 10 trading sessions.",
      "strong_breakout_earnings":stats([x["next_open_gap_pct"] for x in strong_cross]),
      "all_base_gc_seq2_earnings":stats([x["next_open_gap_pct"] for x in base_cross]),
      "same_companies_all_earnings":stats([x["next_open_gap_pct"] for x in same_company]),
      "same_companies_other_earnings":stats([x["next_open_gap_pct"] for x in same_company_other]),
      "counts":{"base_signals":int(len(sig)),"base_earnings_cross":len(base_cross),"strong_earnings_cross":len(strong_cross),"strong_company_count":len(strong_codes)},
      "strong_events":strong_cross,
      "caveats":[
        "Strong-breakout thresholds were discovered post-hoc; 8 events is a very small sample.",
        "Historical universe is current-Prime survivors.",
        "Actual announcement times are unavailable here; events are treated as after-close releases.",
        "Same-company baseline does not control for quarter, year, industry, or market regime."
      ]
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=="__main__":main()

# trigger run 2026-09-22
