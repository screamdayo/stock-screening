import json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

BASE_URL="https://api.jquants.com/v2"
API_KEY=os.environ["JQUANTS_API_KEY"]
HEADERS={"x-api-key":API_KEY}
BATCH_DIR=Path("data/batches")
RESULT_DIR=Path("results")
CACHE_DIR=Path(".cache/gc_earnings")

MA_SHORT=5; MA_LONG=25; SLOPE_LOOKBACK=5; SLOPE_RISING_PCT=0.25
ROLLING_HIGH=60; EPISODE_START_DD=-15.0; EPISODE_RESET_DD=-5.0
TARGET_DD_MIN=-20.0; TARGET_DD_MAX=-15.0
VOL_GRID=[0.85,0.88,0.95]
GAP_GRID=[0.10,0.13,0.16]
CANDLE_GRID=[2.00,2.14,2.30]
ANCHOR=(0.88,0.13,2.14)

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
    s=str(code)
    return s if len(s)==5 else s+"0"

def earnings_records(code):
    CACHE_DIR.mkdir(parents=True,exist_ok=True)
    p=CACHE_DIR/f"{code}.json"
    if p.exists():
        try:return json.loads(p.read_text())
        except:pass
    out=[]; pk=None
    while True:
        params={"code":api_code(code)}
        if pk: params["pagination_key"]=pk
        b=req("/fins/earnings-date",params).json()
        out.extend(b.get("data") or [])
        pk=b.get("pagination_key")
        if not pk: break
    p.write_text(json.dumps(out,ensure_ascii=False))
    return out

def known_cross(records,signal_date,entry_date,exit_date):
    sig=pd.Timestamp(signal_date).normalize(); ent=pd.Timestamp(entry_date).normalize(); ex=pd.Timestamp(exit_date).normalize()
    rows=[]
    for x in records:
        if not x.get("PubDate") or not x.get("SchDate"): continue
        try:
            pub=pd.Timestamp(x["PubDate"]).normalize(); sch=pd.Timestamp(x["SchDate"]).normalize()
        except: continue
        if pub>sig: continue
        rows.append({"pub":pub,"sch":sch,"fy":str(x.get("FYE") or ""),"fq":str(x.get("FQName") or "")})
    if not rows:return False,None
    d=pd.DataFrame(rows).sort_values("pub")
    latest=d.groupby(["fy","fq"],dropna=False,as_index=False).tail(1)
    inside=latest[(latest.sch>=ent)&(latest.sch<=ex)].sort_values("sch")
    if inside.empty:return False,None
    return True,str(inside.iloc[0].sch.date())

def load_archive():
    fs=[]
    for p in sorted(BATCH_DIR.glob("batch_*.parquet")):
        d=pd.read_parquet(p)
        cols=[c for c in ["Code","Date","O","H","L","C","Vo","AdjO","AdjH","AdjL","AdjC","AdjVo","ArchiveMarket"] if c in d.columns]
        fs.append(d[cols].copy())
    if not fs:raise RuntimeError("archive missing")
    d=pd.concat(fs,ignore_index=True)
    if "ArchiveMarket" in d.columns:d=d[d.ArchiveMarket=="プライム"].copy()
    for raw,adj in [("O","AdjO"),("H","AdjH"),("L","AdjL"),("C","AdjC"),("Vo","AdjVo")]:
        if adj in d.columns:d[raw]=d[adj].where(d[adj].notna(),d.get(raw))
    for c in ["O","H","L","C","Vo"]:d[c]=pd.to_numeric(d[c],errors="coerce")
    d["Code"]=d["Code"].astype(str); d["Date"]=pd.to_datetime(d["Date"])
    return d.drop_duplicates(["Code","Date"],keep="last").sort_values(["Code","Date"]).reset_index(drop=True)

def prep(g):
    g=g.dropna(subset=["O","H","L","C"]).sort_values("Date").reset_index(drop=True).copy()
    g["MA5"]=g.C.rolling(5).mean(); g["MA25"]=g.C.rolling(25).mean()
    g["SLOPE25"]=(g.MA25/g.MA25.shift(5)-1)*100
    g["HIGH60"]=g.C.rolling(60,min_periods=60).max(); g["DD60"]=(g.C/g.HIGH60-1)*100
    g["GAP"]=(g.MA5/g.MA25-1)*100; g["VOLR"]=g.Vo/g.Vo.rolling(20).mean(); g["CANDLE"]=(g.C/g.O-1)*100
    g["GC"]=(g.MA5.shift(1)<=g.MA25.shift(1))&(g.MA5>g.MA25)
    active=False; seq=0; arr=[]
    for i in range(len(g)):
        dd=g.DD60.iloc[i]
        if pd.isna(dd):arr.append(0);continue
        if not active and dd<=EPISODE_START_DD:active=True;seq=0
        elif active and dd>EPISODE_RESET_DD:active=False;seq=0
        if active and bool(g.GC.iloc[i]):seq+=1
        arr.append(seq if active else 0)
    g["SEQ"]=arr
    return g

def base(g,i):
    if not bool(g.GC.iloc[i]):return False
    s=g.SLOPE25.iloc[i]; dd=g.DD60.iloc[i]
    return pd.notna(s) and pd.notna(dd) and s>SLOPE_RISING_PCT and TARGET_DD_MIN<=dd<TARGET_DD_MAX and int(g.SEQ.iloc[i])==2

def metrics(p):
    r=p.ret10.dropna()
    if len(r)==0:return {"n":0}
    gain=r[r>0].sum(); loss=-r[r<0].sum()
    return {"n":int(len(r)),"win_rate_pct":round(float((r>0).mean()*100),2),"avg_ret10_pct":round(float(r.mean()),3),"profit_factor":round(float(gain/loss),3) if loss>0 else None}

def main():
    df=load_archive(); rows=[]
    for code,g0 in df.groupby("Code",sort=True):
        g=prep(g0)
        for i in range(60,len(g)-12):
            if not base(g,i):continue
            ei=i+1; xi=ei+10
            entry=float(g.O.iloc[ei]); exitp=float(g.O.iloc[xi])
            if not(np.isfinite(entry) and entry>0 and np.isfinite(exitp)):continue
            rows.append({"code":str(code),"signal_date":g.Date.iloc[i],"entry_date":g.Date.iloc[ei],"exit_date":g.Date.iloc[xi],
                         "vol":g.VOLR.iloc[i],"gap":g.GAP.iloc[i],"candle":g.CANDLE.iloc[i],"ret10":(exitp/entry-1)*100})
    t=pd.DataFrame(rows).dropna().copy()
    # Fetch earnings only for codes that appear in the broadest local-grid envelope.
    envelope=t[(t.vol>min(VOL_GRID))&(t.gap>min(GAP_GRID))&(t.candle>min(CANDLE_GRID))].copy()
    codes=sorted(envelope.code.unique())
    events={}
    for n,c in enumerate(codes,1):
        events[c]=earnings_records(c)
        if n%25==0: print("earnings",n,"/",len(codes))
    flags=[]; dates=[]
    for _,r in envelope.iterrows():
        f,d=known_cross(events.get(r.code,[]),r.signal_date,r.entry_date,r.exit_date)
        flags.append(f); dates.append(d)
    envelope["earnings_cross"]=flags; envelope["earnings_date"]=dates

    combos=[]
    for v in VOL_GRID:
      for gap in GAP_GRID:
        for c in CANDLE_GRID:
          s=envelope[(envelope.vol>v)&(envelope.gap>gap)&(envelope.candle>c)].copy()
          cross=s[s.earnings_cross]; no=s[~s.earnings_cross]
          combos.append({
            "vol_gt":v,"gap_gt":gap,"candle_gt":c,
            "n":int(len(s)),
            "earnings_cross_n":int(len(cross)),
            "earnings_cross_rate_pct":round(float(s.earnings_cross.mean()*100),2) if len(s) else None,
            "no_earnings_n":int(len(no)),
            "all":metrics(s),"earnings_cross":metrics(cross),"no_earnings":metrics(no)
          })

    av,ag,ac=ANCHOR
    a=envelope[(envelope.vol>av)&(envelope.gap>ag)&(envelope.candle>ac)].copy()
    out={
      "definition":"Earnings crossing = the latest J-Quants earnings schedule known as of signal day has SchDate from next-open entry through fixed 10-session exit, inclusive.",
      "anchor":{"rules":{"vol_gt":av,"gap_gt":ag,"candle_gt":ac},"n":int(len(a)),
                "earnings_cross_n":int(a.earnings_cross.sum()),"earnings_cross_rate_pct":round(float(a.earnings_cross.mean()*100),2),
                "all":metrics(a),"earnings_cross":metrics(a[a.earnings_cross]),"no_earnings":metrics(a[~a.earnings_cross])},
      "local_27":combos,
      "envelope_n":int(len(envelope)),
      "unique_codes_queried":int(len(codes)),
      "caveat":"Entry thresholds are post-hoc. Earnings schedules are reconstructed only from records published on or before each signal date."
    }
    RESULT_DIR.mkdir(exist_ok=True)
    (RESULT_DIR/"gc_entry_neighborhood_earnings_cross.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    envelope.to_csv(RESULT_DIR/"gc_entry_neighborhood_earnings_trades.csv",index=False)
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=="__main__":main()

# trigger run 2026-09-22
