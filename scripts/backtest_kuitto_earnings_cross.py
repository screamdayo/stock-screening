import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://api.jquants.com/v2"
API_KEY = os.environ["JQUANTS_API_KEY"]
HEADERS = {"x-api-key": API_KEY}
DATA_DIR = Path("data")
BATCH_DIR = DATA_DIR / "batches"
RESULT_DIR = Path("results")
CACHE_DIR = Path(".cache") / "earnings"

REQUEST_TIMEOUT = 60
MAX_RETRIES = 5
HOLD_DAYS = 10

MA_SHORT=5; MA_LONG=25; TURN_LOOKBACK=2; DECLINE_LOOKBACK=5
DECLINE_MIN_PCT=-5.5; DECLINE_MAX_PCT=-2.0
MA5_VS_MA25_MIN_PCT=-5.0; MA5_VS_MA25_MAX_PCT=0.0
CLOSE_VS_MA5_MAX_PCT=4.0; BULL_MIN_PCT=1.5; BULL_MAX_PCT=3.5
VOLUME_RATIO_MIN=1.25

def request(path, params=None):
    last=None
    for attempt in range(1, MAX_RETRIES+1):
        try:
            r=requests.get(f"{BASE_URL}{path}",headers=HEADERS,params=params,timeout=REQUEST_TIMEOUT)
            if r.status_code in (429,502,503,504):
                time.sleep(min(20,2**attempt)); last=RuntimeError(f"{r.status_code}: {r.text[:300]}"); continue
            return r
        except (requests.Timeout,requests.ConnectionError) as e:
            last=e; time.sleep(min(20,2**attempt))
    raise last or RuntimeError("request failed")

def load_archive():
    fs=[]
    for p in sorted(BATCH_DIR.glob("batch_*.parquet")):
        d=pd.read_parquet(p)
        cols=[c for c in ["Code","Date","O","H","L","C","Vo","AdjO","AdjH","AdjL","AdjC","AdjVo","ArchiveMarket"] if c in d.columns]
        fs.append(d[cols])
    if not fs: raise RuntimeError("archive missing")
    d=pd.concat(fs,ignore_index=True)
    if "ArchiveMarket" in d.columns: d=d[d["ArchiveMarket"]=="プライム"].copy()
    for raw,adj in [("O","AdjO"),("H","AdjH"),("L","AdjL"),("C","AdjC"),("Vo","AdjVo")]:
        if adj in d.columns:
            d[raw]=d[adj].where(d[adj].notna(),d.get(raw))
    for c in ["O","H","L","C","Vo"]: d[c]=pd.to_numeric(d[c],errors="coerce")
    d["Code"]=d["Code"].astype(str); d["Date"]=pd.to_datetime(d["Date"])
    return d.drop_duplicates(["Code","Date"],keep="last").sort_values(["Code","Date"]).reset_index(drop=True)

def prep(g):
    g=g.dropna(subset=["O","C"]).sort_values("Date").reset_index(drop=True).copy()
    g["MA5"]=g["C"].rolling(5).mean(); g["MA25"]=g["C"].rolling(25).mean()
    return g

def sig(g,i):
    if i<25:return False
    r=g.iloc[i]
    if pd.isna(r["MA5"]) or pd.isna(r["MA25"]) or not r["O"]>0:return False
    bull=(r["C"]/r["O"]-1)*100
    if not(BULL_MIN_PCT<=bull<=BULL_MAX_PCT):return False
    gap=(r["MA5"]/r["MA25"]-1)*100
    if not(MA5_VS_MA25_MIN_PCT<=gap<=MA5_VS_MA25_MAX_PCT):return False
    base=i-1; past=base-5
    if pd.isna(g["MA5"].iloc[past]) or not g["MA5"].iloc[past]>0:return False
    dec=(g["MA5"].iloc[base]/g["MA5"].iloc[past]-1)*100
    if not(DECLINE_MIN_PCT<=dec<=DECLINE_MAX_PCT):return False
    for j in range(i-2,i):
        if g["MA5"].iloc[j]>g["MA5"].iloc[j-1]: return False
    if not g["MA5"].iloc[i]>g["MA5"].iloc[i-1]:return False
    if (r["C"]/r["MA5"]-1)*100>CLOSE_VS_MA5_MAX_PCT:return False
    pv=g["Vo"].iloc[i-1]; tv=r["Vo"]
    if pd.isna(pv) or pd.isna(tv) or not float(pv)>0:return False
    return float(tv)/float(pv)>=VOLUME_RATIO_MIN

def build_trades(df):
    rows=[]
    for code,g0 in df.groupby("Code",sort=True):
        g=prep(g0)
        for i in range(25,len(g)-HOLD_DAYS-1):
            if not sig(g,i):continue
            ei=i+1; xi=ei+HOLD_DAYS
            entry=float(g["O"].iloc[ei]); exitp=float(g["O"].iloc[xi])
            if not(entry>0 and np.isfinite(entry) and np.isfinite(exitp)):continue
            rows.append({"code":str(code),"signal_date":g["Date"].iloc[i],"entry_date":g["Date"].iloc[ei],"exit_date":g["Date"].iloc[xi],"ret10":(exitp/entry-1)*100})
    return pd.DataFrame(rows)

def api_code(code):
    code=str(code)
    return code if len(code)==5 else code+"0"

def fetch_earnings_for_code(code):
    """Fetch earnings-schedule history for one stock from /fins/earnings-date."""
    CACHE_DIR.mkdir(parents=True,exist_ok=True)
    cp=CACHE_DIR/f"{code}.json"
    if cp.exists():
        try:return json.loads(cp.read_text(encoding="utf-8"))
        except:pass

    out=[]; pk=None
    while True:
        params={"code":api_code(code)}
        if pk:params["pagination_key"]=pk
        r=request("/fins/earnings-date",params)
        r.raise_for_status()
        b=r.json()
        arr=b.get("data") or []
        out.extend(arr)
        pk=b.get("pagination_key")
        if not pk:break

    cp.write_text(json.dumps(out,ensure_ascii=False),encoding="utf-8")
    return out

def known_schedule_crosses(records, signal_date, entry_date, exit_date):
    """Whether the schedule known on signal day had earnings inside the holding window.

    For each fiscal quarter/year key, use only the latest schedule publication
    available as of the signal date. This avoids using later schedule revisions.
    """
    if not records:
        return False, None

    sig=pd.Timestamp(signal_date).normalize()
    ent=pd.Timestamp(entry_date).normalize()
    ex=pd.Timestamp(exit_date).normalize()

    rows=[]
    for x in records:
        pub=x.get("PubDate")
        sch=x.get("SchDate")
        if not pub or not sch:
            continue
        try:
            pubd=pd.Timestamp(pub).normalize()
            schd=pd.Timestamp(sch).normalize()
        except Exception:
            continue
        if pubd>sig:
            continue
        rows.append({
            "pub":pubd,
            "sch":schd,
            "fy":str(x.get("FYE") or ""),
            "fq":str(x.get("FQName") or ""),
        })

    if not rows:
        return False, None

    d=pd.DataFrame(rows).sort_values("pub")
    latest=d.groupby(["fy","fq"],dropna=False,as_index=False).tail(1)
    inside=latest[(latest["sch"]>=ent)&(latest["sch"]<=ex)].sort_values("sch")
    if inside.empty:
        return False, None
    return True, str(inside.iloc[0]["sch"].date())

def metrics(x):
    if x.empty:return {"n":0}
    r=x["ret10"].dropna(); gain=r[r>0].sum(); loss=-r[r<0].sum()
    return {"n":int(len(r)),"win_rate_pct":round(float((r>0).mean()*100),2),"avg_ret10_pct":round(float(r.mean()),3),"median_ret10_pct":round(float(r.median()),3),"p5_plus_pct":round(float((r>=5).mean()*100),2),"p10_plus_pct":round(float((r>=10).mean()*100),2),"loss5_or_worse_pct":round(float((r<=-5).mean()*100),2),"profit_factor":round(float(gain/loss),3) if loss>0 else None,"worst_ret10_pct":round(float(r.min()),3)}

def main():
    # Fail fast before the expensive 10y calculation if the endpoint is not available.
    probe=fetch_earnings_for_code("8697")
    print("earnings endpoint probe ok:", len(probe), "records")

    df=load_archive(); t=build_trades(df)
    codes=sorted(t["code"].unique())
    ev={}
    sample_shape=list(probe[0].keys()) if probe else None
    for k,code in enumerate(codes,1):
        recs=probe if code=="8697" else fetch_earnings_for_code(code)
        if sample_shape is None and recs:
            sample_shape=list(recs[0].keys())
        ev[code]=recs
        if k%100==0:print("earnings",k,"/",len(codes))

    crosses=[]
    next_days=[]
    for _,r in t.iterrows():
        flag,day=known_schedule_crosses(
            ev.get(str(r["code"]),[]),
            r["signal_date"],r["entry_date"],r["exit_date"]
        )
        crosses.append(flag)
        next_days.append(day)
    t["earnings_within_10d"]=crosses
    t["earnings_date"]=next_days
    split=pd.Timestamp(df["Date"].min()).normalize()+pd.DateOffset(years=5)
    out={"definition":"Frozen Kuitto signal, next-open entry, fixed 10-session exit. Earnings-crossing means the latest J-Quants earnings schedule known as of signal day (/fins/earnings-date, latest PubDate per FYE/FQName) had SchDate from entry through exit inclusive.","split_date":str(split.date()),"api_sample_fields":sample_shape,"overall":{},"older5":{},"recent5":{},"caveat":"Historical universe is current-Prime survivors. Earnings-event classification is reconstructed from schedule records published on or before each signal date, avoiding later schedule revisions."}
    for label,part in [("overall",t),("older5",t[t.signal_date<split]),("recent5",t[t.signal_date>=split])]:
        out[label]={"no_earnings":metrics(part[~part.earnings_within_10d]),"crosses_earnings":metrics(part[part.earnings_within_10d]),"cross_rate_pct":round(float(part.earnings_within_10d.mean()*100),2)}
    RESULT_DIR.mkdir(exist_ok=True)
    (RESULT_DIR/"kuitto_earnings_cross_summary.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    t.to_csv(RESULT_DIR/"kuitto_earnings_cross_trades.csv",index=False)
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
