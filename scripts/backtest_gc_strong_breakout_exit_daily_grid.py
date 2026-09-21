import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR=Path("data")
BATCH_DIR=DATA_DIR/"batches"
RESULT_DIR=Path("results")

MA_SHORT=5; MA_LONG=25; SLOPE_LOOKBACK=5; SLOPE_RISING_PCT=0.25
ROLLING_HIGH=60; EPISODE_START_DD=-15.0; EPISODE_RESET_DD=-5.0
TARGET_DD_MIN=-20.0; TARGET_DD_MAX=-15.0
VOL_RATIO_MIN=0.88; GC_GAP_MIN=0.13; CANDLE_MIN=2.14
HOLDS=list(range(5,16))


def load_archive():
    frames=[]
    for p in sorted(BATCH_DIR.glob("batch_*.parquet")):
        d=pd.read_parquet(p)
        cols=[c for c in ["Code","Date","O","H","L","C","Vo","AdjO","AdjH","AdjL","AdjC","AdjVo","ArchiveMarket"] if c in d.columns]
        frames.append(d[cols].copy())
    if not frames: raise RuntimeError("No archive batches")
    df=pd.concat(frames,ignore_index=True)
    if "ArchiveMarket" in df.columns: df=df[df["ArchiveMarket"]=="プライム"].copy()
    for raw,adj in [("O","AdjO"),("H","AdjH"),("L","AdjL"),("C","AdjC"),("Vo","AdjVo")]:
        if adj in df.columns:
            df[raw]=df[adj].where(df[adj].notna(),df.get(raw))
    for c in ["O","H","L","C","Vo"]: df[c]=pd.to_numeric(df[c],errors="coerce")
    df["Code"]=df["Code"].astype(str); df["Date"]=pd.to_datetime(df["Date"])
    return df.drop_duplicates(["Code","Date"],keep="last").sort_values(["Code","Date"]).reset_index(drop=True)


def prep(g):
    g=g.dropna(subset=["O","H","L","C"]).sort_values("Date").reset_index(drop=True).copy()
    g["MA5"]=g["C"].rolling(5).mean(); g["MA25"]=g["C"].rolling(25).mean()
    g["MA25_SLOPE5_PCT"]=(g["MA25"]/g["MA25"].shift(5)-1)*100
    g["HIGH60"]=g["C"].rolling(60,min_periods=60).max()
    g["DD60_PCT"]=(g["C"]/g["HIGH60"]-1)*100
    g["GC_GAP_PCT"]=(g["MA5"]/g["MA25"]-1)*100
    g["VOL_RATIO20"]=g["Vo"]/g["Vo"].rolling(20).mean()
    g["CANDLE_PCT"]=(g["C"]/g["O"]-1)*100
    g["GC"]=(g["MA5"].shift(1)<=g["MA25"].shift(1))&(g["MA5"]>g["MA25"])
    active=False; seq=0; seqs=[]
    for i in range(len(g)):
        dd=g["DD60_PCT"].iloc[i]
        if pd.isna(dd): seqs.append(0); continue
        if not active and dd<=EPISODE_START_DD: active=True; seq=0
        elif active and dd>EPISODE_RESET_DD: active=False; seq=0
        if active and bool(g["GC"].iloc[i]): seq+=1
        seqs.append(seq if active else 0)
    g["GC_SEQ"]=seqs
    return g


def is_signal(g,i):
    if not bool(g["GC"].iloc[i]): return False
    slope=g["MA25_SLOPE5_PCT"].iloc[i]; dd=g["DD60_PCT"].iloc[i]
    if pd.isna(slope) or pd.isna(dd): return False
    if not (slope>SLOPE_RISING_PCT and TARGET_DD_MIN<=dd<TARGET_DD_MAX and int(g["GC_SEQ"].iloc[i])==2): return False
    vals=[g["VOL_RATIO20"].iloc[i],g["GC_GAP_PCT"].iloc[i],g["CANDLE_PCT"].iloc[i]]
    if any(pd.isna(v) for v in vals): return False
    return vals[0]>VOL_RATIO_MIN and vals[1]>GC_GAP_MIN and vals[2]>CANDLE_MIN


def metrics(vals):
    s=pd.Series(vals,dtype=float).dropna()
    if s.empty:return {"n":0}
    gp=s[s>0].sum(); gl=-s[s<0].sum()
    return {
        "n":int(len(s)),
        "win_rate_pct":round(float((s>0).mean()*100),2),
        "avg_return_pct":round(float(s.mean()),3),
        "median_return_pct":round(float(s.median()),3),
        "profit_factor":round(float(gp/gl),3) if gl>0 else None,
        "p5_plus_pct":round(float((s>=5).mean()*100),2),
        "loss5_or_worse_pct":round(float((s<=-5).mean()*100),2),
    }


def main():
    df=load_archive()
    rows=[]
    maxh=max(HOLDS)
    for code,g0 in df.groupby("Code",sort=True):
        g=prep(g0)
        for i in range(60,len(g)-maxh-2):
            if not is_signal(g,i): continue
            ei=i+1
            entry=float(g["O"].iloc[ei])
            if not(np.isfinite(entry) and entry>0): continue
            rec={"code":str(code),"signal_date":pd.Timestamp(g["Date"].iloc[i])}
            for h in HOLDS:
                xi=ei+h
                exitp=float(g["O"].iloc[xi])
                rec[f"ret_{h}d"]=(exitp/entry-1)*100 if np.isfinite(exitp) else np.nan
            rows.append(rec)
    t=pd.DataFrame(rows)
    split=pd.Timestamp(df["Date"].min()).normalize()+pd.DateOffset(years=5)
    out={
        "strategy":"GC seq2 strong-breakout subset",
        "holds_tested":HOLDS,
        "split_date":str(split.date()),
        "overall":{},
        "older5":{},
        "recent5":{},
        "yearly":{},
        "caveat":"Strong-breakout entry subset is post-hoc; this 1-day exit grid is exploratory."
    }
    for h in HOLDS:
        c=f"ret_{h}d"
        out["overall"][str(h)]=metrics(t[c])
        out["older5"][str(h)]=metrics(t.loc[t.signal_date<split,c])
        out["recent5"][str(h)]=metrics(t.loc[t.signal_date>=split,c])
    for y,p in t.groupby(t.signal_date.dt.year):
        out["yearly"][str(int(y))]={str(h):metrics(p[f"ret_{h}d"]) for h in HOLDS}
    RESULT_DIR.mkdir(exist_ok=True)
    (RESULT_DIR/"gc_strong_breakout_exit_daily_grid_summary.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    t.assign(signal_date=t.signal_date.dt.date.astype(str)).to_csv(RESULT_DIR/"gc_strong_breakout_exit_daily_grid_trades.csv",index=False)
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
