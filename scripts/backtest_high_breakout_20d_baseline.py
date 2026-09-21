import json
from pathlib import Path
import numpy as np
import pandas as pd

BATCH_DIR=Path("data/batches")
OUT=Path("results/high_breakout_20d_baseline.json")
TRADES=Path("results/high_breakout_20d_trades.csv")

LOOKBACK=20
MA=25
MA_SLOPE_LOOKBACK=5
VOL_LOOKBACK=20
HOLD_DAYS=[5,10,15]

def load_archive():
    fs=[]
    for p in sorted(BATCH_DIR.glob("batch_*.parquet")):
        d=pd.read_parquet(p)
        cols=[c for c in ["Code","Date","O","H","L","C","Vo","AdjO","AdjH","AdjL","AdjC","AdjVo","ArchiveMarket"] if c in d.columns]
        fs.append(d[cols].copy())
    if not fs: raise RuntimeError("No archive batches found")
    d=pd.concat(fs,ignore_index=True)
    if "ArchiveMarket" in d.columns:
        d=d[d["ArchiveMarket"]=="プライム"].copy()
    for raw,adj in [("O","AdjO"),("H","AdjH"),("L","AdjL"),("C","AdjC"),("Vo","AdjVo")]:
        if adj in d.columns:
            d[raw]=d[adj].where(d[adj].notna(),d.get(raw))
    for c in ["O","H","L","C","Vo"]:
        d[c]=pd.to_numeric(d[c],errors="coerce")
    d["Code"]=d["Code"].astype(str)
    d["Date"]=pd.to_datetime(d["Date"])
    return d.drop_duplicates(["Code","Date"],keep="last").sort_values(["Code","Date"]).reset_index(drop=True)

def prep(g):
    g=g.dropna(subset=["O","H","L","C"]).sort_values("Date").reset_index(drop=True).copy()
    g["PREV_HIGH20"]=g["C"].shift(1).rolling(LOOKBACK).max()
    g["MA25"]=g["C"].rolling(MA).mean()
    g["MA25_SLOPE5_PCT"]=(g["MA25"]/g["MA25"].shift(MA_SLOPE_LOOKBACK)-1)*100
    g["VOL20"]=g["Vo"].rolling(VOL_LOOKBACK).mean()
    g["VOL_RATIO20"]=g["Vo"]/g["VOL20"]
    g["CANDLE_PCT"]=(g["C"]/g["O"]-1)*100
    g["BREAKOUT_PCT"]=(g["C"]/g["PREV_HIGH20"]-1)*100
    return g

def metrics(vals):
    s=pd.Series(vals,dtype=float).dropna()
    if s.empty:return {"n":0}
    pos=s[s>0];neg=s[s<0]
    gl=-neg.sum()
    pf=pos.sum()/gl if gl>0 else None
    return {
        "n":int(len(s)),
        "win_rate_pct":round(float((s>0).mean()*100),2),
        "avg_return_pct":round(float(s.mean()),3),
        "median_return_pct":round(float(s.median()),3),
        "profit_factor":round(float(pf),3) if pf is not None else None,
        "p10_pct":round(float(s.quantile(.10)),3),
        "p90_pct":round(float(s.quantile(.90)),3),
    }

def main():
    df=load_archive()
    rows=[]
    for code,g0 in df.groupby("Code",sort=True):
        g=prep(g0)
        start=max(LOOKBACK+1,MA+MA_SLOPE_LOOKBACK,VOL_LOOKBACK)
        for i in range(start,len(g)-max(HOLD_DAYS)-1):
            r=g.iloc[i]
            if pd.isna(r["PREV_HIGH20"]) or not float(r["C"])>float(r["PREV_HIGH20"]):
                continue
            entry_i=i+1
            entry=float(g["O"].iloc[entry_i])
            if not(np.isfinite(entry) and entry>0):continue
            rec={
                "code":code,
                "signal_date":str(pd.Timestamp(r["Date"]).date()),
                "entry_date":str(pd.Timestamp(g["Date"].iloc[entry_i]).date()),
                "ma25_slope5_pct":float(r["MA25_SLOPE5_PCT"]) if pd.notna(r["MA25_SLOPE5_PCT"]) else None,
                "vol_ratio20":float(r["VOL_RATIO20"]) if pd.notna(r["VOL_RATIO20"]) else None,
                "candle_pct":float(r["CANDLE_PCT"]) if pd.notna(r["CANDLE_PCT"]) else None,
                "breakout_pct":float(r["BREAKOUT_PCT"]) if pd.notna(r["BREAKOUT_PCT"]) else None,
            }
            for h in HOLD_DAYS:
                exit_i=entry_i+h
                exitp=float(g["O"].iloc[exit_i])
                rec[f"ret_{h}d_pct"]=(exitp/entry-1)*100 if np.isfinite(exitp) else None
            rows.append(rec)

    t=pd.DataFrame(rows)
    if t.empty: raise RuntimeError("No signals")
    t["signal_date_dt"]=pd.to_datetime(t["signal_date"])
    split=pd.Timestamp(df["Date"].min()).normalize()+pd.DateOffset(years=5)

    variants={
      "plain_20d_breakout": pd.Series(True,index=t.index),
      "ma25_rising": t["ma25_slope5_pct"]>0.25,
      "volume_1_5x": t["vol_ratio20"]>=1.5,
      "candle_1pct": t["candle_pct"]>=1.0,
      "ma25_rising_plus_volume": (t["ma25_slope5_pct"]>0.25)&(t["vol_ratio20"]>=1.5),
      "ma25_rising_plus_candle": (t["ma25_slope5_pct"]>0.25)&(t["candle_pct"]>=1.0),
      "volume_plus_candle": (t["vol_ratio20"]>=1.5)&(t["candle_pct"]>=1.0),
      "all_three": (t["ma25_slope5_pct"]>0.25)&(t["vol_ratio20"]>=1.5)&(t["candle_pct"]>=1.0),
    }

    out={
      "strategy":"20-day closing-high breakout baseline",
      "universe":"current Prime listings in saved archive",
      "archive_start":str(pd.Timestamp(df["Date"].min()).date()),
      "archive_end":str(pd.Timestamp(df["Date"].max()).date()),
      "split_date":str(split.date()),
      "entry":"signal when close > highest close of prior 20 sessions; buy next open",
      "exit":"fixed open after 5/10/15 trading sessions",
      "variants":{},
      "caveat":"Current-listing universe implies survivorship bias. This is exploratory; no transaction costs/slippage and no position-overlap constraint."
    }
    for name,mask in variants.items():
        part=t[mask].copy()
        older=part[part.signal_date_dt<split]
        recent=part[part.signal_date_dt>=split]
        out["variants"][name]={
          f"{h}d":{
            "overall":metrics(part[f"ret_{h}d_pct"]),
            "older5":metrics(older[f"ret_{h}d_pct"]),
            "recent5":metrics(recent[f"ret_{h}d_pct"]),
          } for h in HOLD_DAYS
        }

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    t.drop(columns=["signal_date_dt"]).to_csv(TRADES,index=False)
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
