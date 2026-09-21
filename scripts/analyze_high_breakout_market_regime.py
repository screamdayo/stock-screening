import json
from pathlib import Path
import numpy as np
import pandas as pd

BATCH_DIR=Path("data/batches")
OUT=Path("results/high_breakout_market_regime.json")
TRADES=Path("results/high_breakout_market_regime_trades.csv")

LOOKBACK=252
MA=25
MA_SLOPE_LOOKBACK=5
VOL_LOOKBACK=20
VOL_MIN=1.5
MA25_SLOPE_MIN=0.25
HOLD_DAYS=15

def load_archive():
    fs=[]
    for p in sorted(BATCH_DIR.glob("batch_*.parquet")):
        d=pd.read_parquet(p)
        cols=[c for c in ["Code","Date","O","H","L","C","Vo","AdjO","AdjH","AdjL","AdjC","AdjVo","ArchiveMarket"] if c in d.columns]
        fs.append(d[cols].copy())
    if not fs:
        raise RuntimeError("No archive batches found")
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
    d=d.drop_duplicates(["Code","Date"],keep="last").sort_values(["Code","Date"]).reset_index(drop=True)
    return d

def add_stock_features(df):
    out=[]
    for code,g in df.groupby("Code",sort=False):
        g=g.sort_values("Date").reset_index(drop=True).copy()
        g["MA25"]=g["C"].rolling(MA).mean()
        g["MA25_SLOPE5_PCT"]=(g["MA25"]/g["MA25"].shift(MA_SLOPE_LOOKBACK)-1)*100
        g["VOL20"]=g["Vo"].rolling(VOL_LOOKBACK).mean()
        g["VOL_RATIO20"]=g["Vo"]/g["VOL20"]
        g["PREV_HIGH252"]=g["C"].shift(1).rolling(LOOKBACK).max()
        g["ABOVE_MA25"]=g["C"]>g["MA25"]
        g["RET25_PCT"]=(g["C"]/g["C"].shift(25)-1)*100
        g["RET5_PCT"]=(g["C"]/g["C"].shift(5)-1)*100
        out.append(g)
    return pd.concat(out,ignore_index=True)

def build_market_regime(df):
    # Cross-sectional market state known at each close; no future data.
    m=df.groupby("Date").agg(
        breadth_above_ma25=("ABOVE_MA25","mean"),
        median_ret25_pct=("RET25_PCT","median"),
        median_ret5_pct=("RET5_PCT","median"),
        stock_count=("Code","nunique"),
    ).reset_index()
    m["breadth_above_ma25_pct"]=m["breadth_above_ma25"]*100
    m["regime_breadth50"]=m["breadth_above_ma25_pct"]>=50
    m["regime_breadth55"]=m["breadth_above_ma25_pct"]>=55
    m["regime_med25_pos"]=m["median_ret25_pct"]>0
    m["regime_med25_gt2"]=m["median_ret25_pct"]>2
    m["regime_combo"]=m["regime_breadth50"] & m["regime_med25_pos"]
    return m

def metrics(df):
    if df.empty:return {"n":0}
    s=pd.to_numeric(df["ret_15d_pct"],errors="coerce").dropna()
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
    raw=load_archive()
    df=add_stock_features(raw)
    market=build_market_regime(df)

    rows=[]
    for code,g in df.groupby("Code",sort=True):
        g=g.sort_values("Date").reset_index(drop=True)
        start=max(LOOKBACK+1,MA+MA_SLOPE_LOOKBACK,VOL_LOOKBACK)
        for i in range(start,len(g)-HOLD_DAYS-1):
            r=g.iloc[i]
            if pd.isna(r["PREV_HIGH252"]) or not float(r["C"])>float(r["PREV_HIGH252"]):
                continue
            if pd.isna(r["MA25_SLOPE5_PCT"]) or not float(r["MA25_SLOPE5_PCT"])>MA25_SLOPE_MIN:
                continue
            if pd.isna(r["VOL_RATIO20"]) or not float(r["VOL_RATIO20"])>=VOL_MIN:
                continue

            entry_i=i+1
            exit_i=entry_i+HOLD_DAYS
            entry=float(g["O"].iloc[entry_i]); exitp=float(g["O"].iloc[exit_i])
            if not(np.isfinite(entry) and np.isfinite(exitp) and entry>0):
                continue
            rows.append({
                "code":code,
                "signal_date":pd.Timestamp(r["Date"]),
                "entry_date":pd.Timestamp(g["Date"].iloc[entry_i]),
                "exit_date":pd.Timestamp(g["Date"].iloc[exit_i]),
                "ret_15d_pct":(exitp/entry-1)*100,
                "ma25_slope5_pct":float(r["MA25_SLOPE5_PCT"]),
                "vol_ratio20":float(r["VOL_RATIO20"]),
            })

    t=pd.DataFrame(rows)
    if t.empty: raise RuntimeError("No signals")
    t=t.merge(market,on="Date" if "Date" in t.columns else "signal_date",how="left")
    if "Date" in t.columns:
        t=t.rename(columns={"Date":"market_date"})

    split=pd.Timestamp(raw["Date"].min()).normalize()+pd.DateOffset(years=5)
    t["year"]=t["signal_date"].dt.year

    masks={
      "all":pd.Series(True,index=t.index),
      "breadth_above_ma25_ge_50":t["regime_breadth50"].fillna(False),
      "breadth_above_ma25_ge_55":t["regime_breadth55"].fillna(False),
      "median_25d_return_gt_0":t["regime_med25_pos"].fillna(False),
      "median_25d_return_gt_2":t["regime_med25_gt2"].fillna(False),
      "breadth50_and_median25_positive":t["regime_combo"].fillna(False),
    }

    out={
      "strategy":"252d closing-high breakout + MA25 rising + volume >=1.5x",
      "exit":"15 trading sessions, next-open entry / open exit",
      "market_regime_definition":"cross-sectional current-Prime survivors, signal-day close only",
      "split_date":str(split.date()),
      "variants":{},
      "note":"Exploratory regime test. Current-listing survivorship bias remains; regime is derived from the same survivor universe, not historical TOPIX constituents."
    }

    for name,mask in masks.items():
        part=t[mask].copy()
        older=part[part["signal_date"]<split]
        recent=part[part["signal_date"]>=split]
        yearly={str(int(y)):metrics(g) for y,g in part.groupby("year")}
        out["variants"][name]={
          "overall":metrics(part),
          "older5":metrics(older),
          "recent5":metrics(recent),
          "yearly":yearly,
        }

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    t.to_csv(TRADES,index=False)
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
