import json
from pathlib import Path
import numpy as np
import pandas as pd

BATCH_DIR=Path("data/batches")
OUT=Path("results/high_breakout_60_120_252d.json")
TRADES=Path("results/high_breakout_60_120_252d_trades.csv")

LOOKBACKS=[60,120,252]
MA=25
MA_SLOPE_LOOKBACK=5
VOL_LOOKBACK=20
HOLD_DAYS=[5,10,15]
VOL_MIN=1.5
MA25_SLOPE_MIN=0.25

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
    g["MA25"]=g["C"].rolling(MA).mean()
    g["MA25_SLOPE5_PCT"]=(g["MA25"]/g["MA25"].shift(MA_SLOPE_LOOKBACK)-1)*100
    g["VOL20"]=g["Vo"].rolling(VOL_LOOKBACK).mean()
    g["VOL_RATIO20"]=g["Vo"]/g["VOL20"]
    for lb in LOOKBACKS:
        g[f"PREV_HIGH_{lb}"]=g["C"].shift(1).rolling(lb).max()
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
    max_lb=max(LOOKBACKS)

    for code,g0 in df.groupby("Code",sort=True):
        g=prep(g0)
        start=max(max_lb+1,MA+MA_SLOPE_LOOKBACK,VOL_LOOKBACK)
        for i in range(start,len(g)-max(HOLD_DAYS)-1):
            r=g.iloc[i]
            entry_i=i+1
            entry=float(g["O"].iloc[entry_i])
            if not(np.isfinite(entry) and entry>0):
                continue

            ma_up=pd.notna(r["MA25_SLOPE5_PCT"]) and float(r["MA25_SLOPE5_PCT"])>MA25_SLOPE_MIN
            vol_ok=pd.notna(r["VOL_RATIO20"]) and float(r["VOL_RATIO20"])>=VOL_MIN

            for lb in LOOKBACKS:
                ph=r[f"PREV_HIGH_{lb}"]
                if pd.isna(ph) or not float(r["C"])>float(ph):
                    continue

                rec={
                  "code":code,
                  "signal_date":str(pd.Timestamp(r["Date"]).date()),
                  "entry_date":str(pd.Timestamp(g["Date"].iloc[entry_i]).date()),
                  "lookback":lb,
                  "ma25_slope5_pct":float(r["MA25_SLOPE5_PCT"]) if pd.notna(r["MA25_SLOPE5_PCT"]) else None,
                  "vol_ratio20":float(r["VOL_RATIO20"]) if pd.notna(r["VOL_RATIO20"]) else None,
                  "ma25_rising":bool(ma_up),
                  "volume_1_5x":bool(vol_ok),
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

    out={
      "strategy":"closing-high breakout lookback comparison",
      "universe":"current Prime listings in saved archive",
      "archive_start":str(pd.Timestamp(df["Date"].min()).date()),
      "archive_end":str(pd.Timestamp(df["Date"].max()).date()),
      "split_date":str(split.date()),
      "entry":"signal when close > highest close of prior N sessions; buy next open",
      "exit":"fixed open after 5/10/15 trading sessions",
      "filters":{
        "plain":"no extra filter",
        "ma25_rising_plus_volume":f"MA25 5d slope > {MA25_SLOPE_MIN}% and volume / 20d avg >= {VOL_MIN}x"
      },
      "lookbacks":{},
      "caveat":"Current-listing universe implies survivorship bias. No transaction costs/slippage and no position-overlap constraint."
    }

    for lb in LOOKBACKS:
        base=t[t["lookback"]==lb].copy()
        filt=base[base["ma25_rising"] & base["volume_1_5x"]].copy()
        out["lookbacks"][str(lb)]={
          "plain":{},
          "ma25_rising_plus_volume":{}
        }
        for label,part in [("plain",base),("ma25_rising_plus_volume",filt)]:
            older=part[part.signal_date_dt<split]
            recent=part[part.signal_date_dt>=split]
            for h in HOLD_DAYS:
                out["lookbacks"][str(lb)][label][f"{h}d"]={
                  "overall":metrics(part[f"ret_{h}d_pct"]),
                  "older5":metrics(older[f"ret_{h}d_pct"]),
                  "recent5":metrics(recent[f"ret_{h}d_pct"]),
                }

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    t.drop(columns=["signal_date_dt"]).to_csv(TRADES,index=False)
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
