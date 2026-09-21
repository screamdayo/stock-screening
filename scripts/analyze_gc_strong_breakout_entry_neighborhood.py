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

VOL_GRID=[0.85,0.88,0.95]
GAP_GRID=[0.10,0.13,0.16]
CANDLE_GRID=[2.00,2.14,2.30]

VOL_SWEEP=[0.75,0.80,0.85,0.88,0.90,0.95,1.00,1.10]
GAP_SWEEP=[0.05,0.08,0.10,0.13,0.15,0.18,0.20,0.25]
CANDLE_SWEEP=[1.50,1.75,2.00,2.14,2.25,2.50,2.75,3.00]

ANCHOR={"vol":0.88,"gap":0.13,"candle":2.14}


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


def is_base(g,i):
    if not bool(g["GC"].iloc[i]): return False
    slope=g["MA25_SLOPE5_PCT"].iloc[i]; dd=g["DD60_PCT"].iloc[i]
    if pd.isna(slope) or pd.isna(dd): return False
    return slope>SLOPE_RISING_PCT and TARGET_DD_MIN<=dd<TARGET_DD_MAX and int(g["GC_SEQ"].iloc[i])==2


def metrics(df,col):
    s=pd.to_numeric(df[col],errors="coerce").dropna()
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


def subset(t,v,gap,c):
    return t[(t.vol_ratio20>v)&(t.gc_gap_pct>gap)&(t.candle_pct>c)]


def pack(t,split):
    return {
      "overall_9d":metrics(t,"ret9"),
      "overall_10d":metrics(t,"ret10"),
      "older5_10d":metrics(t[t.signal_date<split],"ret10"),
      "recent5_10d":metrics(t[t.signal_date>=split],"ret10"),
    }


def main():
    df=load_archive(); rows=[]
    for code,g0 in df.groupby("Code",sort=True):
        g=prep(g0)
        for i in range(60,len(g)-13):
            if not is_base(g,i): continue
            ei=i+1; entry=float(g["O"].iloc[ei])
            if not(np.isfinite(entry) and entry>0): continue
            r9=float(g["O"].iloc[ei+9]); r10=float(g["O"].iloc[ei+10])
            rows.append({
              "code":str(code),"signal_date":pd.Timestamp(g["Date"].iloc[i]),
              "vol_ratio20":g["VOL_RATIO20"].iloc[i],
              "gc_gap_pct":g["GC_GAP_PCT"].iloc[i],
              "candle_pct":g["CANDLE_PCT"].iloc[i],
              "ret9":(r9/entry-1)*100 if np.isfinite(r9) else np.nan,
              "ret10":(r10/entry-1)*100 if np.isfinite(r10) else np.nan,
            })
    t=pd.DataFrame(rows).dropna()
    split=pd.Timestamp(df["Date"].min()).normalize()+pd.DateOffset(years=5)

    local=[]
    for v in VOL_GRID:
      for gap in GAP_GRID:
        for c in CANDLE_GRID:
          s=subset(t,v,gap,c)
          rec={"vol_gt":v,"gap_gt":gap,"candle_gt":c}
          rec.update(pack(s,split))
          local.append(rec)

    sweeps={"volume":[],"gap":[],"candle":[]}
    for v in VOL_SWEEP:
      s=subset(t,v,ANCHOR["gap"],ANCHOR["candle"])
      sweeps["volume"].append({"threshold":v,**pack(s,split)})
    for gap in GAP_SWEEP:
      s=subset(t,ANCHOR["vol"],gap,ANCHOR["candle"])
      sweeps["gap"].append({"threshold":gap,**pack(s,split)})
    for c in CANDLE_SWEEP:
      s=subset(t,ANCHOR["vol"],ANCHOR["gap"],c)
      sweeps["candle"].append({"threshold":c,**pack(s,split)})

    anchor=subset(t,ANCHOR["vol"],ANCHOR["gap"],ANCHOR["candle"])
    out={
      "strategy":"GC seq2 strong-breakout entry neighborhood robustness",
      "base_signal_count":int(len(t)),
      "anchor_rules":{"vol_ratio20_gt":0.88,"gc_gap_pct_gt":0.13,"candle_pct_gt":2.14},
      "anchor":pack(anchor,split),
      "local_grid_values":{"vol":VOL_GRID,"gap":GAP_GRID,"candle":CANDLE_GRID},
      "local_grid":local,
      "one_at_a_time_sweeps":sweeps,
      "split_date":str(split.date()),
      "interpretation_rule":"Robust if nearby thresholds preserve similar sample size/direction and performance rather than only the exact anchor cell being strong.",
      "caveat":"Anchor rules were found post-hoc. This tests neighborhood stability, not untouched OOS validation."
    }
    RESULT_DIR.mkdir(exist_ok=True)
    (RESULT_DIR/"gc_strong_breakout_entry_neighborhood.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
