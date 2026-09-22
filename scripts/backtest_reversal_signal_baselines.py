import json
from pathlib import Path
import numpy as np
import pandas as pd

BATCH_DIR=Path("data/batches")
OUT=Path("results/reversal_signal_baselines.json")

HOLD=[5,10,15]

def load_archive():
    fs=[]
    for p in sorted(BATCH_DIR.glob("batch_*.parquet")):
        d=pd.read_parquet(p)
        cols=[c for c in ["Code","Date","O","H","L","C","Vo","AdjO","AdjH","AdjL","AdjC","AdjVo","ArchiveMarket"] if c in d.columns]
        fs.append(d[cols].copy())
    if not fs: raise RuntimeError("No archive")
    d=pd.concat(fs,ignore_index=True)
    if "ArchiveMarket" in d.columns:
        d=d[d["ArchiveMarket"]=="プライム"].copy()
    for raw,adj in [("O","AdjO"),("H","AdjH"),("L","AdjL"),("C","AdjC"),("Vo","AdjVo")]:
        if adj in d.columns: d[raw]=d[adj].where(d[adj].notna(),d.get(raw))
    for c in ["O","H","L","C","Vo"]: d[c]=pd.to_numeric(d[c],errors="coerce")
    d["Code"]=d["Code"].astype(str); d["Date"]=pd.to_datetime(d["Date"])
    return d.dropna(subset=["O","H","L","C"]).drop_duplicates(["Code","Date"],keep="last").sort_values(["Code","Date"])

def prep(g):
    g=g.sort_values("Date").reset_index(drop=True).copy()
    prev=g["C"].shift(1)
    tr=pd.concat([(g["H"]-g["L"]).abs(),(g["H"]-prev).abs(),(g["L"]-prev).abs()],axis=1).max(axis=1)
    g["ATR14"]=tr.rolling(14).mean()
    g["ATR14_PCT"]=g["ATR14"]/g["C"]*100
    g["MA5"]=g["C"].rolling(5).mean()
    g["MA25"]=g["C"].rolling(25).mean()
    g["RET5"]=(g["C"]/g["C"].shift(5)-1)*100
    g["RET10"]=(g["C"]/g["C"].shift(10)-1)*100
    g["RET20"]=(g["C"]/g["C"].shift(20)-1)*100
    g["DD20"]=(g["C"]/g["C"].rolling(20).max()-1)*100
    g["RANGE_PCT"]=(g["H"]-g["L"])/g["C"]*100
    g["RANGE5"]=g["RANGE_PCT"].rolling(5).mean()
    g["RANGE20"]=g["RANGE_PCT"].rolling(20).mean()
    g["VOL20"]=g["Vo"].rolling(20).mean()
    g["VOLR"]=g["Vo"]/g["VOL20"]
    g["CANDLE"]=(g["C"]/g["O"]-1)*100
    g["MA5_SLOPE1"]=(g["MA5"]/g["MA5"].shift(1)-1)*100
    return g

def met(s):
    s=pd.Series(s,dtype=float).dropna()
    if s.empty:return {"n":0}
    pos=s[s>0];neg=s[s<0];gl=-neg.sum();pf=pos.sum()/gl if gl>0 else None
    return {"n":int(len(s)),"win_rate_pct":round(float((s>0).mean()*100),2),
            "avg_return_pct":round(float(s.mean()),3),"median_return_pct":round(float(s.median()),3),
            "profit_factor":round(float(pf),3) if pf is not None else None}

def main():
    df=load_archive(); rows=[]
    for code,g0 in df.groupby("Code",sort=True):
        g=prep(g0)
        for i in range(30,len(g)-max(HOLD)-1):
            r=g.iloc[i]
            # Two intentionally simple families; no optimization yet.
            sharp=(r["RET5"]<=-8) and (r["DD20"]<=-10) and (r["CANDLE"]>=1.0)
            squeeze=(r["RANGE5"]<=0.65*r["RANGE20"]) and (r["RET20"]<=-3) and (r["C"]>r["MA5"]) and (r["MA5_SLOPE1"]>0)
            if not (sharp or squeeze): continue
            ei=i+1; entry=float(g["O"].iloc[ei])
            if not(np.isfinite(entry) and entry>0):continue
            for typ,ok in [("sharp_drop_reversal",sharp),("volatility_squeeze_reversal",squeeze)]:
                if not ok:continue
                rec={"code":code,"signal_date":str(pd.Timestamp(r["Date"]).date()),"type":typ,
                     "ret5_before_pct":float(r["RET5"]),"ret20_before_pct":float(r["RET20"]),
                     "dd20_pct":float(r["DD20"]),"atr14_pct":float(r["ATR14_PCT"]),
                     "range5_vs20":float(r["RANGE5"]/r["RANGE20"]) if r["RANGE20"] else None,
                     "candle_pct":float(r["CANDLE"]),"vol_ratio20":float(r["VOLR"]) if pd.notna(r["VOLR"]) else None}
                for h in HOLD:
                    xp=float(g["O"].iloc[ei+h]); rec[f"ret_{h}d_pct"]=(xp/entry-1)*100
                rows.append(rec)
    t=pd.DataFrame(rows); t["signal_date_dt"]=pd.to_datetime(t["signal_date"])
    split=pd.Timestamp(df["Date"].min()).normalize()+pd.DateOffset(years=5)
    out={"study":"simple reversal baselines","split_date":str(split.date()),"families":{},
         "note":"Exploratory baselines only; thresholds intentionally coarse. Current-Prime survivor bias, no costs/slippage/overlap constraints."}
    for typ,p in t.groupby("type"):
        old=p[p.signal_date_dt<split]; recent=p[p.signal_date_dt>=split]
        out["families"][typ]={f"{h}d":{"overall":met(p[f"ret_{h}d_pct"]),"older5":met(old[f"ret_{h}d_pct"]),"recent5":met(recent[f"ret_{h}d_pct"])} for h in HOLD}
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    Path("results/reversal_signal_baseline_trades.csv").write_text(t.drop(columns=["signal_date_dt"]).to_csv(index=False),encoding="utf-8")
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=="__main__":main()

# trigger 2026-09-22
