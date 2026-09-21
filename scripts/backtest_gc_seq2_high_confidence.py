import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

DATA_DIR = Path("data")
BATCH_DIR = DATA_DIR / "batches"
RESULT_DIR = Path("results")

MA_SHORT=5; MA_LONG=25; SLOPE_LOOKBACK=5; SLOPE_RISING_PCT=0.25
ROLLING_HIGH=60; EPISODE_START_DD=-15.0; EPISODE_RESET_DD=-5.0
TARGET_DD_MIN=-20.0; TARGET_DD_MAX=-15.0; HOLD_DAYS=10

FEATURES = [
    "atr14_pct","dd60_pct","ma25_slope5_pct","avg_turnover20",
    "vol_ratio20","gc_gap_pct","ret5_prior_pct","ret10_prior_pct",
    "ma5_slope1_pct","ma5_slope5_pct","candle_pct",
]


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
        if adj in df.columns: df[raw]=df[adj].where(df[adj].notna(),df.get(raw))
    for c in ["O","H","L","C","Vo"]: df[c]=pd.to_numeric(df[c],errors="coerce")
    df["Code"]=df["Code"].astype(str); df["Date"]=pd.to_datetime(df["Date"])
    return df.drop_duplicates(["Code","Date"],keep="last").sort_values(["Code","Date"]).reset_index(drop=True)


def prep(g):
    g=g.dropna(subset=["O","H","L","C"]).sort_values("Date").reset_index(drop=True).copy()
    g["MA5"]=g["C"].rolling(5).mean(); g["MA25"]=g["C"].rolling(25).mean()
    g["MA25_SLOPE5_PCT"]=(g["MA25"]/g["MA25"].shift(5)-1)*100
    g["MA5_SLOPE1_PCT"]=(g["MA5"]/g["MA5"].shift(1)-1)*100
    g["MA5_SLOPE5_PCT"]=(g["MA5"]/g["MA5"].shift(5)-1)*100
    g["HIGH60"]=g["C"].rolling(60,min_periods=60).max()
    g["DD60_PCT"]=(g["C"]/g["HIGH60"]-1)*100
    g["GC_GAP_PCT"]=(g["MA5"]/g["MA25"]-1)*100
    pc=g["C"].shift(1)
    tr=pd.concat([g["H"]-g["L"],(g["H"]-pc).abs(),(g["L"]-pc).abs()],axis=1).max(axis=1)
    g["ATR14_PCT"]=tr.rolling(14).mean()/g["C"]*100
    g["TURNOVER"]=g["C"]*g["Vo"]; g["AVG_TURNOVER20"]=g["TURNOVER"].rolling(20).mean()
    g["VOL_RATIO20"]=g["Vo"]/g["Vo"].rolling(20).mean()
    g["RET5_PRIOR_PCT"]=(g["C"]/g["C"].shift(5)-1)*100
    g["RET10_PRIOR_PCT"]=(g["C"]/g["C"].shift(10)-1)*100
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


def target(g,i):
    slope=g["MA25_SLOPE5_PCT"].iloc[i]; dd=g["DD60_PCT"].iloc[i]
    return bool(g["GC"].iloc[i]) and pd.notna(slope) and pd.notna(dd) and slope>SLOPE_RISING_PCT and TARGET_DD_MIN<=dd<TARGET_DD_MAX and int(g["GC_SEQ"].iloc[i])==2


def metrics(p):
    r=p["ret10"].dropna()
    if len(r)==0:return {"n":0}
    gp=r[r>0].sum(); gl=-r[r<0].sum()
    return {
        "n":int(len(r)),
        "win_rate_pct":round(float((r>0).mean()*100),2),
        "avg_ret10_pct":round(float(r.mean()),3),
        "median_ret10_pct":round(float(r.median()),3),
        "p5_plus_pct":round(float((r>=5).mean()*100),2),
        "loss5_or_worse_pct":round(float((r<=-5).mean()*100),2),
        "profit_factor":round(float(gp/gl),3) if gl>0 else None,
    }


def main():
    df=load_archive(); rows=[]
    mapping={
        "atr14_pct":"ATR14_PCT","dd60_pct":"DD60_PCT","ma25_slope5_pct":"MA25_SLOPE5_PCT",
        "avg_turnover20":"AVG_TURNOVER20","vol_ratio20":"VOL_RATIO20","gc_gap_pct":"GC_GAP_PCT",
        "ret5_prior_pct":"RET5_PRIOR_PCT","ret10_prior_pct":"RET10_PRIOR_PCT",
        "ma5_slope1_pct":"MA5_SLOPE1_PCT","ma5_slope5_pct":"MA5_SLOPE5_PCT","candle_pct":"CANDLE_PCT",
    }
    for code,g0 in df.groupby("Code",sort=True):
        g=prep(g0)
        for i in range(60,len(g)-HOLD_DAYS-2):
            if not target(g,i): continue
            ei=i+1; xi=ei+HOLD_DAYS
            entry=float(g["O"].iloc[ei]); exitp=float(g["O"].iloc[xi])
            if not(np.isfinite(entry) and np.isfinite(exitp) and entry>0):continue
            rec={"code":str(code),"signal_date":pd.Timestamp(g["Date"].iloc[i]),"ret10":(exitp/entry-1)*100}
            for outcol,incol in mapping.items(): rec[outcol]=g[incol].iloc[i]
            rows.append(rec)
    t=pd.DataFrame(rows).dropna(subset=FEATURES).copy()
    split=pd.Timestamp(df["Date"].min()).normalize()+pd.DateOffset(years=5)
    tr=t[t.signal_date<split].copy(); te=t[t.signal_date>=split].copy()
    Xtr=tr[FEATURES].copy(); Xte=te[FEATURES].copy(); ytr=(tr.ret10>0).astype(int)

    # Shallow tree only; choose the most selective profitable leaf using TRAIN data only.
    tree=DecisionTreeClassifier(max_depth=3,min_samples_leaf=20,random_state=42,class_weight=None)
    tree.fit(Xtr,ytr)
    tr["leaf"]=tree.apply(Xtr); te["leaf"]=tree.apply(Xte)
    leaves=[]
    for leaf,p in tr.groupby("leaf"):
        m=metrics(p)
        if m["n"]>=20 and m["avg_ret10_pct"]>0:
            leaves.append((m["win_rate_pct"],m["avg_ret10_pct"],m["n"],int(leaf),m))
    leaves.sort(reverse=True)
    if not leaves: raise RuntimeError("No eligible leaf")
    _,_,_,best_leaf,best_train=leaves[0]
    train_sel=tr[tr.leaf==best_leaf]; test_sel=te[te.leaf==best_leaf]

    # Also report all leaves to see stability, but selection is frozen from train.
    out={
      "strategy":"GC rising + DD15-20 + exact 2nd GC",
      "goal":"Find a smaller, higher-confidence GC subset; no candidate-ranking score.",
      "split_date":str(split.date()),
      "features":FEATURES,
      "model":"DecisionTreeClassifier max_depth=3, min_samples_leaf=20, trained only on older5; best profitable leaf selected only by older5 win rate, tie-break avg return.",
      "tree_rules":export_text(tree,feature_names=FEATURES),
      "baseline":{"older5":metrics(tr),"recent5":metrics(te)},
      "selected_leaf":best_leaf,
      "selected":{"older5_train":metrics(train_sel),"recent5_test":metrics(test_sel)},
      "train_leaf_table":{str(int(k)):metrics(p) for k,p in tr.groupby("leaf")},
      "recent_leaf_table":{str(int(k)):metrics(p) for k,p in te.groupby("leaf")},
      "caveat":"Current-Prime survivor universe; small sample. Recent5 is untouched by model fitting and leaf selection."
    }
    RESULT_DIR.mkdir(exist_ok=True)
    (RESULT_DIR/"gc_seq2_high_confidence_pseudo_oos.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    t.assign(signal_date=t.signal_date.dt.date.astype(str)).to_csv(RESULT_DIR/"gc_seq2_high_confidence_trades.csv",index=False)
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
