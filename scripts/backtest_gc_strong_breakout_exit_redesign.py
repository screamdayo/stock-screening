import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
BATCH_DIR = DATA_DIR / "batches"
RESULT_DIR = Path("results")

MA_SHORT=5; MA_LONG=25; SLOPE_LOOKBACK=5; SLOPE_RISING_PCT=0.25
ROLLING_HIGH=60; EPISODE_START_DD=-15.0; EPISODE_RESET_DD=-5.0
TARGET_DD_MIN=-20.0; TARGET_DD_MAX=-15.0

VOL_RATIO_MIN=0.88
GC_GAP_MIN=0.13
CANDLE_MIN=2.14
MAX_HOLD=20


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


def is_strong_signal(g,i):
    if not bool(g["GC"].iloc[i]): return False
    slope=g["MA25_SLOPE5_PCT"].iloc[i]; dd=g["DD60_PCT"].iloc[i]
    if pd.isna(slope) or pd.isna(dd): return False
    base=(slope>SLOPE_RISING_PCT and TARGET_DD_MIN<=dd<TARGET_DD_MAX and int(g["GC_SEQ"].iloc[i])==2)
    if not base: return False
    vals=[g["VOL_RATIO20"].iloc[i],g["GC_GAP_PCT"].iloc[i],g["CANDLE_PCT"].iloc[i]]
    if any(pd.isna(v) for v in vals): return False
    return vals[0]>VOL_RATIO_MIN and vals[1]>GC_GAP_MIN and vals[2]>CANDLE_MIN


def result(g,si,ei,xi,entry,exitp,reason):
    return {
        "code":str(g["Code"].iloc[si]),
        "signal_date":str(pd.Timestamp(g["Date"].iloc[si]).date()),
        "entry_date":str(pd.Timestamp(g["Date"].iloc[ei]).date()),
        "exit_date":str(pd.Timestamp(g["Date"].iloc[xi]).date()),
        "return_pct":(exitp/entry-1)*100,
        "hold_sessions":int(xi-ei),
        "exit_reason":reason,
    }


def fixed(g,si,days):
    ei=si+1; xi=ei+days
    if xi>=len(g): return None
    entry=float(g["O"].iloc[ei]); exitp=float(g["O"].iloc[xi])
    if not(np.isfinite(entry) and entry>0 and np.isfinite(exitp)): return None
    return result(g,si,ei,xi,entry,exitp,f"fixed{days}")


def close_rule(g,si,kind,max_hold=MAX_HOLD):
    ei=si+1; last=min(ei+max_hold,len(g)-1)
    entry=float(g["O"].iloc[ei])
    if not(np.isfinite(entry) and entry>0): return None
    # Signal is confirmed on close, execute next open.
    for i in range(ei,last):
        hit=False
        if kind=="close_below_ma5":
            hit=pd.notna(g["MA5"].iloc[i]) and float(g["C"].iloc[i]) < float(g["MA5"].iloc[i])
        elif kind=="ma5_turn_down":
            hit=i>=1 and pd.notna(g["MA5"].iloc[i-1]) and pd.notna(g["MA5"].iloc[i]) and float(g["MA5"].iloc[i]) < float(g["MA5"].iloc[i-1])
        elif kind=="close_below_ma25":
            hit=pd.notna(g["MA25"].iloc[i]) and float(g["C"].iloc[i]) < float(g["MA25"].iloc[i])
        if hit:
            xi=i+1; exitp=float(g["O"].iloc[xi])
            return result(g,si,ei,xi,entry,exitp,kind+"_next_open")
    xi=last; exitp=float(g["O"].iloc[xi])
    return result(g,si,ei,xi,entry,exitp,f"max{max_hold}")


def trailing(g,si,trail_pct,activate_pct=0.0,max_hold=MAX_HOLD):
    ei=si+1; last=min(ei+max_hold,len(g)-1)
    entry=float(g["O"].iloc[ei])
    if not(np.isfinite(entry) and entry>0): return None
    peak=entry; active=activate_pct<=0
    for i in range(ei,last):
        o=float(g["O"].iloc[i]); h=float(g["H"].iloc[i]); l=float(g["L"].iloc[i])
        if np.isfinite(h):
            peak=max(peak,h)
        if not active and peak>=entry*(1+activate_pct/100):
            active=True
        if active:
            stop=peak*(1-trail_pct/100)
            if np.isfinite(o) and o<=stop:
                return result(g,si,ei,i,entry,o,f"trail{trail_pct:g}_gap")
            if np.isfinite(l) and l<=stop:
                return result(g,si,ei,i,entry,stop,f"trail{trail_pct:g}")
    xi=last; exitp=float(g["O"].iloc[xi])
    return result(g,si,ei,xi,entry,exitp,f"max{max_hold}")


def tp_then_trail(g,si,tp_pct,trail_pct,max_hold=MAX_HOLD):
    # Before activation: no stop. Once intraday high reaches +TP, trail from the running peak.
    return trailing(g,si,trail_pct,activate_pct=tp_pct,max_hold=max_hold)


def metrics(df):
    if df.empty:return {"n":0}
    r=df["return_pct"].dropna(); gp=r[r>0].sum(); gl=-r[r<0].sum()
    return {
        "n":int(len(r)),
        "win_rate_pct":round(float((r>0).mean()*100),2),
        "avg_return_pct":round(float(r.mean()),3),
        "median_return_pct":round(float(r.median()),3),
        "p5_plus_pct":round(float((r>=5).mean()*100),2),
        "loss5_or_worse_pct":round(float((r<=-5).mean()*100),2),
        "profit_factor":round(float(gp/gl),3) if gl>0 else None,
        "avg_hold_sessions":round(float(df["hold_sessions"].mean()),2),
        "worst_return_pct":round(float(r.min()),3),
    }


def summarize(df,split):
    d=df.copy(); d["signal_date_dt"]=pd.to_datetime(d["signal_date"])
    return {
        "overall":metrics(d),
        "older5":metrics(d[d.signal_date_dt<split]),
        "recent5":metrics(d[d.signal_date_dt>=split]),
        "exit_reasons":{str(k):int(v) for k,v in d["exit_reason"].value_counts().to_dict().items()},
    }


def main():
    df=load_archive()
    variants={
        "fixed5":lambda g,i:fixed(g,i,5),
        "fixed10":lambda g,i:fixed(g,i,10),
        "fixed15":lambda g,i:fixed(g,i,15),
        "fixed20":lambda g,i:fixed(g,i,20),
        "ma5_turn_down_max20":lambda g,i:close_rule(g,i,"ma5_turn_down"),
        "close_below_ma5_max20":lambda g,i:close_rule(g,i,"close_below_ma5"),
        "close_below_ma25_max20":lambda g,i:close_rule(g,i,"close_below_ma25"),
        "trail5_max20":lambda g,i:trailing(g,i,5.0),
        "trail7_max20":lambda g,i:trailing(g,i,7.0),
        "plus5_then_trail3_max20":lambda g,i:tp_then_trail(g,i,5.0,3.0),
        "plus5_then_trail5_max20":lambda g,i:tp_then_trail(g,i,5.0,5.0),
        "plus7_then_trail3_max20":lambda g,i:tp_then_trail(g,i,7.0,3.0),
    }
    rows={k:[] for k in variants}
    for code,g0 in df.groupby("Code",sort=True):
        g=prep(g0)
        for i in range(60,len(g)-MAX_HOLD-2):
            if not is_strong_signal(g,i): continue
            for name,fn in variants.items():
                x=fn(g,i)
                if x: rows[name].append(x)

    split=pd.Timestamp(df["Date"].min()).normalize()+pd.DateOffset(years=5)
    out={
        "strategy":"GC seq2 strong-breakout subset",
        "selection_rules":{"vol_ratio20_gt":VOL_RATIO_MIN,"gc_gap_pct_gt":GC_GAP_MIN,"candle_pct_gt":CANDLE_MIN},
        "purpose":"Rebuild exits using price/MA behavior instead of assuming fixed 10 days.",
        "execution_note":"Close/MA rules are confirmed at close and executed next open. Trailing stops use daily OHLC; gap-through exits at open, otherwise at stop level. Activated trailing has no protective stop before activation.",
        "split_date":str(split.date()),
        "variants":{},
        "caveat":"The strong-breakout entry subset itself is post-hoc. Exit variants are exploratory and should not be treated as untouched OOS optimization."
    }
    frames=[]
    for name,recs in rows.items():
        t=pd.DataFrame(recs)
        out["variants"][name]=summarize(t,split) if len(t) else {"overall":{"n":0}}
        if len(t):
            t["variant"]=name; frames.append(t)
    RESULT_DIR.mkdir(exist_ok=True)
    (RESULT_DIR/"gc_strong_breakout_exit_redesign_summary.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    if frames:
        pd.concat(frames,ignore_index=True).to_csv(RESULT_DIR/"gc_strong_breakout_exit_redesign_trades.csv",index=False)
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
