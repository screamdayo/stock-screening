import json
from pathlib import Path
import numpy as np
import pandas as pd

BATCH_DIR=Path("data/batches")
TRADES=Path("results/gc_entry_neighborhood_earnings_trades.csv")
OUT=Path("results/gc_anchor_earnings_path_analysis.json")

ANCHOR={"vol":0.88,"gap":0.13,"candle":2.14}

def load_archive():
    fs=[]
    for p in sorted(BATCH_DIR.glob("batch_*.parquet")):
        d=pd.read_parquet(p)
        cols=[c for c in ["Code","Date","O","H","L","C","AdjO","AdjH","AdjL","AdjC","ArchiveMarket"] if c in d.columns]
        fs.append(d[cols].copy())
    d=pd.concat(fs,ignore_index=True)
    if "ArchiveMarket" in d.columns:
        d=d[d["ArchiveMarket"]=="プライム"].copy()
    for raw,adj in [("O","AdjO"),("H","AdjH"),("L","AdjL"),("C","AdjC")]:
        if adj in d.columns:
            d[raw]=d[adj].where(d[adj].notna(),d.get(raw))
    for c in ["O","H","L","C"]:
        d[c]=pd.to_numeric(d[c],errors="coerce")
    d["Code"]=d["Code"].astype(str)
    d["Date"]=pd.to_datetime(d["Date"])
    return d.drop_duplicates(["Code","Date"],keep="last").sort_values(["Code","Date"])

def pct(a,b):
    if not(np.isfinite(a) and np.isfinite(b) and a!=0): return None
    return (b/a-1)*100

def r3(x):
    return None if x is None or not np.isfinite(x) else round(float(x),3)

def main():
    t=pd.read_csv(TRADES,dtype={"code":str},parse_dates=["signal_date","entry_date","exit_date","earnings_date"])
    a=t[(t["vol"]>ANCHOR["vol"])&(t["gap"]>ANCHOR["gap"])&(t["candle"]>ANCHOR["candle"])&(t["earnings_cross"]==True)].copy()
    px=load_archive()
    rows=[]
    for _,r in a.iterrows():
        g=px[px["Code"]==r["code"]].sort_values("Date").reset_index(drop=True)
        eidx=g.index[g["Date"]==r["entry_date"]].tolist()
        didx=g.index[g["Date"]==r["earnings_date"]].tolist()
        xidx=g.index[g["Date"]==r["exit_date"]].tolist()
        if not(eidx and didx and xidx): continue
        ei,di,xi=eidx[0],didx[0],xidx[0]
        entry=float(g.loc[ei,"O"])
        exitp=float(g.loc[xi,"O"])
        prev_i=di-1 if di>0 else None
        prev_close=float(g.loc[prev_i,"C"]) if prev_i is not None else np.nan
        earn_open=float(g.loc[di,"O"]); earn_high=float(g.loc[di,"H"]); earn_low=float(g.loc[di,"L"]); earn_close=float(g.loc[di,"C"])
        pre_slice=g.loc[ei:max(ei,di-1)] if di-1>=ei else g.loc[ei:ei]
        max_pre=float(pre_slice["H"].max()) if len(pre_slice) else entry
        min_pre=float(pre_slice["L"].min()) if len(pre_slice) else entry
        pre_ret=pct(entry,prev_close) if di>ei else 0.0
        pre_mfe=pct(entry,max_pre)
        pre_mae=pct(entry,min_pre)
        gap=pct(prev_close,earn_open) if di>0 else None
        event_close=pct(prev_close,earn_close) if di>0 else None
        event_intraday=pct(earn_open,earn_close)
        post_ret=pct(earn_close,exitp)
        total=pct(entry,exitp)
        rows.append({
            "code":r["code"],
            "signal_date":str(r["signal_date"].date()),
            "entry_date":str(r["entry_date"].date()),
            "earnings_date":str(r["earnings_date"].date()),
            "exit_date":str(r["exit_date"].date()),
            "pre_earnings_ret_pct":r3(pre_ret),
            "pre_earnings_mfe_pct":r3(pre_mfe),
            "pre_earnings_mae_pct":r3(pre_mae),
            "earnings_gap_pct":r3(gap),
            "earnings_day_close_vs_prev_close_pct":r3(event_close),
            "earnings_day_intraday_pct":r3(event_intraday),
            "post_earnings_to_exit_pct":r3(post_ret),
            "total_10d_ret_pct":r3(total),
            "sessions_entry_to_earnings":int(di-ei),
            "sessions_earnings_to_exit":int(xi-di),
        })
    d=pd.DataFrame(rows)
    def avg(col):
        s=pd.to_numeric(d[col],errors="coerce").dropna()
        return round(float(s.mean()),3) if len(s) else None
    summary={
      "anchor_rules":ANCHOR,
      "n":int(len(d)),
      "averages":{
        "pre_earnings_ret_pct":avg("pre_earnings_ret_pct"),
        "pre_earnings_mfe_pct":avg("pre_earnings_mfe_pct"),
        "earnings_gap_pct":avg("earnings_gap_pct"),
        "earnings_day_close_vs_prev_close_pct":avg("earnings_day_close_vs_prev_close_pct"),
        "post_earnings_to_exit_pct":avg("post_earnings_to_exit_pct"),
        "total_10d_ret_pct":avg("total_10d_ret_pct")
      },
      "counts":{
        "positive_before_earnings":int((d["pre_earnings_ret_pct"]>0).sum()),
        "positive_earnings_day_vs_prev_close":int((d["earnings_day_close_vs_prev_close_pct"]>0).sum()),
        "positive_after_earnings_to_exit":int((d["post_earnings_to_exit_pct"]>0).sum()),
        "earnings_gap_up":int((d["earnings_gap_pct"]>0).sum()),
      },
      "trades":rows,
      "note":"Pre-earnings return is entry open to close immediately before earnings. Earnings-day move is previous close to earnings-day close. Post-earnings return is earnings-day close to fixed 10-session exit open."
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__": main()

# trigger run 2026-09-22
