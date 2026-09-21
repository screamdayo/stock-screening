import json
from pathlib import Path
import numpy as np
import pandas as pd

BATCH_DIR=Path("data/batches")
TRADES=Path("results/gc_entry_neighborhood_earnings_trades.csv")
OUT=Path("results/gc_anchor_earnings_path_after_close_assumption.json")

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
        earn_close=float(g.loc[di,"C"])
        next_i=di+1 if di+1<len(g) else None
        next_open=float(g.loc[next_i,"O"]) if next_i is not None else np.nan
        next_close=float(g.loc[next_i,"C"]) if next_i is not None else np.nan
        exitp=float(g.loc[xi,"O"])

        # Conservative assumption: earnings are released after market close on SchDate.
        # Therefore all price action through SchDate close is pre-announcement.
        pre_ret=pct(entry,earn_close)
        reaction_gap=pct(earn_close,next_open) if next_i is not None else None
        reaction_day_close=pct(earn_close,next_close) if next_i is not None else None
        post_nextopen_to_exit=pct(next_open,exitp) if next_i is not None else None
        total=pct(entry,exitp)

        rows.append({
            "code":r["code"],
            "signal_date":str(r["signal_date"].date()),
            "entry_date":str(r["entry_date"].date()),
            "earnings_date":str(r["earnings_date"].date()),
            "next_session":str(g.loc[next_i,"Date"].date()) if next_i is not None else None,
            "exit_date":str(r["exit_date"].date()),
            "pre_announcement_to_earnings_close_pct":r3(pre_ret),
            "next_open_gap_pct":r3(reaction_gap),
            "next_session_close_vs_earnings_close_pct":r3(reaction_day_close),
            "post_reaction_next_open_to_exit_pct":r3(post_nextopen_to_exit),
            "total_10d_ret_pct":r3(total),
            "sessions_entry_to_earnings":int(di-ei),
            "sessions_next_session_to_exit":int(xi-next_i) if next_i is not None else None,
        })
    d=pd.DataFrame(rows)
    def avg(col):
        s=pd.to_numeric(d[col],errors="coerce").dropna()
        return round(float(s.mean()),3) if len(s) else None
    summary={
      "assumption":"Treat scheduled earnings as released after market close on SchDate. Thus price through SchDate close is pre-announcement, and the next trading session is the first earnings-reaction session.",
      "anchor_rules":ANCHOR,
      "n":int(len(d)),
      "averages":{
        "pre_announcement_to_earnings_close_pct":avg("pre_announcement_to_earnings_close_pct"),
        "next_open_gap_pct":avg("next_open_gap_pct"),
        "next_session_close_vs_earnings_close_pct":avg("next_session_close_vs_earnings_close_pct"),
        "post_reaction_next_open_to_exit_pct":avg("post_reaction_next_open_to_exit_pct"),
        "total_10d_ret_pct":avg("total_10d_ret_pct")
      },
      "counts":{
        "positive_before_announcement":int((d["pre_announcement_to_earnings_close_pct"]>0).sum()),
        "next_open_gap_up":int((d["next_open_gap_pct"]>0).sum()),
        "next_session_positive_vs_earnings_close":int((d["next_session_close_vs_earnings_close_pct"]>0).sum()),
        "positive_after_next_open_to_exit":int((d["post_reaction_next_open_to_exit_pct"]>0).sum())
      },
      "trades":rows,
      "caveat":"Actual announcement times are not available here. Some companies announce during market hours, so this after-close treatment is a conservative simplifying assumption, not a verified event-time reconstruction."
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
