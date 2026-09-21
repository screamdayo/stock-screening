import json
from pathlib import Path
import numpy as np
import pandas as pd

BATCH_DIR=Path("data/batches")
SRC=Path("results/gc_earnings_presignal_comparison.json")
OUT=Path("results/gc_earnings_days_to_event_analysis.json")

def load_archive():
    fs=[]
    for p in sorted(BATCH_DIR.glob("batch_*.parquet")):
        d=pd.read_parquet(p)
        cols=[c for c in ["Code","Date","O","C","AdjO","AdjC","ArchiveMarket"] if c in d.columns]
        fs.append(d[cols].copy())
    d=pd.concat(fs,ignore_index=True)
    if "ArchiveMarket" in d.columns:
        d=d[d["ArchiveMarket"]=="プライム"].copy()
    for raw,adj in [("O","AdjO"),("C","AdjC")]:
        if adj in d.columns:
            d[raw]=d[adj].where(d[adj].notna(),d.get(raw))
    d["Code"]=d["Code"].astype(str)
    d["Date"]=pd.to_datetime(d["Date"])
    return d.drop_duplicates(["Code","Date"],keep="last").sort_values(["Code","Date"])

def stats(rows):
    if not rows:
        return {"n":0}
    s=pd.Series([r["next_open_gap_pct"] for r in rows],dtype=float)
    return {
      "n":int(len(s)),
      "gap_up_rate_pct":round(float((s>0).mean()*100),2),
      "avg_gap_pct":round(float(s.mean()),3),
      "median_gap_pct":round(float(s.median()),3),
      "gap_5pct_plus_rate_pct":round(float((s>=5).mean()*100),2),
      "gap_minus5pct_or_worse_rate_pct":round(float((s<=-5).mean()*100),2),
    }

def main():
    src=json.loads(SRC.read_text())
    px=load_archive()
    rows=[]
    for e in src["strong_events"]:
        code=str(e["code"])
        sig=pd.Timestamp(e["signal_date"])
        er=pd.Timestamp(e["earnings_date"])
        g=px[px["Code"]==code].sort_values("Date").reset_index(drop=True)
        si=g.index[g["Date"]==sig].tolist()
        ei=g.index[g["Date"]==er].tolist()
        if not(si and ei): continue
        days=int(ei[0]-si[0])
        rows.append({
          **e,
          "trading_days_signal_to_earnings":days,
          "bucket":"1-3" if days<=3 else ("4-6" if days<=6 else "7-10")
        })
    rows=sorted(rows,key=lambda x:(x["trading_days_signal_to_earnings"],x["signal_date"]))

    buckets={}
    for b in ["1-3","4-6","7-10"]:
        part=[r for r in rows if r["bucket"]==b]
        buckets[b]=stats(part)

    by_day={}
    for d in sorted({r["trading_days_signal_to_earnings"] for r in rows}):
        by_day[str(d)]=stats([r for r in rows if r["trading_days_signal_to_earnings"]==d])

    out={
      "definition":"Trading days from GC strong-breakout signal date to scheduled earnings date. Earnings reaction remains SchDate close to next-session open under the after-close assumption.",
      "overall":stats(rows),
      "distribution_counts":{str(d):sum(r["trading_days_signal_to_earnings"]==d for r in rows) for d in range(1,11)},
      "buckets":buckets,
      "by_exact_day":by_day,
      "events":rows,
      "caveat":"Only 8 strong-breakout earnings events; bucket results are descriptive, not statistically stable."
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
