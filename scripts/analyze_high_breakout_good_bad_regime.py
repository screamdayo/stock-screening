import json
from pathlib import Path
import numpy as np
import pandas as pd

BATCH_DIR=Path("data/batches")
TRADES=Path("results/high_breakout_market_regime_trades.csv")
OUT=Path("results/high_breakout_good_bad_regime_diagnostics.json")

GOOD_YEARS={2017,2025,2026}
BAD_YEARS={2018,2021,2022}

def load_archive():
    fs=[]
    for p in sorted(BATCH_DIR.glob("batch_*.parquet")):
        d=pd.read_parquet(p)
        cols=[c for c in ["Code","Date","C","AdjC","ArchiveMarket"] if c in d.columns]
        fs.append(d[cols].copy())
    if not fs: raise RuntimeError("No archive batches")
    d=pd.concat(fs,ignore_index=True)
    if "ArchiveMarket" in d.columns:
        d=d[d["ArchiveMarket"]=="プライム"].copy()
    if "AdjC" in d.columns:
        d["C"]=d["AdjC"].where(d["AdjC"].notna(),d["C"])
    d["C"]=pd.to_numeric(d["C"],errors="coerce")
    d["Code"]=d["Code"].astype(str)
    d["Date"]=pd.to_datetime(d["Date"])
    d=d.dropna(subset=["C"]).drop_duplicates(["Code","Date"],keep="last")
    return d.sort_values(["Code","Date"]).reset_index(drop=True)

def build_market_proxy(df):
    parts=[]
    for code,g in df.groupby("Code",sort=False):
        g=g.sort_values("Date").copy()
        g["ret1"]=g["C"].pct_change()
        g["ma25"]=g["C"].rolling(25).mean()
        g["ma60"]=g["C"].rolling(60).mean()
        g["above25"]=g["C"]>g["ma25"]
        g["above60"]=g["C"]>g["ma60"]
        parts.append(g[["Date","ret1","above25","above60"]])
    x=pd.concat(parts,ignore_index=True)
    m=x.groupby("Date").agg(
        ew_ret1=("ret1","mean"),
        median_ret1=("ret1","median"),
        breadth25=("above25","mean"),
        breadth60=("above60","mean"),
        n=("ret1","count"),
    ).reset_index().sort_values("Date")

    # Equal-weight proxy index based only on same-day constituent returns.
    m["proxy"]=(1+m["ew_ret1"].fillna(0)).cumprod()
    for w in [25,60,120]:
        m[f"proxy_ret{w}_pct"]=(m["proxy"]/m["proxy"].shift(w)-1)*100
    m["proxy_ma25"]=m["proxy"].rolling(25).mean()
    m["proxy_ma100"]=m["proxy"].rolling(100).mean()
    m["proxy_ma25_vs_100_pct"]=(m["proxy_ma25"]/m["proxy_ma100"]-1)*100
    m["proxy_above_ma100"]=m["proxy"]>m["proxy_ma100"]
    m["proxy_ma25_above_ma100"]=m["proxy_ma25"]>m["proxy_ma100"]
    m["proxy_vol20_ann_pct"]=m["ew_ret1"].rolling(20).std()*np.sqrt(252)*100
    m["breadth25_pct"]=m["breadth25"]*100
    m["breadth60_pct"]=m["breadth60"]*100
    return m

def stats(s):
    s=pd.to_numeric(s,errors="coerce").dropna()
    if s.empty:return {"n":0}
    return {
      "n":int(len(s)),
      "mean":round(float(s.mean()),3),
      "median":round(float(s.median()),3),
      "p25":round(float(s.quantile(.25)),3),
      "p75":round(float(s.quantile(.75)),3),
    }

def perf(df):
    s=pd.to_numeric(df["ret_15d_pct"],errors="coerce").dropna()
    if s.empty:return {"n":0}
    pos=s[s>0];neg=s[s<0];gl=-neg.sum()
    pf=pos.sum()/gl if gl>0 else None
    return {
      "n":int(len(s)),
      "win_rate_pct":round(float((s>0).mean()*100),2),
      "avg_return_pct":round(float(s.mean()),3),
      "median_return_pct":round(float(s.median()),3),
      "profit_factor":round(float(pf),3) if pf is not None else None,
    }

def main():
    market=build_market_proxy(load_archive()).rename(columns={"Date":"signal_date"})
    t=pd.read_csv(TRADES,dtype={"code":str})
    t["signal_date"]=pd.to_datetime(t["signal_date"])
    t=t.merge(market,on="signal_date",how="left")
    t["year"]=t["signal_date"].dt.year
    t["group"]=np.where(t["year"].isin(GOOD_YEARS),"good",
                np.where(t["year"].isin(BAD_YEARS),"bad","other"))

    features=[
      "proxy_ret25_pct","proxy_ret60_pct","proxy_ret120_pct",
      "proxy_ma25_vs_100_pct","proxy_vol20_ann_pct",
      "breadth25_pct","breadth60_pct"
    ]

    out={
      "strategy":"252d breakout + MA25 rising + volume >=1.5x, 15d exit",
      "good_years":sorted(GOOD_YEARS),
      "bad_years":sorted(BAD_YEARS),
      "market_proxy":"equal-weight current-Prime-survivor daily return proxy; descriptive, not actual TOPIX",
      "feature_comparison":{},
      "candidate_filters":{},
      "performance_by_group":{
        g:perf(x) for g,x in t.groupby("group")
      },
      "note":"Exploratory diagnostics only. Current-listing survivorship bias remains, and proxy market features are not actual historical TOPIX."
    }

    for f in features:
        out["feature_comparison"][f]={
          g:stats(x[f]) for g,x in t.groupby("group")
        }

    candidates={
      "proxy_ret60_gt_0":t["proxy_ret60_pct"]>0,
      "proxy_ret60_gt_5":t["proxy_ret60_pct"]>5,
      "proxy_ret120_gt_0":t["proxy_ret120_pct"]>0,
      "proxy_above_ma100":t["proxy_above_ma100"].fillna(False),
      "proxy_ma25_above_ma100":t["proxy_ma25_above_ma100"].fillna(False),
      "proxy_ma25_vs100_gt_0":t["proxy_ma25_vs_100_pct"]>0,
      "proxy_ma25_vs100_gt_2":t["proxy_ma25_vs_100_pct"]>2,
      "vol20_lt_15":t["proxy_vol20_ann_pct"]<15,
      "vol20_lt_20":t["proxy_vol20_ann_pct"]<20,
      "ret60_pos_and_ma25_above100":(t["proxy_ret60_pct"]>0)&t["proxy_ma25_above_ma100"].fillna(False),
      "ret120_pos_and_ma25_above100":(t["proxy_ret120_pct"]>0)&t["proxy_ma25_above_ma100"].fillna(False),
    }

    for name,mask in candidates.items():
        part=t[mask].copy()
        out["candidate_filters"][name]={
          "overall":perf(part),
          "good_years":perf(part[part["group"]=="good"]),
          "bad_years":perf(part[part["group"]=="bad"]),
          "kept_pct":round(float(len(part)/len(t)*100),2) if len(t) else None,
          "yearly":{str(int(y)):perf(g) for y,g in part.groupby("year")}
        }

    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
