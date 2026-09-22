import json
from pathlib import Path
import pandas as pd
import numpy as np

ALL_TRADES=Path("results/high_breakout_60_120_252d_trades.csv")
TARGET_TRADES=Path("results/high_breakout_market_regime_trades.csv")
OUT=Path("results/high_breakout_signal_crowding.json")

def metrics(df):
    if df.empty:
        return {"n":0}
    s=pd.to_numeric(df["ret_15d_pct"],errors="coerce").dropna()
    if s.empty:
        return {"n":0}
    pos=s[s>0]; neg=s[s<0]
    gl=-neg.sum()
    pf=pos.sum()/gl if gl>0 else None
    return {
        "n":int(len(s)),
        "win_rate_pct":round(float((s>0).mean()*100),2),
        "avg_return_pct":round(float(s.mean()),3),
        "median_return_pct":round(float(s.median()),3),
        "profit_factor":round(float(pf),3) if pf is not None else None,
    }

def summarize_by_bins(t, count_col, bins, labels):
    x=t.copy()
    x["bucket"]=pd.cut(x[count_col],bins=bins,labels=labels,include_lowest=True,right=True)
    return {str(k):metrics(g) for k,g in x.groupby("bucket",observed=False)}

def main():
    all_t=pd.read_csv(ALL_TRADES,dtype={"code":str})
    target=pd.read_csv(TARGET_TRADES,dtype={"code":str})
    for d in (all_t,target):
        d["signal_date"]=pd.to_datetime(d["signal_date"])

    plain252=all_t[all_t["lookback"]==252].copy()
    filtered252=plain252[
        plain252["ma25_rising"].astype(bool) &
        plain252["volume_1_5x"].astype(bool)
    ].copy()

    plain_counts=plain252.groupby("signal_date").size().rename("plain_252_count")
    filtered_counts=filtered252.groupby("signal_date").size().rename("filtered_252_count")

    t=target.merge(plain_counts,on="signal_date",how="left").merge(filtered_counts,on="signal_date",how="left")
    t["plain_252_count"]=t["plain_252_count"].fillna(0).astype(int)
    t["filtered_252_count"]=t["filtered_252_count"].fillna(0).astype(int)
    t["year"]=t["signal_date"].dt.year

    # Quantiles are reported descriptively; fixed thresholds below are more useful operationally.
    q_plain={str(q):round(float(t["plain_252_count"].quantile(q)),2) for q in [0,.25,.5,.75,.9,.95,.99,1]}
    q_filt={str(q):round(float(t["filtered_252_count"].quantile(q)),2) for q in [0,.25,.5,.75,.9,.95,.99,1]}

    out={
      "strategy":"252d closing-high breakout + MA25 rising + volume >=1.5x; 15d exit",
      "question":"Can crowded breakout days identify losing setups?",
      "count_definitions":{
        "plain_252_count":"number of all 252d closing-high breakouts on the same signal date",
        "filtered_252_count":"number of 252d breakouts also passing MA25 rising + volume >=1.5x on the same signal date",
      },
      "count_quantiles":{
        "plain_252_count":q_plain,
        "filtered_252_count":q_filt,
      },
      "baseline":metrics(t),
      "plain_count_buckets":summarize_by_bins(
        t,"plain_252_count",
        [-np.inf,10,25,50,100,200,np.inf],
        ["<=10","11-25","26-50","51-100","101-200",">200"]
      ),
      "filtered_count_buckets":summarize_by_bins(
        t,"filtered_252_count",
        [-np.inf,3,5,10,20,40,np.inf],
        ["<=3","4-5","6-10","11-20","21-40",">40"]
      ),
      "filters":{
        "plain_count_le_25":metrics(t[t["plain_252_count"]<=25]),
        "plain_count_le_50":metrics(t[t["plain_252_count"]<=50]),
        "plain_count_le_100":metrics(t[t["plain_252_count"]<=100]),
        "exclude_plain_top10pct":metrics(t[t["plain_252_count"]<=t["plain_252_count"].quantile(.90)]),
        "filtered_count_le_5":metrics(t[t["filtered_252_count"]<=5]),
        "filtered_count_le_10":metrics(t[t["filtered_252_count"]<=10]),
        "filtered_count_le_20":metrics(t[t["filtered_252_count"]<=20]),
        "exclude_filtered_top10pct":metrics(t[t["filtered_252_count"]<=t["filtered_252_count"].quantile(.90)]),
      },
      "yearly_baseline":{str(int(y)):metrics(g) for y,g in t.groupby("year")},
      "yearly_exclude_plain_top10pct":{},
      "yearly_exclude_filtered_top10pct":{},
      "note":"Exploratory. Quantile cutoffs are derived from the same sample and are not untouched OOS. Current-listing survivorship bias remains."
    }

    p90=t["plain_252_count"].quantile(.90)
    f90=t["filtered_252_count"].quantile(.90)
    for y,g in t.groupby("year"):
        out["yearly_exclude_plain_top10pct"][str(int(y))]=metrics(g[g["plain_252_count"]<=p90])
        out["yearly_exclude_filtered_top10pct"][str(int(y))]=metrics(g[g["filtered_252_count"]<=f90])

    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()

# trigger run 2026-09-22
