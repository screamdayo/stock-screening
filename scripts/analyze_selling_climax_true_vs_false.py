import os, json, requests, math
from pathlib import Path
import pandas as pd
import numpy as np

TR=Path("results/reversal_signal_baseline_trades.csv")
OUT=Path("results/selling_climax_true_vs_false_diagnostic.json")

# Good sell-climax examples already observed, plus notable false/early examples.
GOOD_DATES=pd.to_datetime(["2018-12-25","2020-03-13","2025-04-07"])
FALSE_DATES=pd.to_datetime(["2018-02-06","2020-02-28"])
FOCUS_DATES=sorted(set(GOOD_DATES.tolist()+FALSE_DATES.tolist()))

def get_topix():
    h={"x-api-key":os.environ["JQUANTS_API_KEY"]}
    r=requests.get(
        "https://api.jquants.com/v2/indices/bars/daily/topix",
        params={"from":"2017-12-01","to":"2026-09-30"},
        headers=h,timeout=60
    )
    r.raise_for_status()
    d=pd.DataFrame(r.json().get("data",[]))
    d["Date"]=pd.to_datetime(d["Date"])
    for c in ["O","H","L","C"]:
        d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d.dropna(subset=["O","H","L","C"]).sort_values("Date").reset_index(drop=True)
    for n in [1,2,3,5,10]:
        d[f"ret{n}"]=(d.C/d.C.shift(n)-1)*100
    d["dd20"]=(d.C/d.C.rolling(20).max()-1)*100
    d["dd60"]=(d.C/d.C.rolling(60).max()-1)*100
    d["range_pct"]=(d.H/d.L-1)*100
    d["body_pct"]=(d.C/d.O-1)*100
    d["gap_pct"]=(d.O/d.C.shift(1)-1)*100
    d["clv"]=(d.C-d.L)/(d.H-d.L)
    d["prev_low"]=d.L.shift(1)
    d["low_vs_prev_low_pct"]=(d.L/d.prev_low-1)*100
    d["low_up"]=d.L>d.prev_low
    d["from20low_pct"]=(d.C/d.L.rolling(20).min()-1)*100
    d["down3_count_5"]=pd.Series((d.ret1<=-3).astype(int)).rolling(5).sum()
    d["down2_count_10"]=pd.Series((d.ret1<=-2).astype(int)).rolling(10).sum()
    d["min_ret1_prev5"]=d.ret1.shift(1).rolling(5).min()
    d["sum_ret1_prev5"]=d.ret1.shift(1).rolling(5).sum()
    return d

def load_reversal():
    t=pd.read_csv(TR,dtype={"code":str})
    t=t[t.type=="sharp_drop_reversal"].copy()
    t["signal_date"]=pd.to_datetime(t.signal_date)
    daily=t.groupby("signal_date").agg(
        crowd=("code","size"),
        coarse_med_ret5=("ret5_before_pct","median"),
        coarse_med_ret20=("ret20_before_pct","median"),
        coarse_med_dd20=("dd20_pct","median"),
        coarse_med_candle=("candle_pct","median"),
        coarse_med_vol=("vol_ratio20","median"),
    ).reset_index().rename(columns={"signal_date":"Date"})
    anchor=t[(t.ret5_before_pct<=-12)&(t.dd20_pct<=-20)&(t.candle_pct>=2)&(t.vol_ratio20>=1.5)].copy()
    a=anchor.groupby("signal_date").agg(
        anchor_n=("code","size"),
        anchor_med_ret5=("ret5_before_pct","median"),
        anchor_med_dd20=("dd20_pct","median"),
        anchor_med_candle=("candle_pct","median"),
        anchor_med_vol=("vol_ratio20","median"),
        anchor_win15=("ret_15d_pct",lambda s:(pd.to_numeric(s,errors="coerce")>0).mean()*100),
        anchor_avg15=("ret_15d_pct",lambda s:pd.to_numeric(s,errors="coerce").mean()),
    ).reset_index().rename(columns={"signal_date":"Date"})
    x=daily.merge(a,on="Date",how="left").fillna({"anchor_n":0})
    x["anchor_share_pct"]=np.where(x.crowd>0,x.anchor_n/x.crowd*100,np.nan)
    x=x.sort_values("Date").reset_index(drop=True)
    x["crowd_prev_obs"]=x.crowd.shift(1)
    x["crowd_vs_prev_obs"]=x.crowd/x.crowd_prev_obs
    x["crowd_prev5max"]=x.crowd.shift(1).rolling(5,min_periods=1).max()
    x["crowd_vs_prev5max"]=x.crowd/x.crowd_prev5max
    return x

def clean(v):
    if pd.isna(v): return None
    if isinstance(v,(np.bool_,bool)): return bool(v)
    if isinstance(v,(np.integer,int)): return int(v)
    if isinstance(v,(np.floating,float)): return round(float(v),3)
    return v

def main():
    ix=get_topix()
    rv=load_reversal()
    x=ix.merge(rv,on="Date",how="left")
    for c in ["crowd","anchor_n"]:
        x[c]=x[c].fillna(0)
    x["stress20"]=(x.crowd>=20).astype(int)
    x["stress50"]=(x.crowd>=50).astype(int)
    x["anchor5"]=(x.anchor_n>=5).astype(int)
    x["stress20_prev5"]=x.stress20.shift(1).rolling(5,min_periods=1).sum()
    x["stress20_prev10"]=x.stress20.shift(1).rolling(10,min_periods=1).sum()
    x["stress50_prev10"]=x.stress50.shift(1).rolling(10,min_periods=1).sum()
    x["anchor5_prev10"]=x.anchor5.shift(1).rolling(10,min_periods=1).sum()

    cols=[
      "ret1","ret2","ret3","ret5","ret10","dd20","dd60","range_pct","body_pct","gap_pct","clv",
      "low_vs_prev_low_pct","low_up","from20low_pct","down3_count_5","down2_count_10",
      "min_ret1_prev5","sum_ret1_prev5","crowd","anchor_n","anchor_share_pct",
      "coarse_med_ret5","coarse_med_ret20","coarse_med_dd20","coarse_med_candle","coarse_med_vol",
      "anchor_med_ret5","anchor_med_dd20","anchor_med_candle","anchor_med_vol",
      "crowd_vs_prev_obs","crowd_vs_prev5max","stress20_prev5","stress20_prev10","stress50_prev10","anchor5_prev10",
      "anchor_win15","anchor_avg15"
    ]

    rows=[]
    for d in FOCUS_DATES:
        z=x[x.Date==d]
        if z.empty: continue
        r=z.iloc[0]
        label="good" if d in GOOD_DATES.tolist() else "false_early"
        rows.append({"date":str(d.date()),"label":label,**{c:clean(r.get(c)) for c in cols}})

    df=pd.DataFrame(rows)

    # Crude one-feature separator scan: only report thresholds that perfectly separate these labeled examples.
    numeric=[c for c in cols if c not in ["low_up"]]
    separators=[]
    good=df[df.label=="good"]; bad=df[df.label=="false_early"]
    for c in numeric:
        gv=pd.to_numeric(good[c],errors="coerce").dropna()
        bv=pd.to_numeric(bad[c],errors="coerce").dropna()
        if len(gv)!=len(good) or len(bv)!=len(bad): continue
        # good greater than bad
        if gv.min()>bv.max():
            thr=(gv.min()+bv.max())/2
            separators.append({"feature":c,"direction":"good_above","threshold":round(float(thr),3),
                               "good_min":round(float(gv.min()),3),"false_max":round(float(bv.max()),3),
                               "margin":round(float(gv.min()-bv.max()),3)})
        # good lower than bad
        if gv.max()<bv.min():
            thr=(gv.max()+bv.min())/2
            separators.append({"feature":c,"direction":"good_below","threshold":round(float(thr),3),
                               "good_max":round(float(gv.max()),3),"false_min":round(float(bv.min()),3),
                               "margin":round(float(bv.min()-gv.max()),3)})
    separators.sort(key=lambda q:q["margin"],reverse=True)

    # Pairwise simple AND search among top sensible features, avoiding future performance fields.
    candidate_features=[
      "ret5","ret10","dd20","dd60","clv","low_vs_prev_low_pct","anchor_share_pct",
      "crowd","anchor_n","crowd_vs_prev5max","stress20_prev10","stress50_prev10","anchor5_prev10"
    ]
    pair_rules=[]
    # Build thresholds from midpoints between observed values.
    def possible_rules(c):
        vals=sorted(set(pd.to_numeric(df[c],errors="coerce").dropna()))
        th=[(a+b)/2 for a,b in zip(vals,vals[1:])]
        out=[]
        for t in th:
            out.append((f"{c}>={t:.3f}", lambda row,c=c,t=t: float(row[c])>=t))
            out.append((f"{c}<={t:.3f}", lambda row,c=c,t=t: float(row[c])<=t))
        return out
    rules={c:possible_rules(c) for c in candidate_features if c in df.columns and df[c].notna().all()}
    all_rows=df.to_dict("records")
    goods=[r for r in all_rows if r["label"]=="good"]
    bads=[r for r in all_rows if r["label"]=="false_early"]
    keys=list(rules)
    for i,c1 in enumerate(keys):
        for c2 in keys[i+1:]:
            for n1,f1 in rules[c1]:
                for n2,f2 in rules[c2]:
                    if all(f1(r) and f2(r) for r in goods) and all(not (f1(r) and f2(r)) for r in bads):
                        pair_rules.append({"rule":f"{n1} AND {n2}"})
                        if len(pair_rules)>=50: break
                if len(pair_rules)>=50: break
            if len(pair_rules)>=50: break
        if len(pair_rules)>=50: break

    out={
      "study":"True vs false/early selling-climax diagnostic",
      "good_dates":[str(d.date()) for d in GOOD_DATES],
      "false_early_dates":[str(d.date()) for d in FALSE_DATES],
      "rows":rows,
      "perfect_single_feature_separators_on_these_examples":separators,
      "example_two_feature_rules_that_separate_these_examples":pair_rules[:20],
      "warning":"Tiny hand-picked sample. Separators are diagnostic clues only, not production thresholds. Validate on all candidate days before using."
    }
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({
      "rows":rows,
      "top_single_separators":separators[:12],
      "pair_rules":pair_rules[:10]
    },ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
