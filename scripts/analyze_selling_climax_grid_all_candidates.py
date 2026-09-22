import os, json, requests
from pathlib import Path
import pandas as pd
import numpy as np

TR=Path("results/reversal_signal_baseline_trades.csv")
OUT=Path("results/selling_climax_grid_all_candidates.json")

DD20_THRESH=[-10,-12.5,-15,-17.5,-20,-22.5,-25]
ANCHOR_SHARE_THRESH=[5,7.5,10,12.5,15,20,25]
RET5_THRESH=[-7.5,-10,-12.5,-15,-17.5]
CROWD_THRESH=[20,50,100,150,200,300]
BASE_TOPIX_RET1=-2.0

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
    d["ret1"]=(d.C/d.C.shift(1)-1)*100
    d["ret5"]=(d.C/d.C.shift(5)-1)*100
    d["dd20"]=(d.C/d.C.rolling(20).max()-1)*100
    for n in [5,10,15,20]:
        d[f"topix_fwd{n}"]=(d.C.shift(-n)/d.C-1)*100
    return d

def load_reversal():
    t=pd.read_csv(TR,dtype={"code":str})
    t=t[t.type=="sharp_drop_reversal"].copy()
    t["signal_date"]=pd.to_datetime(t.signal_date)
    for c in ["ret5_before_pct","dd20_pct","candle_pct","vol_ratio20","ret_15d_pct"]:
        t[c]=pd.to_numeric(t[c],errors="coerce")

    daily=t.groupby("signal_date").agg(
        crowd=("code","size"),
        coarse_med_ret5=("ret5_before_pct","median"),
        coarse_med_dd20=("dd20_pct","median"),
    ).reset_index().rename(columns={"signal_date":"Date"})

    anchor=t[
        (t.ret5_before_pct<=-12)&
        (t.dd20_pct<=-20)&
        (t.candle_pct>=2)&
        (t.vol_ratio20>=1.5)
    ].copy()
    a=anchor.groupby("signal_date").agg(
        anchor_n=("code","size"),
        anchor_avg15=("ret_15d_pct","mean"),
        anchor_win15=("ret_15d_pct",lambda s:(s>0).mean()*100),
        anchor_med15=("ret_15d_pct","median"),
    ).reset_index().rename(columns={"signal_date":"Date"})

    x=daily.merge(a,on="Date",how="left")
    x["anchor_n"]=x.anchor_n.fillna(0)
    x["anchor_share_pct"]=np.where(x.crowd>0,x.anchor_n/x.crowd*100,np.nan)
    return x

def metrics(df):
    if df.empty:
        return {
            "days":0,"anchor_days":0,"topix_15d_n":0,
            "topix_15d_avg_pct":None,"topix_15d_median_pct":None,"topix_15d_positive_pct":None,
            "anchor_day_avg15_pct":None,"anchor_day_median15_pct":None,
            "anchor_day_positive_pct":None,"anchor_win15_mean_pct":None,
        }
    top=pd.to_numeric(df.topix_fwd15,errors="coerce").dropna()
    aa=pd.to_numeric(df.anchor_avg15,errors="coerce").dropna()
    aw=pd.to_numeric(df.anchor_win15,errors="coerce").dropna()
    def rr(v):
        return None if pd.isna(v) else round(float(v),3)
    return {
        "days":int(len(df)),
        "anchor_days":int(len(aa)),
        "topix_15d_n":int(len(top)),
        "topix_15d_avg_pct":rr(top.mean()) if len(top) else None,
        "topix_15d_median_pct":rr(top.median()) if len(top) else None,
        "topix_15d_positive_pct":rr((top>0).mean()*100) if len(top) else None,
        "anchor_day_avg15_pct":rr(aa.mean()) if len(aa) else None,
        "anchor_day_median15_pct":rr(aa.median()) if len(aa) else None,
        "anchor_day_positive_pct":rr((aa>0).mean()*100) if len(aa) else None,
        "anchor_win15_mean_pct":rr(aw.mean()) if len(aw) else None,
    }

def add_examples(df, n=20):
    z=df.sort_values("Date")
    out=[]
    for _,r in z.tail(n).iterrows():
        out.append({
            "date":str(r.Date.date()),
            "ret1":round(float(r.ret1),3),
            "ret5":round(float(r.ret5),3),
            "dd20":round(float(r.dd20),3),
            "crowd":int(r.crowd),
            "anchor_n":int(r.anchor_n),
            "anchor_share_pct":round(float(r.anchor_share_pct),3) if pd.notna(r.anchor_share_pct) else None,
            "topix_fwd15":round(float(r.topix_fwd15),3) if pd.notna(r.topix_fwd15) else None,
            "anchor_avg15":round(float(r.anchor_avg15),3) if pd.notna(r.anchor_avg15) else None,
        })
    return out

def main():
    ix=get_topix()
    rv=load_reversal()
    x=ix.merge(rv,on="Date",how="left")
    x["crowd"]=x.crowd.fillna(0)
    x["anchor_n"]=x.anchor_n.fillna(0)
    x["anchor_share_pct"]=x.anchor_share_pct.fillna(0)

    base=x[(x.ret1<=BASE_TOPIX_RET1)&(x.crowd>=1)&(x.anchor_n>=1)].copy()

    grid_dd_share=[]
    for dd in DD20_THRESH:
        for sh in ANCHOR_SHARE_THRESH:
            q=base[(base.dd20<=dd)&(base.anchor_share_pct>=sh)]
            grid_dd_share.append({
                "dd20_lte":dd,"anchor_share_gte_pct":sh,**metrics(q)
            })

    grid_ret5_crowd=[]
    for r5 in RET5_THRESH:
        for cr in CROWD_THRESH:
            q=base[(base.ret5<=r5)&(base.crowd>=cr)]
            grid_ret5_crowd.append({
                "ret5_lte":r5,"crowd_gte":cr,**metrics(q)
            })

    # Robust shortlist: require at least 3 distinct days, then sort by day-level anchor outcome
    # with TOPIX follow-through as a secondary diagnostic. No threshold is auto-promoted to production.
    short1=sorted(
        [r for r in grid_dd_share if r["days"]>=3 and r["anchor_day_avg15_pct"] is not None],
        key=lambda r:(-r["anchor_day_avg15_pct"],-(r["topix_15d_avg_pct"] or -999),-r["days"])
    )[:20]
    short2=sorted(
        [r for r in grid_ret5_crowd if r["days"]>=3 and r["anchor_day_avg15_pct"] is not None],
        key=lambda r:(-r["anchor_day_avg15_pct"],-(r["topix_15d_avg_pct"] or -999),-r["days"])
    )[:20]

    # Explicitly inspect the intuitive round-number candidates discussed in chat.
    named_rules={}
    named=[
      ("dd15_share10", (base.dd20<=-15)&(base.anchor_share_pct>=10)),
      ("dd15_share12_5", (base.dd20<=-15)&(base.anchor_share_pct>=12.5)),
      ("dd17_5_share10", (base.dd20<=-17.5)&(base.anchor_share_pct>=10)),
      ("ret5_10_crowd150", (base.ret5<=-10)&(base.crowd>=150)),
      ("ret5_12_5_crowd100", (base.ret5<=-12.5)&(base.crowd>=100)),
      ("ret5_12_5_crowd150", (base.ret5<=-12.5)&(base.crowd>=150)),
    ]
    for name,mask in named:
        q=base[mask].copy()
        named_rules[name]={**metrics(q),"examples":add_examples(q,30)}

    out={
      "study":"All-candidate selling-climax grid: depth x breadth",
      "base_universe_rule":"TOPIX daily return <= -2%, crowd >=1, anchor_n >=1",
      "anchor_definition":"ret5<=-12%, dd20<=-20%, candle>=+2%, volume ratio20>=1.5",
      "base_candidate_days":int(len(base)),
      "axes":{
        "dd20_x_anchor_share":{"dd20_thresholds":DD20_THRESH,"anchor_share_thresholds_pct":ANCHOR_SHARE_THRESH},
        "ret5_x_crowd":{"ret5_thresholds":RET5_THRESH,"crowd_thresholds":CROWD_THRESH},
      },
      "grid_dd20_anchor_share":grid_dd_share,
      "grid_ret5_crowd":grid_ret5_crowd,
      "top_by_anchor_day_avg15_dd20_share":short1,
      "top_by_anchor_day_avg15_ret5_crowd":short2,
      "named_rules":named_rules,
      "warning":"Exploratory. Multiple thresholds are scanned on the same historical sample, so top cells are subject to selection bias. Prefer broad plateaus with multiple independent dates over a single best cell."
    }
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({
      "base_candidate_days":len(base),
      "top_dd20_share":short1[:10],
      "top_ret5_crowd":short2[:10],
      "named_rules":{k:{kk:vv for kk,vv in v.items() if kk!="examples"} for k,v in named_rules.items()}
    },ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()

# trigger
