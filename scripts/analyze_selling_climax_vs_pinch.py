import os, json, requests
from pathlib import Path
import pandas as pd

TR=Path("results/reversal_signal_baseline_trades.csv")
OUT=Path("results/selling_climax_vs_pinch_relationship.json")
PINCH_DATES=pd.to_datetime(["2018-12-27","2020-04-06","2022-03-10","2024-08-06","2025-04-08"])
LOOKBACK_BD=30

def get_topix():
    h={"x-api-key":os.environ["JQUANTS_API_KEY"]}
    r=requests.get(
        "https://api.jquants.com/v2/indices/bars/daily/topix",
        params={"from":"2018-01-01","to":"2026-09-30"},
        headers=h,timeout=60
    )
    r.raise_for_status()
    rows=r.json().get("data",[])
    d=pd.DataFrame(rows)
    d["Date"]=pd.to_datetime(d["Date"])
    for c in ["O","H","L","C"]:
        d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d.dropna(subset=["C"]).sort_values("Date").reset_index(drop=True)
    d["topix_ret1"]=(d.C/d.C.shift(1)-1)*100
    d["topix_ret2"]=(d.C/d.C.shift(2)-1)*100
    d["topix_ret5"]=(d.C/d.C.shift(5)-1)*100
    d["prev_low"]=d.L.shift(1)
    d["low_up"]=d.L>d.prev_low
    return d

def load_reversal():
    t=pd.read_csv(TR,dtype={"code":str})
    t=t[t.type=="sharp_drop_reversal"].copy()
    t["signal_date"]=pd.to_datetime(t.signal_date)
    daily=t.groupby("signal_date").agg(
        crowd=("code","size"),
        med_candle=("candle_pct","median"),
        med_vol=("vol_ratio20","median"),
        med_dd20=("dd20_pct","median"),
    ).reset_index().rename(columns={"signal_date":"Date"})
    a=t[(t.ret5_before_pct<=-12)&(t.dd20_pct<=-20)&(t.candle_pct>=2)&(t.vol_ratio20>=1.5)].copy()
    anchor=a.groupby("signal_date").agg(
        anchor_n=("code","size"),
        anchor_win15=("ret_15d_pct",lambda s:float((pd.to_numeric(s,errors="coerce")>0).mean()*100)),
        anchor_avg15=("ret_15d_pct",lambda s:float(pd.to_numeric(s,errors="coerce").mean())),
    ).reset_index().rename(columns={"signal_date":"Date"})
    return daily.merge(anchor,on="Date",how="left").fillna({"anchor_n":0})

def flags(r):
    # Deliberately broad exploratory families; not production rules.
    return {
      "loose": bool(r.crowd>=20 and r.anchor_n>=1 and r.topix_ret1<=-2.0),
      "medium": bool(r.crowd>=20 and r.anchor_n>=5 and r.topix_ret1<=-3.0),
      "strict": bool(r.crowd>=50 and r.anchor_n>=10 and r.topix_ret1<=-4.0),
    }

def rowdict(r,pinch):
    f=flags(r)
    return {
      "date":str(r.Date.date()),
      "bd_before_pinch":int((r._cal_idx - pinch._cal_idx)),
      "topix_ret1":round(float(r.topix_ret1),3),
      "topix_ret5":round(float(r.topix_ret5),3) if pd.notna(r.topix_ret5) else None,
      "topix_low_up":bool(r.low_up),
      "crowd":int(r.crowd),
      "anchor_n":int(r.anchor_n),
      "anchor_win15":None if pd.isna(r.anchor_win15) else round(float(r.anchor_win15),2),
      "anchor_avg15":None if pd.isna(r.anchor_avg15) else round(float(r.anchor_avg15),3),
      "med_candle":round(float(r.med_candle),3),
      "med_vol":round(float(r.med_vol),3),
      "flags":f,
    }

def main():
    ix=get_topix()
    rev=load_reversal()
    x=ix.merge(rev,on="Date",how="left")
    for c in ["crowd","anchor_n"]:
        x[c]=x[c].fillna(0)
    x["_cal_idx"]=range(len(x))

    events=[]
    for pdte in PINCH_DATES:
        z=x[x.Date==pdte]
        if z.empty: continue
        pinch=z.iloc[0]
        lo=max(0,int(pinch._cal_idx)-LOOKBACK_BD)
        w=x.iloc[lo:int(pinch._cal_idx)].copy()
        w=w[w.crowd>0].copy()

        flagged={k:[] for k in ["loose","medium","strict"]}
        for _,r in w.iterrows():
            rd=rowdict(r,pinch)
            for k,v in rd["flags"].items():
                if v: flagged[k].append(rd)

        # Also surface the strongest negative-TOPIX reversal days, regardless of provisional flags.
        candidates=w[(w.topix_ret1<0)&(w.anchor_n>0)].copy()
        if not candidates.empty:
            candidates["score"]=(-candidates.topix_ret1.clip(upper=0))*candidates.anchor_n
            strongest=candidates.sort_values(["score","Date"],ascending=[False,True]).head(8)
            strongest_rows=[rowdict(r,pinch) for _,r in strongest.iterrows()]
        else:
            strongest_rows=[]

        events.append({
          "pinch_date":str(pdte.date()),
          "lookback_business_days":LOOKBACK_BD,
          "provisional_flag_counts":{k:len(v) for k,v in flagged.items()},
          "nearest_prior_flag":{k:(v[-1] if v else None) for k,v in flagged.items()},
          "all_flagged_days":flagged,
          "strongest_negative_topix_reversal_days":strongest_rows,
        })

    # False-positive context: count provisional flagged days across entire sample and whether a pinch follows within 30bd.
    pinch_idx={int(x[x.Date==d]._cal_idx.iloc[0]):str(d.date()) for d in PINCH_DATES if not x[x.Date==d].empty}
    context={}
    revdays=x[x.crowd>0].copy()
    for rule in ["loose","medium","strict"]:
        rows=[]
        for _,r in revdays.iterrows():
            if not flags(r)[rule]: continue
            i=int(r._cal_idx)
            future=[(pi,pd) for pi,pd in pinch_idx.items() if 0 < pi-i <= LOOKBACK_BD]
            nearest=min(future,key=lambda q:q[0]-i) if future else None
            rows.append({
              "date":str(r.Date.date()),
              "topix_ret1":round(float(r.topix_ret1),3),
              "crowd":int(r.crowd),"anchor_n":int(r.anchor_n),
              "pinch_within_30bd":nearest is not None,
              "next_pinch_date":nearest[1] if nearest else None,
              "bd_to_pinch":(nearest[0]-i) if nearest else None,
            })
        context[rule]={
          "total_flag_days":len(rows),
          "flag_days_followed_by_pinch_within_30bd":sum(r["pinch_within_30bd"] for r in rows),
          "rows":rows,
        }

    out={
      "study":"Exploratory relationship: provisional selling-climax days vs frozen Pinch sensor dates",
      "pinch_dates":[str(d.date()) for d in PINCH_DATES],
      "provisional_rules":{
        "loose":"TOPIX <= -2%, coarse reversal count >=20, anchor >=1",
        "medium":"TOPIX <= -3%, coarse reversal count >=20, anchor >=5",
        "strict":"TOPIX <= -4%, coarse reversal count >=50, anchor >=10",
      },
      "note":"These are exploratory flags only, not optimized or production rules. Goal is first to see whether a selloff/reversal day tends to precede Pinch.",
      "events":events,
      "false_positive_context":context,
    }
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    compact={
      "events":[{"pinch":e["pinch_date"],"counts":e["provisional_flag_counts"],
                 "nearest":{k:(v["date"] if v else None) for k,v in e["nearest_prior_flag"].items()}}
                for e in events],
      "context":{k:{kk:v[kk] for kk in ["total_flag_days","flag_days_followed_by_pinch_within_30bd"]} for k,v in context.items()}
    }
    print(json.dumps(compact,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()

# trigger
