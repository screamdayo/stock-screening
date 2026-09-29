import json
from pathlib import Path
import pandas as pd

TRADES=Path("results/kuitto_runner_features_trades.csv")
TOPIX=Path("data/topix/topix_daily.parquet")
OUT=Path("results/kuitto_bad_market_sweetspot_duel.json")
SPLIT=pd.Timestamp("2021-11-01")

def metrics(df):
    r=pd.to_numeric(df["ret10"],errors="coerce").dropna()
    if r.empty:return {"n":0}
    gp=r[r>0].sum(); gl=-r[r<0].sum()
    return {
        "n":int(len(r)),
        "win_rate_pct":round(float((r>0).mean()*100),2),
        "avg_ret10_pct":round(float(r.mean()),3),
        "median_ret10_pct":round(float(r.median()),3),
        "p5_plus_pct":round(float((r>=5).mean()*100),2),
        "p10_plus_pct":round(float((r>=10).mean()*100),2),
        "loss3_pct":round(float((r<=-3).mean()*100),2),
        "loss5_pct":round(float((r<=-5).mean()*100),2),
        "profit_factor":round(float(gp/gl),3) if gl>0 else None,
    }

def pack(df):
    return {
        "all":metrics(df),
        "winner":metrics(df[df["winner"]]),
        "older":metrics(df[df["signal_date"]<SPLIT]),
        "recent":metrics(df[df["signal_date"]>=SPLIT]),
        "winner_older":metrics(df[(df["signal_date"]<SPLIT)&df["winner"]]),
        "winner_recent":metrics(df[(df["signal_date"]>=SPLIT)&df["winner"]]),
    }

def main():
    t=pd.read_csv(TRADES,dtype={"code":str})
    t["signal_date"]=pd.to_datetime(t["signal_date"]).dt.normalize()
    for c in ["ret10","atr14_pct","dd20_pct"]:
        t[c]=pd.to_numeric(t[c],errors="coerce")
    t["winner"]=(t["atr14_pct"]>=3.0)&(t["dd20_pct"]<=-5.5)

    x=pd.read_parquet(TOPIX).copy()
    x["date"]=pd.to_datetime(x["Date"]).dt.normalize()
    for c in ["O","C"]:x[c]=pd.to_numeric(x[c],errors="coerce")
    x=x.dropna(subset=["date","C"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    x["ret1_pct"]=x["C"].pct_change()*100
    x["MA5"]=x["C"].rolling(5).mean()
    x["ma5_down"]=x["MA5"]<x["MA5"].shift(1)
    t=t.merge(x[["date","ret1_pct","ma5_down"]],left_on="signal_date",right_on="date",how="left")

    md=t["ma5_down"].eq(True)
    r=t["ret1_pct"]

    zones={
        "ma5_down_any_negative": md&(r<0),
        "ma5_down_0_to_m0_25": md&(r<0)&(r>-0.25),
        "ma5_down_m0_25_to_m0_5": md&(r<=-0.25)&(r>-0.5),
        "ma5_down_m0_5_to_m0_75": md&(r<=-0.5)&(r>-0.75),
        "ma5_down_m0_75_to_m1": md&(r<=-0.75)&(r>-1.0),
        "ma5_down_le_m1": md&(r<=-1.0),
        "ma5_down_gt_m1_lt0": md&(r<0)&(r>-1.0),
        "ma5_down_gt_m0_75_lt0": md&(r<0)&(r>-0.75),
        "ma5_down_gt_m0_5_lt0": md&(r<0)&(r>-0.5),
        "ma5_down_gt_m0_25_lt0": md&(r<0)&(r>-0.25),
    }

    out={
        "baseline":{"all":metrics(t),"winner":metrics(t[t["winner"]])},
        "zones":{}
    }
    for name,mask in zones.items():
        z=t[mask.fillna(False)].copy()
        nz=t[~mask.fillna(False)].copy()
        out["zones"][name]=pack(z)
        out["zones"][name]["vs_rest"]={
            "all_avg_diff_pctpt":round(out["zones"][name]["all"].get("avg_ret10_pct",0)-metrics(nz).get("avg_ret10_pct",0),3) if len(z) else None,
            "winner_avg_diff_pctpt":round(out["zones"][name]["winner"].get("avg_ret10_pct",0)-metrics(nz[nz["winner"]]).get("avg_ret10_pct",0),3) if out["zones"][name]["winner"].get("n",0) else None,
        }

    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
