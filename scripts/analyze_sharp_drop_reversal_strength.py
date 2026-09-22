import json
from pathlib import Path
import numpy as np
import pandas as pd

SRC=Path("results/reversal_signal_baseline_trades.csv")
OUT=Path("results/sharp_drop_reversal_strength_scan.json")

def met(d):
    s=pd.to_numeric(d["ret_15d_pct"],errors="coerce").dropna()
    if s.empty:return {"n":0}
    pos=s[s>0]; neg=s[s<0]; gl=-neg.sum()
    return {"n":int(len(s)),"win_rate_pct":round((s>0).mean()*100,2),
            "avg_return_pct":round(s.mean(),3),"median_return_pct":round(s.median(),3),
            "profit_factor":round(pos.sum()/gl,3) if gl>0 else None}

def main():
    t=pd.read_csv(SRC,dtype={"code":str})
    t=t[t["type"]=="sharp_drop_reversal"].copy()
    t["signal_date"]=pd.to_datetime(t["signal_date"])
    split=pd.Timestamp("2021-09-20")
    t["older"]=t.signal_date<split
    scans={}

    # One-factor scans first: identify broad plateaus, not magic cutoffs.
    defs={
      "drop5_max":[("ret5_before_pct","<=",x) for x in [-8,-9,-10,-12,-15]],
      "dd20_max":[("dd20_pct","<=",x) for x in [-10,-12,-15,-18,-20]],
      "reversal_candle_min":[("candle_pct",">=",x) for x in [1,1.5,2,2.5,3,4]],
      "volume_ratio_min":[("vol_ratio20",">=",x) for x in [0.8,1.0,1.2,1.5,2.0]],
      "atr14_min":[("atr14_pct",">=",x) for x in [2,3,4,5,6]],
    }
    for name,items in defs.items():
        scans[name]=[]
        for col,op,x in items:
            mask=t[col]<=x if op=="<=" else t[col]>=x
            p=t[mask]
            scans[name].append({"threshold":x,"overall":met(p),"older5":met(p[p.older]),"recent5":met(p[~p.older])})

    # Coarse combinations, intentionally rounded.
    combos=[]
    for drop in [-8,-10,-12]:
      for dd in [-10,-15,-20]:
       for candle in [1,2,3]:
        for vol in [1.0,1.5,2.0]:
          p=t[(t.ret5_before_pct<=drop)&(t.dd20_pct<=dd)&(t.candle_pct>=candle)&(t.vol_ratio20>=vol)]
          if len(p)<100: continue
          combos.append({"drop5_max":drop,"dd20_max":dd,"candle_min":candle,"vol_ratio_min":vol,
                         "overall":met(p),"older5":met(p[p.older]),"recent5":met(p[~p.older])})
    # Surface robust candidates: require both halves positive PF and enough observations.
    robust=[x for x in combos if x["older5"]["n"]>=100 and x["recent5"]["n"]>=100
            and (x["older5"]["profit_factor"] or 0)>1.15 and (x["recent5"]["profit_factor"] or 0)>1.15]
    robust=sorted(robust,key=lambda x:(min(x["older5"]["profit_factor"],x["recent5"]["profit_factor"]),
                                       x["overall"]["avg_return_pct"]),reverse=True)
    out={"baseline":met(t),"one_factor_scans":scans,"robust_combinations_top30":robust[:30],
         "combo_count_tested":len(combos),"robust_combo_count":len(robust),
         "note":"Exploratory strength scan. Prefer broad neighboring robustness over the top row. Same survivor-bias/cost caveats as baseline."}
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=="__main__":main()

# trigger
