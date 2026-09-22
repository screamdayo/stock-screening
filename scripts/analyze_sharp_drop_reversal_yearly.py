import json
from pathlib import Path
import pandas as pd

SRC=Path("results/reversal_signal_baseline_trades.csv")
OUT=Path("results/sharp_drop_reversal_yearly.json")

def met(d):
    s=pd.to_numeric(d["ret_15d_pct"],errors="coerce").dropna()
    if s.empty:return {"n":0}
    pos=s[s>0]; neg=s[s<0]; gl=-neg.sum()
    return {"n":int(len(s)),"win_rate_pct":round((s>0).mean()*100,2),
            "avg_return_pct":round(s.mean(),3),"median_return_pct":round(s.median(),3),
            "profit_factor":round(pos.sum()/gl,3) if gl>0 else None,
            "loss_5pct_rate":round((s<=-5).mean()*100,2),
            "gain_10pct_rate":round((s>=10).mean()*100,2)}

def main():
    t=pd.read_csv(SRC,dtype={"code":str})
    t=t[t["type"]=="sharp_drop_reversal"].copy()
    t["signal_date"]=pd.to_datetime(t["signal_date"])
    t["year"]=t.signal_date.dt.year
    rules={
      "anchor_-12_dd20_c2_v1.5":(t.ret5_before_pct<=-12)&(t.dd20_pct<=-20)&(t.candle_pct>=2)&(t.vol_ratio20>=1.5),
      "looser_-10_dd20_c1_v1.5":(t.ret5_before_pct<=-10)&(t.dd20_pct<=-20)&(t.candle_pct>=1)&(t.vol_ratio20>=1.5),
      "deep_-15_dd20_c2_v1.5":(t.ret5_before_pct<=-15)&(t.dd20_pct<=-20)&(t.candle_pct>=2)&(t.vol_ratio20>=1.5),
    }
    out={"holding_days":15,"rules":{},"note":"Exploratory yearly stability check; current-Prime survivor bias and no costs/slippage/overlap constraints remain."}
    for name,m in rules.items():
        p=t[m].copy()
        out["rules"][name]={"overall":met(p),"yearly":{str(int(y)):met(g) for y,g in p.groupby("year")}}
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
