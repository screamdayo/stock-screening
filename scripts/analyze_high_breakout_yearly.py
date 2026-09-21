import json
from pathlib import Path
import pandas as pd

SRC=Path("results/high_breakout_60_120_252d_trades.csv")
OUT=Path("results/high_breakout_yearly_15d.json")

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

def main():
    t=pd.read_csv(SRC,dtype={"code":str})
    t["signal_date_dt"]=pd.to_datetime(t["signal_date"])
    t["year"]=t["signal_date_dt"].dt.year

    out={
      "focus":"15-trading-day exit",
      "variants":{}
    }

    for lb in [60,120,252]:
        base=t[t["lookback"]==lb].copy()
        filt=base[base["ma25_rising"].astype(bool) & base["volume_1_5x"].astype(bool)].copy()

        out["variants"][str(lb)]={
          "plain":{str(int(y)):metrics(g) for y,g in base.groupby("year")},
          "ma25_rising_plus_volume":{str(int(y)):metrics(g) for y,g in filt.groupby("year")},
        }

    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
