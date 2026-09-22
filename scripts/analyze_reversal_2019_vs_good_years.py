import json
from pathlib import Path
import pandas as pd
TR=Path("results/reversal_signal_baseline_trades.csv")
OUT=Path("results/reversal_2019_vs_good_years_diagnostic.json")
def met(d):
 s=pd.to_numeric(d.ret_15d_pct,errors="coerce").dropna()
 if not len(s): return {"n":0}
 pos=s[s>0]; neg=s[s<0]; gl=-neg.sum()
 return {"n":len(s),"win":round((s>0).mean()*100,2),"avg":round(s.mean(),3),"med":round(s.median(),3),"pf":round(pos.sum()/gl,3) if gl>0 else None}
def stats(d,cols):
 o={}
 for c in cols:
  s=pd.to_numeric(d[c],errors="coerce").dropna()
  o[c]={"mean":round(s.mean(),3),"median":round(s.median(),3)} if len(s) else {}
 return o
def main():
 t=pd.read_csv(TR,dtype={"code":str}); t=t[t.type=="sharp_drop_reversal"].copy()
 t.signal_date=pd.to_datetime(t.signal_date); t["year"]=t.signal_date.dt.year
 cnt=t.groupby("signal_date").size().rename("crowd"); t=t.join(cnt,on="signal_date")
 p=t[(t.ret5_before_pct<=-12)&(t.dd20_pct<=-20)&(t.candle_pct>=2)&(t.vol_ratio20>=1.5)&(t.crowd>=100)].copy()
 cols=["ret5_before_pct","ret20_before_pct","dd20_pct","atr14_pct","candle_pct","vol_ratio20","crowd"]
 out={"rule":"anchor + crowd>=100","years":{},"event_dates":{}}
 for y in [2018,2019,2020,2024,2025]:
  d=p[p.year==y]
  out["years"][str(y)]={"performance":met(d),"signal_features":stats(d,cols)}
  dates=d.groupby("signal_date").agg(n=("code","size"),avg15=("ret_15d_pct","mean"),win15=("ret_15d_pct",lambda x:(x>0).mean()*100),
    crowd=("crowd","first"),med_ret5=("ret5_before_pct","median"),med_ret20=("ret20_before_pct","median"),
    med_dd20=("dd20_pct","median"),med_atr=("atr14_pct","median"),med_candle=("candle_pct","median"),med_vol=("vol_ratio20","median")).reset_index()
  out["event_dates"][str(y)]=dates.to_dict("records")
 OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
 print(json.dumps(out,ensure_ascii=False,indent=2,default=str))
if __name__=="__main__":main()

# trigger
