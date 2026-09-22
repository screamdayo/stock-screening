import json, statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd

TR=Path("results/reversal_signal_baseline_trades.csv")
OUT=Path("results/pinch_top3_vs_top5_capital_efficiency.json")
SENSOR_DATES=pd.to_datetime(["2018-12-27","2020-04-06","2022-03-10","2024-08-06","2025-04-08"])
KS=[1,2,3,4,5]

def met(x):
    s=pd.to_numeric(pd.Series(x),errors="coerce").dropna()
    if s.empty:return {"n":0}
    neg=s[s<0].sum()
    return {"n":int(len(s)),"win":round(float((s>0).mean()*100),2),"avg":round(float(s.mean()),3),"med":round(float(s.median()),3),"pf":None if neg==0 else round(float(s[s>0].sum()/abs(neg)),3)}

def prepare():
    t=pd.read_csv(TR,dtype={"code":str})
    t=t[t.type=="sharp_drop_reversal"].copy()
    t.signal_date=pd.to_datetime(t.signal_date)
    t=t[t.signal_date.isin(SENSOR_DATES)]
    t=t[(t.ret5_before_pct<=-12)&(t.dd20_pct<=-20)&(t.candle_pct>=2)&(t.vol_ratio20>=1.5)].copy()
    return t

DATA=prepare()

def topk_event(g,k):
    return g.sort_values(["dd20_pct","code"]).head(k).copy()

def eval_k(k):
    event_rows=[]; allp=[]
    for d,g in DATA.groupby("signal_date"):
        p=topk_event(g,k); allp.append(p)
        rec={"date":str(d.date()),"available":int(len(g)),"picked":int(len(p))}
        for h in [5,10,15]:
            vals=pd.to_numeric(p[f"ret_{h}d_pct"],errors="coerce").dropna()
            # fixed total capital split equally across names => portfolio return is mean constituent return
            rec[f"portfolio_{h}d_pct"]=round(float(vals.mean()),3) if len(vals) else None
        event_rows.append(rec)
    p=pd.concat(allp,ignore_index=True)
    ret15=[r["portfolio_15d_pct"] for r in event_rows if r["portfolio_15d_pct"] is not None]
    return {
      "top_k":k,
      "selected_trade_stats":{"r5":met(p.ret_5d_pct),"r10":met(p.ret_10d_pct),"r15":met(p.ret_15d_pct)},
      "equal_total_capital_event_portfolios":event_rows,
      "portfolio15":{"events":len(ret15),"avg":round(float(pd.Series(ret15).mean()),3),"median":round(float(pd.Series(ret15).median()),3),"worst":round(float(min(ret15)),3),"best":round(float(max(ret15)),3),"stdev":round(float(pd.Series(ret15).std(ddof=0)),3)}
    }

def marginal_4_5():
    rows=[]
    for d,g in DATA.groupby("signal_date"):
        rg=g.sort_values(["dd20_pct","code"]).reset_index(drop=True)
        p=rg.iloc[3:5].copy()
        if p.empty: continue
        rows.append({"date":str(d.date()),"n":len(p),"avg5":round(float(p.ret_5d_pct.mean()),3),"avg10":round(float(p.ret_10d_pct.mean()),3),"avg15":round(float(p.ret_15d_pct.mean()),3),"win15":round(float((p.ret_15d_pct>0).mean()*100),2)})
    return rows

def compare_3_5():
    rows=[]
    for d,g in DATA.groupby("signal_date"):
        p3=topk_event(g,3); p5=topk_event(g,5)
        r3=float(p3.ret_15d_pct.mean()); r5=float(p5.ret_15d_pct.mean())
        rows.append({"date":str(d.date()),"top3_portfolio15":round(r3,3),"top5_portfolio15":round(r5,3),"top3_minus_top5_pp":round(r3-r5,3),"available":len(g)})
    return rows

def main():
    with ThreadPoolExecutor(max_workers=len(KS)) as ex:
        curve=list(ex.map(eval_k,KS))
    cmp=compare_3_5()
    out={
      "study":"Pinch sensor DD20 ranking capital efficiency",
      "assumption":"Same total capital per event, split equally among selected names. Fees/slippage/lot-size constraints excluded.",
      "topk_curve":curve,
      "top3_vs_top5":cmp,
      "marginal_rank4_5":marginal_4_5(),
      "note":"Use event-level portfolio returns for capital efficiency; selected-trade win rates are secondary because names within an event are correlated."
    }
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=="__main__":main()

# trigger
