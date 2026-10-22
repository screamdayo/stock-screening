"""Compare current production Kuitto signals by market using local J-Quants snapshots."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strategies.kuitto_pullback_auto import _prepare, _features

def run(root):
    master = pd.read_parquet(root / "master.parquet")
    df = pd.concat([pd.read_parquet(root / "recent_archive.parquet"),
                    pd.read_parquet(root / "daily.parquet")], ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Code"] = df["Code"].astype(str)
    df = df.drop_duplicates(["Code","Date"], keep="last").sort_values(["Code","Date"])
    for raw, adj in [("O","AdjO"),("H","AdjH"),("L","AdjL"),("C","AdjC"),("Vo","AdjVo")]:
        if adj in df:
            df[raw] = df[adj].where(df[adj].notna(), df[raw])
    master["Code"] = master.Code.astype(str)
    meta = master.set_index("Code").to_dict("index")
    latest = df.Date.max()
    records = []
    counts = {}
    for code, group in df[df.Code.isin(meta)].groupby("Code"):
        m = meta[code]
        g = _prepare(group)
        market = m["MktNm"]
        c = counts.setdefault(market, {"master_codes":0,"latest_codes":0,"latest_liquid_codes":0})
        c["master_codes"] += 1
        if g.iloc[-1].Date == latest:
            c["latest_codes"] += 1
            if g.iloc[-1].AVG_TURNOVER_20 >= 500_000_000:
                c["latest_liquid_codes"] += 1
        for idx in range(25,len(g)):
            if g.iloc[idx].Date < pd.Timestamp("2026-09-16"):
                continue
            f = _features(g, idx, apply_liquidity=False)
            if not f:
                continue
            liquid = f["avg_turnover_20"] is not None and f["avg_turnover_20"] >= 500_000_000
            # Cross-check actual production filter, not a duplicated set of entry conditions.
            assert bool(_features(g,idx,apply_liquidity=True)) == liquid
            records.append({"date":str(g.iloc[idx].Date.date()),"code":code,
                            "name":m["CoName"],"market":market,"close":float(g.iloc[idx].C),
                            "passes_liquidity":bool(liquid),**f})
    days = sorted(str(x.date()) for x in df.Date.unique() if x >= pd.Timestamp("2026-09-16"))
    daily = []
    for day in days:
        row={"date":day}
        for market in counts:
            rows=[r for r in records if r["date"]==day and r["market"]==market]
            row[market]={"before_liquidity":len(rows),"after_liquidity":sum(r["passes_liquidity"] for r in rows)}
        daily.append(row)
    result={"latest_date":str(latest.date()),"period_start":"2026-09-16",
            "trading_days":len(days),"master_date":str(pd.to_datetime(master.Date).max().date()),
            "classification_note":"Market classification uses the saved master snapshot; historical membership is not reconstructed.",
            "strategy":"unchanged strategies/kuitto_pullback_auto.py; volume>=1.25x; rolling20 mean adjusted close*adjusted volume >=500m JPY",
            "coverage":counts,"daily_counts":daily,
            "latest_candidates_before_liquidity":[r for r in records if r["date"]==str(latest.date())],
            "period_signals":records}
    out=Path(__file__).resolve().parents[1]/"results"/"kuitto_market_comparison_20261022.json"
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="period_signals"},ensure_ascii=False,indent=2))
    print("period totals:",{m:sum(r["passes_liquidity"] for r in records if r["market"]==m) for m in counts})

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data-root",type=Path,required=True)
    run(p.parse_args().data_root)
