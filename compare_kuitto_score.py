"""くいっと押し目版本番ルールの伸びスコア別成績。"""
import json, os
from pathlib import Path
import pandas as pd
import config, download
from strategies import kuitto_pullback_auto as strat
from forward_test import _prepare, _update_future

def summarize(rows):
    done=[r for r in rows if r.get("gap_pass") is True and r.get("exit_pnl_pct") is not None]
    wins=[r for r in done if float(r["exit_pnl_pct"])>0]
    losses=[r for r in done if float(r["exit_pnl_pct"])<0]
    gp=sum(float(r["exit_pnl_pct"]) for r in wins)
    gl=-sum(float(r["exit_pnl_pct"]) for r in losses)
    return {
        "signals":len(rows),
        "gap_pass":sum(1 for r in rows if r.get("gap_pass") is True),
        "gap_skip":sum(1 for r in rows if r.get("gap_pass") is False),
        "completed":len(done),
        "win_rate_pct":round(len(wins)/len(done)*100,2) if done else None,
        "avg_return_pct":round(sum(float(r["exit_pnl_pct"]) for r in done)/len(done),3) if done else None,
        "median_return_pct":round(float(pd.Series([float(r["exit_pnl_pct"]) for r in done]).median()),3) if done else None,
        "profit_factor":round(gp/gl,3) if gl else None,
    }

def _as_bool(v):
    if isinstance(v, bool):
        return v
    if pd.isna(v):
        return None
    return str(v).strip().lower() in ("true", "1", "yes")

def load_or_build_rows():
    trade_cache=Path("output/kuitto_score_trades.csv")
    if trade_cache.exists() and trade_cache.stat().st_size > 0:
        print(f"完成済みトレードキャッシュを使用: {trade_cache}")
        df=pd.read_csv(trade_cache)
        rows=df.to_dict("records")
        for r in rows:
            if "gap_pass" in r:
                r["gap_pass"]=_as_bool(r["gap_pass"])
        return rows

    print("完成済みトレードキャッシュなし。5年分を初回計算します。")
    targets=download.get_target_codes()
    cache=f"backtest_prices_{config.TARGET_MARKET}_{config.BACKTEST_YEARS}y.csv"
    price_df=download.get_price_history_incremental(cache_filename=cache,years=config.BACKTEST_YEARS)
    signals,groups=strat.find_signals(price_df,targets)
    rows=[]
    for s in signals:
        g=_prepare(groups[s["code"]])
        r=dict(s)
        _update_future(r,g,int(s["signal_idx"]))
        rows.append(r)
    pd.DataFrame(rows).to_csv(trade_cache,index=False,encoding="utf-8-sig")
    return rows

def main():
    os.makedirs("output",exist_ok=True)
    rows=load_or_build_rows()

    report={"years":config.BACKTEST_YEARS,"rule":"kuitto_pullback_auto + next open gap <= +0.5% + -5% stop / strict gakutto / max15","scores":{}}
    lines=[f"くいっと押し目版 伸びスコア別成績（{config.BACKTEST_YEARS}年）","翌朝 +0.5%以内のみ買い / 現行出口",""]
    for score in range(5):
        rr=[r for r in rows if int(r.get("runner_score",0))==score]
        if not rr: continue
        s=summarize(rr); report["scores"][str(score)]=s
        lines.append(f"{score}/4: シグナル {s['signals']} / 買えた {s['gap_pass']} / 見送り {s['gap_skip']} / 完了 {s['completed']} / 勝率 {s['win_rate_pct']}% / 平均 {s['avg_return_pct']:+.3f}% / 中央値 {s['median_return_pct']:+.3f}% / PF {s['profit_factor']}")
    two=[r for r in rows if int(r.get("runner_score",0))==2]
    atr2=[]
    combo2=[]
    other2=[]
    for r in two:
        atr=r.get("atr14_pct")
        dd=r.get("dd20_pct")
        turn=r.get("avg_turnover_20")
        atr_hit=pd.notna(atr) and float(atr)>=strat.RUNNER_ATR14_MIN
        dd_hit=pd.notna(dd) and float(dd)<=strat.RUNNER_DD20_MAX
        turn_hit=pd.notna(turn) and float(turn)>=strat.RUNNER_TURNOVER20_MIN
        if atr_hit and not dd_hit and not turn_hit:
            atr2.append(r)
        elif (not atr_hit) and dd_hit and turn_hit:
            combo2.append(r)
        else:
            other2.append(r)

    report["score2_breakdown"]={}
    lines += ["","2/4 内訳"]
    for name, rr in [("ATRのみ2点",atr2),("DD20+売買代金",combo2),("その他",other2)]:
        s=summarize(rr)
        report["score2_breakdown"][name]=s
        lines.append(f"{name}: シグナル {s['signals']} / 買えた {s['gap_pass']} / 完了 {s['completed']} / 勝率 {s['win_rate_pct']}% / 平均 {s['avg_return_pct']:+.3f}% / 中央値 {s['median_return_pct']:+.3f}% / PF {s['profit_factor']}")

    Path("output/kuitto_score_comparison.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    text="\n".join(lines)
    Path("output/kuitto_score_comparison.txt").write_text(text,encoding="utf-8")
    pd.DataFrame(rows).to_csv("output/kuitto_score_trades.csv",index=False,encoding="utf-8-sig")
    print(text)

if __name__=="__main__":
    main()
