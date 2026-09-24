"""100万円の実資金制約を入れたA/B本番候補の資金配分比較。"""
import json, math, os
import pandas as pd
import config, download
from strategies import registry
from run_backtest_ranking import (
    _build_strategy_trades, _merge_day_candidates, _release_is_before_entry,
    A_CUT_PCT, B_CUT_PCT
)

INITIAL_CAPITAL=1_000_000
MAX_POSITIONS=8
LOT=100
MODES=[
 ("fixed100","100株固定"),
 ("target190k","1銘柄19万円目安"),
 ("target195k","1銘柄19.5万円目安"),
 ("target150k","1銘柄15万円目安"),
 ("target175k","1銘柄17.5万円目安"),
 ("target200k","1銘柄20万円目安"),
 ("target205k","1銘柄20.5万円目安"),
 ("target210k","1銘柄21万円目安"),
 ("target225k","1銘柄22.5万円目安"),
 ("target250k","1銘柄25万円目安"),
 ("equal_slots","現金÷残り枠で均等"),
]

def qty_for(mode, price, cash, free_slots):
    lot_cost=price*LOT
    if lot_cost>cash: return 0
    if mode=="fixed100": return LOT
    if mode=="target190k": target=190_000
    elif mode=="target195k": target=195_000
    elif mode=="target150k": target=150_000
    elif mode=="target175k": target=175_000
    elif mode=="target200k": target=200_000
    elif mode=="target205k": target=205_000
    elif mode=="target210k": target=210_000
    elif mode=="target225k": target=225_000
    elif mode=="target250k": target=250_000
    elif mode=="equal_slots": target=cash/max(1,free_slots)
    else: raise ValueError(mode)
    lots=max(1, math.floor(target/lot_cost))
    return int(min(lots, math.floor(cash/lot_cost))*LOT)

def simulate(a,b,mode):
    by_a={}; by_b={}; dates=set()
    for tr in a:
        d=pd.Timestamp(tr["entry_date"]); by_a.setdefault(d,[]).append(tr); dates.add(d)
    for tr in b:
        d=pd.Timestamp(tr["entry_date"]); by_b.setdefault(d,[]).append(tr); dates.add(d)
    cash=float(INITIAL_CAPITAL); positions=[]; done=[]
    skipped_cash=skipped_slots=skipped_dup=0; max_pos=0
    invested_samples=[]

    for d in sorted(dates):
        remain=[]
        for p in positions:
            if _release_is_before_entry(p, d):
                proceeds=p["qty"]*float(p["exit_price"])
                cash += proceeds
                p["pnl_yen"]=round(proceeds-p["cost"],2)
                done.append(p)
            else: remain.append(p)
        positions=remain

        candidates,_,_,_= _merge_day_candidates(by_a.get(d,[]),by_b.get(d,[]),"balanced_rank")
        for tr in candidates:
            if len(positions)>=MAX_POSITIONS:
                skipped_slots+=1; continue
            if any(str(p["code"])==str(tr["code"]) for p in positions):
                skipped_dup+=1; continue
            price=float(tr["entry_price"])
            free_slots=MAX_POSITIONS-len(positions)
            qty=qty_for(mode,price,cash,free_slots)
            if qty<LOT:
                skipped_cash+=1; continue
            cost=qty*price
            cash-=cost
            p=dict(tr); p.update({"qty":qty,"cost":round(cost,2),"allocation_mode":mode})
            positions.append(p); invested_samples.append(cost); max_pos=max(max_pos,len(positions))

    for p in sorted(positions,key=lambda x:pd.Timestamp(x["exit_date"])):
        proceeds=p["qty"]*float(p["exit_price"]); cash+=proceeds
        p["pnl_yen"]=round(proceeds-p["cost"],2); done.append(p)

    pnl=sum(float(x["pnl_yen"]) for x in done)
    wins=[x for x in done if x["pnl_yen"]>0]; losses=[x for x in done if x["pnl_yen"]<0]
    gp=sum(x["pnl_yen"] for x in wins); gl=-sum(x["pnl_yen"] for x in losses)
    return done,{
      "label":dict(MODES)[mode],"initial_capital":INITIAL_CAPITAL,
      "ending_capital":round(cash,0),"profit_yen":round(pnl,0),
      "return_pct":round((cash/INITIAL_CAPITAL-1)*100,2),
      "entries":len(done),"win_rate_pct":round(len(wins)/len(done)*100,2) if done else 0,
      "profit_factor_yen":round(gp/gl,3) if gl else None,
      "avg_investment_yen":round(sum(invested_samples)/len(invested_samples),0) if invested_samples else 0,
      "max_positions":max_pos,"skipped_cash":skipped_cash,"skipped_slots":skipped_slots,
      "skipped_active_duplicate":skipped_dup
    }

def load_or_build_trades():
    """候補/出口計算は重いので一度CSV化し、配分比較では再利用する。"""
    os.makedirs("output",exist_ok=True)
    a_cache="output/real_allocation_candidates_A.csv"
    b_cache="output/real_allocation_candidates_B.csv"
    if os.path.exists(a_cache) and os.path.exists(b_cache):
        a=pd.read_csv(a_cache).to_dict("records")
        b=pd.read_csv(b_cache).to_dict("records")
        print(f"候補キャッシュ使用: A={len(a)} / B={len(b)}")
        return a,b
    strategy=registry.get_strategy("ma5_breakout")
    target_codes=download.get_target_codes()
    cache=f"backtest_prices_{config.TARGET_MARKET}_{config.BACKTEST_YEARS}y.csv"
    price_df=download.get_price_history_incremental(cache_filename=cache,years=config.BACKTEST_YEARS)
    signals,by_code=strategy(price_df,target_codes)
    a=_build_strategy_trades(signals,by_code,"A"); b=_build_strategy_trades(signals,by_code,"B")
    pd.DataFrame(a).to_csv(a_cache,index=False)
    pd.DataFrame(b).to_csv(b_cache,index=False)
    print(f"候補キャッシュ作成: A={len(a)} / B={len(b)}")
    return a,b

def main():
    a,b=load_or_build_trades()
    out={"settings":{"capital":INITIAL_CAPITAL,"lot":LOT,"max_positions":MAX_POSITIONS,
      "ranking":"balanced A/B production rank","A_cut_pct":A_CUT_PCT,"B_cut_pct":B_CUT_PCT},
      "results":{}}
    lines=["100万円・100株単位・最大8銘柄・本番A/B順位 資金配分比較",""]
    for mode,label in MODES:
        trades,s=simulate(a,b,mode); out["results"][mode]=s
        pd.DataFrame(trades).to_csv(f"output/real_allocation_{mode}.csv",index=False,encoding="utf-8-sig")
        lines.append(f"{label}: 最終 {s['ending_capital']:,.0f}円 / 損益 {s['profit_yen']:+,.0f}円 / "
                     f"収益率 {s['return_pct']:+.2f}% / {s['entries']}件 / 勝率 {s['win_rate_pct']:.2f}% / "
                     f"PF {s['profit_factor_yen']} / 平均投入 {s['avg_investment_yen']:,.0f}円 / "
                     f"資金不足見送り {s['skipped_cash']} / 8枠満杯見送り {s['skipped_slots']}")
    # 20万円 vs 20.5万円の差分を直接分解
    t200=pd.DataFrame(simulate(a,b,"target200k")[0])
    t205=pd.DataFrame(simulate(a,b,"target205k")[0])
    key=["entry_date","code"]
    a200=t200.set_index(key)
    a205=t205.set_index(key)
    k200=set(a200.index); k205=set(a205.index)
    only200=k200-k205; only205=k205-k200; common=k200&k205
    pnl_only200=sum(float(a200.loc[k,"pnl_yen"]) for k in only200)
    pnl_only205=sum(float(a205.loc[k,"pnl_yen"]) for k in only205)
    qty_delta_pnl=0.0
    qty_changed=0
    for k in common:
        r200=a200.loc[k]; r205=a205.loc[k]
        if hasattr(r200,"iloc") and getattr(r200,"ndim",1)>1: r200=r200.iloc[0]
        if hasattr(r205,"iloc") and getattr(r205,"ndim",1)>1: r205=r205.iloc[0]
        qdiff=float(r200["qty"])-float(r205["qty"])
        if qdiff:
            qty_changed+=1
            qty_delta_pnl += qdiff*(float(r200["exit_price"])-float(r200["entry_price"]))
    lines += [
      "",
      "20万円 vs 20.5万円 差分分解",
      f"20万円だけで買えた取引: {len(only200)}件 / 合計損益 {pnl_only200:+,.0f}円",
      f"20.5万円だけで買えた取引: {len(only205)}件 / 合計損益 {pnl_only205:+,.0f}円",
      f"共通取引で株数が違う: {qty_changed}件 / 20万円側の株数差による損益差 {qty_delta_pnl:+,.0f}円",
      f"差分取引の純寄与: {(pnl_only200-pnl_only205):+,.0f}円",
    ]
    pd.DataFrame([dict(a200.loc[k])|{"entry_date":k[0],"code":k[1]} for k in only200]).to_csv("output/only_target200k.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([dict(a205.loc[k])|{"entry_date":k[0],"code":k[1]} for k in only205]).to_csv("output/only_target205k.csv",index=False,encoding="utf-8-sig")
    with open("output/real_allocation_comparison.json","w",encoding="utf-8") as f: json.dump(out,f,ensure_ascii=False,indent=2)
    with open("output/real_allocation_comparison.txt","w",encoding="utf-8") as f: f.write("\n".join(lines))
    print("\n".join(lines))

if __name__=="__main__": main()
