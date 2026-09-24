"""最大8枠・均等配分で、満枠時の入れ替え3方式を比較する。"""
import json, os
from pathlib import Path
import pandas as pd
import config, download
from compare_real_allocation import load_or_build_trades, qty_for, INITIAL_CAPITAL, MAX_POSITIONS, LOT
from run_backtest_ranking import _merge_day_candidates, _release_is_before_entry

VARIANTS = {
    "baseline": {"label":"入れ替えなし","min_profit":None,"max_rank":None},
    "profit_any": {"label":"含み益>0なら入れ替え","min_profit":0.0,"max_rank":None},
    "profit_3": {"label":"含み益+3%以上","min_profit":3.0,"max_rank":None},
    "profit_5": {"label":"含み益+5%以上","min_profit":5.0,"max_rank":None},
    "profit_7": {"label":"含み益+7%以上","min_profit":7.0,"max_rank":None},
    "profit_10": {"label":"含み益+10%以上","min_profit":10.0,"max_rank":None},
    "profit_5_rank1": {"label":"含み益+5%以上×新候補1位だけ","min_profit":5.0,"max_rank":1},
    "profit_5_rank2": {"label":"含み益+5%以上×新候補2位以内","min_profit":5.0,"max_rank":2},
}

def build_market_maps():
    cache=f"backtest_prices_{config.TARGET_MARKET}_{config.BACKTEST_YEARS}y.csv"
    df=download.get_price_history_incremental(cache_filename=cache, years=config.BACKTEST_YEARS).copy()
    df["Code"]=df["Code"].astype(str)
    df["Date"]=pd.to_datetime(df["Date"])
    if "O" not in df.columns and "Open" in df.columns: df["O"]=df["Open"]
    if "C" not in df.columns and "Close" in df.columns: df["C"]=df["Close"]
    open_map={}
    momentum_map={}
    for code,g in df.groupby("Code"):
        g=g.sort_values("Date").copy()
        g["MA5"]=g["C"].rolling(5).mean()
        g["MA5_SLOPE"]=g["MA5"].pct_change()*100
        for i,row in enumerate(g.itertuples()):
            d=pd.Timestamp(row.Date)
            open_map[(str(code),d)]=float(row.O)
            # entry open時点で確定している前営業日のMA5傾きだけを使う
            if i>0:
                prev=g.iloc[i-1]
                slope=prev["MA5_SLOPE"]
                momentum_map[(str(code),d)] = float(slope) if pd.notna(slope) else 999.0
    return open_map, momentum_map

def close_position(p, px, d, reason):
    proceeds=p["qty"]*px
    q=dict(p)
    q["exit_date_actual"]=str(pd.Timestamp(d).date())
    q["exit_price_actual"]=px
    q["exit_reason_actual"]=reason
    q["pnl_yen"]=round(proceeds-p["cost"],2)
    q["pnl_pct_actual"]=round((px/float(p["entry_price"])-1)*100,4)
    return proceeds,q

def pick_victim(positions, d, min_profit, open_map):
    rows=[]
    for p in positions:
        code=str(p["code"])
        px=open_map.get((code,d))
        if px is None: continue
        upct=(px/float(p["entry_price"])-1)*100
        if upct >= min_profit:
            rows.append((p,px,upct))
    return max(rows,key=lambda r:r[2]) if rows else None

def simulate(a,b,variant,open_map,momentum_map):
    by_a={}; by_b={}; dates=set()
    for tr in a:
        d=pd.Timestamp(tr["entry_date"]); by_a.setdefault(d,[]).append(tr); dates.add(d)
    for tr in b:
        d=pd.Timestamp(tr["entry_date"]); by_b.setdefault(d,[]).append(tr); dates.add(d)
    cash=float(INITIAL_CAPITAL); positions=[]; done=[]; rotations=[]
    skipped_slots=skipped_cash=skipped_dup=0; entries=0

    for d in sorted(dates):
        remain=[]
        for p in positions:
            if _release_is_before_entry(p,d):
                proceeds,q=close_position(p,float(p["exit_price"]),d,p.get("exit_reason","normal"))
                cash+=proceeds; done.append(q)
            else:
                remain.append(p)
        positions=remain
        candidates,_,_,_=_merge_day_candidates(by_a.get(d,[]),by_b.get(d,[]),"balanced_rank")
        rotated_today=False
        for rank_idx,tr in enumerate(candidates, start=1):
            if any(str(p["code"])==str(tr["code"]) for p in positions):
                skipped_dup+=1; continue

            if len(positions)>=MAX_POSITIONS:
                rule=VARIANTS[variant]
                if variant=="baseline" or rotated_today or (rule["max_rank"] is not None and rank_idx>rule["max_rank"]):
                    skipped_slots+=1; continue
                victim=pick_victim(positions,d,rule["min_profit"],open_map)
                if victim is None:
                    skipped_slots+=1; continue
                old,old_px,old_upct=victim
                old_mom=momentum_map.get((str(old["code"]),d),999.0)
                proceeds,q=close_position(old,old_px,d,f"rotation_{variant}")
                positions.remove(old); cash+=proceeds; done.append(q)
                rotations.append({
                    "date":str(d.date()),"mode":variant,
                    "sold_code":str(old["code"]),"sold_unrealized_pct":old_upct,
                    "sold_prev_ma5_slope_pct":old_mom,
                    "replacement_code":str(tr["code"]),
                })
                rotated_today=True

            price=float(tr["entry_price"])
            free_slots=MAX_POSITIONS-len(positions)
            qty=qty_for("equal_slots",price,cash,free_slots)
            if qty<LOT:
                skipped_cash+=1; continue
            cost=qty*price; cash-=cost
            p=dict(tr); p.update({"qty":qty,"cost":round(cost,2),"allocation_mode":"equal_slots"})
            positions.append(p); entries+=1

    for p in sorted(positions,key=lambda x:pd.Timestamp(x["exit_date"])):
        proceeds,q=close_position(p,float(p["exit_price"]),pd.Timestamp(p["exit_date"]),p.get("exit_reason","normal"))
        cash+=proceeds; done.append(q)

    gains=sum(float(x["pnl_yen"]) for x in done if float(x["pnl_yen"])>0)
    losses=-sum(float(x["pnl_yen"]) for x in done if float(x["pnl_yen"])<0)
    return {
        "label":VARIANTS[variant]["label"],"ending_capital":round(cash,0),
        "profit_yen":round(cash-INITIAL_CAPITAL,0),
        "return_pct":round((cash/INITIAL_CAPITAL-1)*100,2),
        "entries":entries,"closed":len(done),"rotations":len(rotations),
        "win_rate_pct":round(sum(1 for x in done if float(x["pnl_yen"])>0)/len(done)*100,2) if done else 0,
        "profit_factor_yen":round(gains/losses,3) if losses else None,
        "skipped_slots":skipped_slots,"skipped_cash":skipped_cash,"skipped_dup":skipped_dup,
    }, rotations, done

def main():
    os.makedirs("output",exist_ok=True)
    a,b=load_or_build_trades()
    open_map,momentum_map=build_market_maps()
    report={"settings":{"capital":INITIAL_CAPITAL,"max_positions":MAX_POSITIONS,"sizing":"equal_slots","rotation_limit":"1 per day","momentum":"previous-day MA5 slope"},"results":{}}
    lines=["満枠時 入れ替え条件比較（100万円・最大8枠・均等配分）",""]
    all_rots=[]
    for variant in VARIANTS:
        s,rots,done=simulate(a,b,variant,open_map,momentum_map)
        report["results"][variant]=s
        all_rots.extend(rots)
        pd.DataFrame(done).to_csv(f"output/full_slot_{variant}_trades.csv",index=False,encoding="utf-8-sig")
        lines.append(f"{s['label']}: 最終 {s['ending_capital']:,.0f}円 / 損益 {s['profit_yen']:+,.0f}円 / 収益率 {s['return_pct']:+.2f}% / PF {s['profit_factor_yen']} / 入れ替え {s['rotations']}回 / 満枠見送り {s['skipped_slots']}")
    pd.DataFrame(all_rots).to_csv("output/full_slot_rotations.csv",index=False,encoding="utf-8-sig")
    Path("output/full_slot_rotation_comparison.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    text="\n".join(lines)
    Path("output/full_slot_rotation_comparison.txt").write_text(text,encoding="utf-8")
    print(text)

if __name__=="__main__":
    main()
