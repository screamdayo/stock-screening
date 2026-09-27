import csv, math, datetime as dt
from collections import defaultdict
from pathlib import Path
SRC=Path("data/kuitto_score_trades.csv")
OUT=Path("output/kuitto_capital_1m_8slots.txt")
INITIAL=1_000_000.0; LOT=100; MAX_POS=8
CAP={0:None,1:0.0,2:0.25,3:0.0,4:1.0}
tr=[]
with SRC.open(encoding="utf-8-sig",newline="") as f:
  for r in csv.DictReader(f):
    try:
      s=int(float(r["runner_score"])); gap=float(r["next_open_gap_pct"]); entry=float(r["entry_open"]); exitp=float(r["exit_price"])
      ma=float(r["ma5_vs_ma25_pct"]); pnl=float(r["exit_pnl_pct"])
      ed=dt.datetime.strptime(r["entry_date"],"%Y-%m-%d").date(); xd=dt.datetime.strptime(r["exit_date"],"%Y-%m-%d").date()
    except: continue
    if any(math.isnan(x) for x in (gap,entry,exitp,ma,pnl)): continue
    cap=CAP[s]
    if cap is not None and gap>cap: continue
    tr.append({"code":str(r["code"]),"score":s,"gap":gap,"entry":entry,"exit":exitp,"ma":ma,"ed":ed,"xd":xd,"pnl":pnl})

by_entry=defaultdict(list)
for x in tr: by_entry[x["ed"]].append(x)
dates=sorted(set([x["ed"] for x in tr]+[x["xd"] for x in tr]))
cash=INITIAL; positions=[]; history=[]; buys=0; skips_cash=0; skips_slots=0; skips_overlap=0; max_pos=0; max_candidates_day=0; used_cap=[]
peak=INITIAL; max_dd=0.0; dd_date=None

for d in dates:
  # exit at open first; proceeds can fund same-day entries
  still=[]
  for p in positions:
    if p["xd"]==d:
      proceeds=p["exit"]*LOT
      cash += proceeds
    else: still.append(p)
  positions=still
  # entry ranking: score desc, then MA5/25 distance to zero asc, code
  cand=sorted(by_entry.get(d,[]), key=lambda x:(-x["score"],abs(x["ma"]),x["code"]))
  max_candidates_day=max(max_candidates_day,len(cand))
  for x in cand:
    if any(p["code"]==x["code"] for p in positions):
      skips_overlap+=1; continue
    if len(positions)>=MAX_POS:
      skips_slots+=1; continue
    cost=x["entry"]*LOT
    if cost>cash+1e-9:
      skips_cash+=1; continue
    cash-=cost; positions.append(x); buys+=1
  max_pos=max(max_pos,len(positions))
  invested=sum(p["entry"]*LOT for p in positions)
  equity_cost=cash+invested
  used_cap.append(invested)
  # realized/cost-basis equity drawdown; not daily MTM
  if equity_cost>peak: peak=equity_cost
  dd=(equity_cost/peak-1)*100 if peak else 0
  if dd<max_dd: max_dd=dd; dd_date=d
  history.append((d,cash,invested,equity_cost,len(positions)))

# liquidate any impossible leftovers at cached exits (should be none after final date)
final=cash+sum(p["exit"]*LOT for p in positions)
ret=(final/INITIAL-1)*100
avg_used=sum(used_cap)/len(used_cap) if used_cap else 0
qualified=len(tr)
lines=[
  "くいっと 実運用資金制約BT",
  "前提: 初期100万円 / 100株 / 最大8銘柄 / 現行スコア別gap / score降順→同点MA25近い順 / 同日売却を先に現金化 / 手数料税金なし",
  "",
  f"条件通過トレード: {qualified}",
  f"実際に買えた: {buys}",
  f"資金不足見送り: {skips_cash}",
  f"8枠満杯見送り: {skips_slots}",
  f"同一銘柄保有中見送り: {skips_overlap}",
  f"最大同時保有: {max_pos}",
  f"1日の最大候補数: {max_candidates_day}",
  f"平均投下資金: {avg_used:,.0f}円 ({avg_used/INITIAL*100:.1f}%)",
  f"最終資産: {final:,.0f}円",
  f"累積リターン: {ret:+.2f}%",
  f"実現損益ベース最大DD(保有中は取得原価評価): {max_dd:.2f}% / {dd_date}",
  "",
  "注: 真の時価評価DDではなく、固定CSVだけで高速計算するため保有中を取得原価で評価。最終資金・買付可否・枠/資金見送りは実売買順で計算。"
]
OUT.parent.mkdir(exist_ok=True)
OUT.write_text("\n".join(lines),encoding="utf-8")
print("\n".join(lines))