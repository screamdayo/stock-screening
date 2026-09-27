import csv, math, statistics
from pathlib import Path
SRC=Path("data/kuitto_score_trades.csv")
OUT=Path("output/kuitto_rejected_winner_features.txt")
POLICY={0:None,1:0.0,2:0.25,3:0.0,4:1.0}
FIELDS=["runner_score","next_open_gap_pct","atr14_pct","dd20_pct","avg_turnover_20","bull_candle_pct","volume_ratio","ma5_prior5d_decline_pct","ma5_vs_ma25_pct","close_vs_ma5_pct"]
rows=[]
with SRC.open(encoding="utf-8-sig",newline="") as f:
  for r in csv.DictReader(f):
    try:
      s=int(float(r["runner_score"])); g=float(r["next_open_gap_pct"]); p=float(r["exit_pnl_pct"]); r5=float(r["return_5d_pct"]); r10=float(r["return_10d_pct"]); r15=float(r["return_15d_pct"])
    except: continue
    if any(math.isnan(x) for x in (g,p,r5,r10,r15)): continue
    th=POLICY.get(s); keep=True if th is None else g<=th
    if keep: continue
    rec={"code":r["code"],"date":r["signal_date"],"success":max(r5,r10,r15)>=10,"best":max(r5,r10,r15)}
    ok=True
    for fld in FIELDS:
      try: rec[fld]=float(r[fld])
      except: ok=False; break
    if ok: rows.append(rec)

good=[r for r in rows if r["success"]]
bad=[r for r in rows if not r["success"]]
def med(xs): return statistics.median(xs) if xs else None
def mean(xs): return sum(xs)/len(xs) if xs else None
def fmt(v,n=3): return "-" if v is None else f"{v:.{n}f}"
lines=[f"除外群 +10%以上上昇66件 共通項比較","成功群={len(good)}件 / その他={len(bad)}件",""]
labels={
"runner_score":"スコア","next_open_gap_pct":"翌朝ギャップ%","atr14_pct":"ATR14%","dd20_pct":"DD20%","avg_turnover_20":"20日平均売買代金","bull_candle_pct":"当日上昇率%","volume_ratio":"出来高比","ma5_prior5d_decline_pct":"MA5直前5日%","ma5_vs_ma25_pct":"MA5/25乖離%","close_vs_ma5_pct":"終値/MA5乖離%"
}
for fld in FIELDS:
  ga=[r[fld] for r in good]; ba=[r[fld] for r in bad]
  lines.append(f"{labels[fld]}: 成功 中央 {fmt(med(ga))} / 平均 {fmt(mean(ga))} | その他 中央 {fmt(med(ba))} / 平均 {fmt(mean(ba))}")
lines.append("")
lines.append("スコア構成")
for s in range(5):
  g=sum(1 for r in good if int(r["runner_score"])==s); b=sum(1 for r in bad if int(r["runner_score"])==s)
  lines.append(f"{s}/4: 成功 {g}/{len(good)} ({g/len(good)*100:.1f}%) | その他 {b}/{len(bad)} ({b/len(bad)*100:.1f}%)")
lines.append("")
cuts=[("gap<=0.5",lambda r:r["next_open_gap_pct"]<=0.5),("gap<=1.0",lambda r:r["next_open_gap_pct"]<=1.0),("ATR>=2.9",lambda r:r["atr14_pct"]>=2.9),("DD20<=-5.86",lambda r:r["dd20_pct"]<=-5.855856),("turnover>=8.28e8",lambda r:r["avg_turnover_20"]>=828434090),("vol>=1.5",lambda r:r["volume_ratio"]>=1.5),("bull>=2.0",lambda r:r["bull_candle_pct"]>=2.0)]
lines.append("単純条件の成功群/その他 通過率")
for name,fn in cuts:
  gp=sum(fn(r) for r in good); bp=sum(fn(r) for r in bad)
  lines.append(f"{name}: 成功 {gp/len(good)*100:.1f}% | その他 {bp/len(bad)*100:.1f}%")
lines.append("")
lines.append("成功群 上位15件")
for r in sorted(good,key=lambda x:x["best"],reverse=True)[:15]:
  lines.append(f"{r['code']} {r['date']} best {r['best']:+.2f}% / score{int(r['runner_score'])} gap {r['next_open_gap_pct']:+.2f}% ATR {r['atr14_pct']:.2f}% DD20 {r['dd20_pct']:.2f}% turnover {r['avg_turnover_20']/1e8:.1f}億 vol {r['volume_ratio']:.2f}")
OUT.parent.mkdir(exist_ok=True)
OUT.write_text("\n".join(lines),encoding="utf-8")
print("\n".join(lines))