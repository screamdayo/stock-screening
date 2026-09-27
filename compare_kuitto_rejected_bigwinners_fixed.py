import csv, math
from pathlib import Path
SRC=Path("data/kuitto_score_trades.csv")
OUT=Path("output/kuitto_rejected_bigwinners_fixed.txt")
POLICY={0:None,1:0.0,2:0.25,3:0.0,4:1.0}
rows=[]
with SRC.open(encoding="utf-8-sig",newline="") as f:
  for r in csv.DictReader(f):
    try:
      s=int(float(r["runner_score"])); g=float(r["next_open_gap_pct"]); p=float(r["exit_pnl_pct"])
      r5=float(r["return_5d_pct"]); r10=float(r["return_10d_pct"]); r15=float(r["return_15d_pct"])
    except: continue
    if any(math.isnan(x) for x in (g,p,r5,r10,r15)): continue
    th=POLICY.get(s); keep=True if th is None else g<=th
    if not keep:
      rows.append({"code":r["code"],"date":r["signal_date"],"score":s,"gap":g,"pnl":p,"r5":r5,"r10":r10,"r15":r15,"best":max(r5,r10,r15)})
lines=[f"除外群 大幅上昇チェック（5/10/15日リターン基準） / 件数 {len(rows)}",""]
for th in [5,10,15,20,30]:
  xs=[r for r in rows if r["best"]>=th]
  lines.append(f"5/10/15日のどこかで +{th}%以上: {len(xs)}件 ({len(xs)/len(rows)*100:.2f}%)")
lines.append("")
lines.append("上位20件")
for r in sorted(rows,key=lambda x:x["best"],reverse=True)[:20]:
  lines.append(f"{r['code']} {r['date']} score{r['score']} gap {r['gap']:+.3f}% / 5d {r['r5']:+.2f}% / 10d {r['r10']:+.2f}% / 15d {r['r15']:+.2f}% / 最終 {r['pnl']:+.2f}%")
OUT.parent.mkdir(exist_ok=True)
OUT.write_text("\n".join(lines),encoding="utf-8")
print("\n".join(lines))