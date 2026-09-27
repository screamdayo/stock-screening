"""固定CSVで除外675件の大幅上昇・取り逃しを分析。"""
import csv, math
from pathlib import Path

SRC=Path("data/kuitto_score_trades.csv")
OUT=Path("output/kuitto_rejected_bigwinners.txt")
POLICY={0:None,1:0.0,2:0.25,3:0.0,4:1.0}

rows=[]
with SRC.open(encoding="utf-8-sig",newline="") as f:
    for r in csv.DictReader(f):
        try:
            s=int(float(r["runner_score"])); g=float(r["next_open_gap_pct"]); pnl=float(r["exit_pnl_pct"]); mfe=float(r["mfe_pct"])
        except (TypeError,ValueError,KeyError): continue
        if any(math.isnan(x) for x in (g,pnl,mfe)): continue
        th=POLICY.get(s)
        keep=True if th is None else g<=th
        if not keep:
            rows.append({"code":r["code"],"date":r["signal_date"],"score":s,"gap":g,"pnl":pnl,"mfe":mfe,"reason":r.get("exit_reason","")})

lines=[f"除外群 大幅上昇チェック / 件数 {len(rows)}",""]
for th in [5,10,15,20,30]:
    xs=[r for r in rows if r["mfe"]>=th]
    avgp=sum(r["pnl"] for r in xs)/len(xs) if xs else 0
    win=sum(r["pnl"]>0 for r in xs)
    lines.append(f"MFE +{th}%以上: {len(xs)}件 ({len(xs)/len(rows)*100:.2f}%) / 最終勝率 {win/len(xs)*100:.2f}% / 最終平均 {avgp:+.3f}%" if xs else f"MFE +{th}%以上: 0件")

lines.append("")
lines.append("MFE上位20件")
for r in sorted(rows,key=lambda x:x["mfe"],reverse=True)[:20]:
    lines.append(f"{r['code']} {r['date']} score{r['score']} gap {r['gap']:+.3f}% / MFE {r['mfe']:+.3f}% / 最終 {r['pnl']:+.3f}% / {r['reason']}")

missed=[r for r in rows if r["mfe"]>=10 and r["pnl"]<=0]
lines.append("")
lines.append(f"MFE+10%以上なのに最終0%以下: {len(missed)}件")
for r in sorted(missed,key=lambda x:x["mfe"],reverse=True)[:20]:
    lines.append(f"{r['code']} {r['date']} score{r['score']} gap {r['gap']:+.3f}% / MFE {r['mfe']:+.3f}% / 最終 {r['pnl']:+.3f}% / {r['reason']}")
OUT.parent.mkdir(exist_ok=True)
OUT.write_text("\n".join(lines),encoding="utf-8")
print("\n".join(lines))