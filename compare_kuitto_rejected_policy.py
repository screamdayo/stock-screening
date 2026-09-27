"""固定CSVでスコア別ギャップルールにより除外されたトレードだけを分析。"""
import csv, math
from pathlib import Path

SRC=Path("data/kuitto_score_trades.csv")
OUT=Path("output/kuitto_rejected_policy.txt")
POLICY={0:None,1:0.0,2:0.25,3:0.0,4:1.0}

def stats(vals):
    if not vals: return (0,None,None,None,None)
    vals=sorted(vals)
    wins=sum(v>0 for v in vals)
    avg=sum(vals)/len(vals)
    med=vals[len(vals)//2] if len(vals)%2 else (vals[len(vals)//2-1]+vals[len(vals)//2])/2
    gp=sum(v for v in vals if v>0)
    gl=-sum(v for v in vals if v<0)
    pf=gp/gl if gl else None
    return len(vals), wins/len(vals)*100, avg, med, pf

def fmt(v,n=3): return "-" if v is None else f"{v:.{n}f}"
rows=[]
with SRC.open(encoding="utf-8-sig",newline="") as f:
    for r in csv.DictReader(f):
        try:
            s=int(float(r["runner_score"])); g=float(r["next_open_gap_pct"]); p=float(r["exit_pnl_pct"])
        except (TypeError,ValueError,KeyError): continue
        if math.isnan(g) or math.isnan(p): continue
        th=POLICY.get(s)
        keep=True if th is None else g<=th
        if not keep: rows.append((s,g,p))

lines=["スコア別ギャップルールで除外されたトレード分析",""]
allvals=[p for _,_,p in rows]
n,wr,avg,med,pf=stats(allvals)
lines.append(f"除外全体: 件数 {n} / 勝率 {fmt(wr,2)}% / 平均 {fmt(avg)}% / 中央値 {fmt(med)}% / PF {fmt(pf)}")
lines.append("")
for s in range(5):
    vals=[p for ss,_,p in rows if ss==s]
    n,wr,avg,med,pf=stats(vals)
    lines.append(f"{s}/4: 件数 {n} / 勝率 {fmt(wr,2)}% / 平均 {fmt(avg)}% / 中央値 {fmt(med)}% / PF {fmt(pf)}")
lines.append("")
for lo,hi,label in [(0,0.5,"0〜0.5%"),(0.5,1.0,"0.5〜1.0%"),(1.0,2.0,"1.0〜2.0%"),(2.0,999,">2.0%")]:
    vals=[p for _,g,p in rows if g>lo and g<=hi]
    n,wr,avg,med,pf=stats(vals)
    lines.append(f"除外ギャップ帯 {label}: 件数 {n} / 勝率 {fmt(wr,2)}% / 平均 {fmt(avg)}% / 中央値 {fmt(med)}% / PF {fmt(pf)}")

OUT.parent.mkdir(exist_ok=True)
OUT.write_text("\n".join(lines),encoding="utf-8")
print("\n".join(lines))