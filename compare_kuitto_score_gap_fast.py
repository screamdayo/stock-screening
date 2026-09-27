"""固定CSVだけで runner_score × 翌朝ギャップ帯を比較する軽量集計。"""
import csv, math
from pathlib import Path

SRC=Path("data/kuitto_score_trades.csv")
OUT=Path("output/kuitto_score_gap_matrix.txt")
BANDS=[("< =0.0",-999,0.0),("0.0-0.5",0.0,0.5),("0.5-1.0",0.5,1.0),("1.0-2.0",1.0,2.0),(">2.0",2.0,999)]

def stats(vals):
    if not vals: return (0,None,None,None)
    wins=sum(v>0 for v in vals)
    avg=sum(vals)/len(vals)
    gp=sum(v for v in vals if v>0)
    gl=-sum(v for v in vals if v<0)
    pf=(gp/gl) if gl else None
    return len(vals), wins/len(vals)*100, avg, pf

def fmt(v,n=2):
    return "-" if v is None else f"{v:.{n}f}"

rows=[]
with SRC.open(encoding="utf-8-sig",newline="") as f:
    for r in csv.DictReader(f):
        try:
            score=int(float(r["runner_score"]))
            gap=float(r["next_open_gap_pct"])
            pnl=float(r["exit_pnl_pct"])
        except (TypeError,ValueError,KeyError):
            continue
        if math.isnan(gap) or math.isnan(pnl): continue
        rows.append((score,gap,pnl))

lines=["くいっと押し目版 スコア × 翌朝ギャップ帯（固定CSV）",f"対象トレード: {len(rows)}件",""]
for score in range(5):
    lines.append(f"[{score}/4]")
    for label,lo,hi in BANDS:
        if label=="< =0.0": vals=[p for s,g,p in rows if s==score and g<=hi]
        elif label==">2.0": vals=[p for s,g,p in rows if s==score and g>lo]
        else: vals=[p for s,g,p in rows if s==score and g>lo and g<=hi]
        n,wr,avg,pf=stats(vals)
        lines.append(f"{label}: 件数 {n} / 勝率 {fmt(wr)}% / 平均 {fmt(avg,3)}% / PF {fmt(pf,3)}")
    lines.append("")

CUTS=[0.0,0.25,0.5,0.75,1.0,1.5,2.0,None]
for score in range(5):
    lines.append(f"[{score}/4 累積上限]")
    for th in CUTS:
        vals=[p for s,g,p in rows if s==score and (th is None or g<=th)]
        n,wr,avg,pf=stats(vals)
        label="制限なし" if th is None else f"+{th:.2f}%以下"
        lines.append(f"{label}: 件数 {n} / 勝率 {fmt(wr)}% / 平均 {fmt(avg,3)}% / PF {fmt(pf,3)}")
    lines.append("")
OUT.parent.mkdir(exist_ok=True)
OUT.write_text("\n".join(lines),encoding="utf-8")
print("\n".join(lines))