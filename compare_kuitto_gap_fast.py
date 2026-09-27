"""固定CSVだけで翌朝ギャップ上限を比較する軽量集計。"""
import csv, math, statistics
from pathlib import Path

SRC=Path("data/kuitto_score_trades.csv")
OUT=Path("output/kuitto_gap_comparison.txt")
THRESHOLDS=[-0.5,0.0,0.25,0.5,0.75,1.0,1.5,2.0,None]

def pf(vals):
    gp=sum(v for v in vals if v>0)
    gl=-sum(v for v in vals if v<0)
    return gp/gl if gl else None

def fmt(v,n=3):
    return "-" if v is None else f"{v:.{n}f}"

rows=[]
with SRC.open(encoding="utf-8-sig",newline="") as f:
    for r in csv.DictReader(f):
        try:
            gap=float(r["next_open_gap_pct"])
            pnl=float(r["exit_pnl_pct"])
        except (TypeError,ValueError,KeyError):
            continue
        if math.isnan(gap) or math.isnan(pnl):
            continue
        rows.append((gap,pnl))

lines=[
    "くいっと押し目版 翌朝ギャップ上限 再検証（固定CSV）",
    f"対象トレード: {len(rows)}件",
    "",
]
for th in THRESHOLDS:
    vals=[pnl for gap,pnl in rows if th is None or gap<=th]
    wins=sum(v>0 for v in vals)
    wr=wins/len(vals)*100 if vals else None
    avg=sum(vals)/len(vals) if vals else None
    med=statistics.median(vals) if vals else None
    p=pf(vals)
    label="制限なし" if th is None else f"+{th:.2f}%以下" if th>=0 else f"{th:.2f}%以下"
    lines.append(
        f"{label}: 件数 {len(vals)} / 勝率 {fmt(wr,2)}% / 平均 {fmt(avg)}% / 中央値 {fmt(med)}% / PF {fmt(p)}"
    )

OUT.parent.mkdir(exist_ok=True)
OUT.write_text("\n".join(lines),encoding="utf-8")
print("\n".join(lines))
