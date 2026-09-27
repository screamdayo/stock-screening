"""固定CSVでスコア別ギャップ上限ルールを合成比較。"""
import csv, math
from pathlib import Path

SRC=Path("data/kuitto_score_trades.csv")
OUT=Path("output/kuitto_score_gap_policy.txt")
POLICY={0:None,1:0.0,2:0.25,3:0.0,4:1.0}

def stats(vals):
    if not vals: return {"n":0,"wr":None,"avg":None,"med":None,"pf":None}
    vals=sorted(vals)
    wins=sum(v>0 for v in vals)
    avg=sum(vals)/len(vals)
    med=vals[len(vals)//2] if len(vals)%2 else (vals[len(vals)//2-1]+vals[len(vals)//2])/2
    gp=sum(v for v in vals if v>0)
    gl=-sum(v for v in vals if v<0)
    return {"n":len(vals),"wr":wins/len(vals)*100,"avg":avg,"med":med,"pf":gp/gl if gl else None}

def fmt(v,n=3): return "-" if v is None else f"{v:.{n}f}"
rows=[]
with SRC.open(encoding="utf-8-sig",newline="") as f:
    for r in csv.DictReader(f):
        try:
            s=int(float(r["runner_score"])); g=float(r["next_open_gap_pct"]); p=float(r["exit_pnl_pct"])
        except (TypeError,ValueError,KeyError): continue
        if math.isnan(g) or math.isnan(p): continue
        rows.append((s,g,p))

def pick(mode):
    vals=[]
    for s,g,p in rows:
        if mode=="none": ok=True
        elif mode=="flat05": ok=g<=0.5
        else:
            th=POLICY.get(s)
            ok=True if th is None else g<=th
        if ok: vals.append(p)
    return stats(vals)

cases=[("制限なし","none"),("一律+0.5%","flat05"),("スコア別ルール","policy")]
lines=["くいっと スコア別ギャップ上限 合成比較","ルール: 0/4=制限なし, 1/4=0%以下, 2/4=+0.25%以下, 3/4=0%以下, 4/4=+1.0%以下",""]
for label,mode in cases:
    s=pick(mode)
    lines.append(f"{label}: 件数 {s['n']} / 勝率 {fmt(s['wr'],2)}% / 平均 {fmt(s['avg'])}% / 中央値 {fmt(s['med'])}% / PF {fmt(s['pf'])}")
OUT.parent.mkdir(exist_ok=True)
OUT.write_text("\n".join(lines),encoding="utf-8")
print("\n".join(lines))