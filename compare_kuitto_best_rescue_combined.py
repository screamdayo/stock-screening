import csv, math
from pathlib import Path
SRC=Path("data/kuitto_score_trades.csv")
OUT=Path("output/kuitto_best_rescue_combined.txt")
BASE={0:None,1:0.0,2:0.25,3:0.0,4:1.0}
rows=[]
with SRC.open(encoding="utf-8-sig",newline="") as f:
  for r in csv.DictReader(f):
    try:
      s=int(float(r["runner_score"])); g=float(r["next_open_gap_pct"]); p=float(r["exit_pnl_pct"]); atr=float(r["atr14_pct"]); dd=float(r["dd20_pct"])
    except: continue
    if any(math.isnan(x) for x in (g,p,atr,dd)): continue
    rows.append((s,g,p,atr,dd))

def st(vals):
  vals=sorted(vals); n=len(vals); wins=sum(v>0 for v in vals); gp=sum(v for v in vals if v>0); gl=-sum(v for v in vals if v<0)
  med=vals[n//2] if n%2 else (vals[n//2-1]+vals[n//2])/2
  return n,wins/n*100,sum(vals)/n,med,gp/gl
def keepbase(s,g):
  th=BASE.get(s); return True if th is None else g<=th
cases={
 "スコア別ベース": lambda s,g,a,d: keepbase(s,g),
 "最良17件救済": lambda s,g,a,d: keepbase(s,g) or (a>=3.1 and d<=-5.86 and g<=0.75),
 "少し広め25件系": lambda s,g,a,d: keepbase(s,g) or (a>=3.1 and d<=-5.0 and g<=0.75),
}
lines=[]
for name,fn in cases.items():
  vals=[p for s,g,p,a,d in rows if fn(s,g,a,d)]
  n,wr,avg,med,pf=st(vals)
  lines.append(f"{name}: 件数 {n} / 勝率 {wr:.2f}% / 平均 {avg:+.3f}% / 中央値 {med:+.3f}% / PF {pf:.3f}")
OUT.parent.mkdir(exist_ok=True)
OUT.write_text("\n".join(lines),encoding="utf-8")
print("\n".join(lines))