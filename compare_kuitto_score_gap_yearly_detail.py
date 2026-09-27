import csv, math, datetime as dt
from pathlib import Path
SRC=Path("data/kuitto_score_trades.csv")
OUT=Path("output/kuitto_score_gap_yearly_detail.txt")
rows=[]
with SRC.open(encoding="utf-8-sig",newline="") as f:
  for r in csv.DictReader(f):
    try:
      d=dt.datetime.strptime(r["signal_date"],"%Y-%m-%d").date()
      s=int(float(r["runner_score"])); g=float(r["next_open_gap_pct"]); p=float(r["exit_pnl_pct"])
    except: continue
    if any(math.isnan(x) for x in (g,p)): continue
    rows.append({"d":d,"s":s,"g":g,"p":p})

def st(vals):
  if not vals: return None
  n=len(vals); gp=sum(v for v in vals if v>0); gl=-sum(v for v in vals if v<0)
  return n,sum(v>0 for v in vals)/n*100,sum(vals)/n,(gp/gl if gl else None)
def fmt(v): return "-" if v is None else f"{v:.3f}"
caps=[-0.25,0.0,0.25,0.5,0.75,1.0,1.5,2.0,99.0]
years=sorted({r["d"].year for r in rows})
lines=["score x gap yearly detail",""]
for s in range(5):
  lines.append(f"===== SCORE {s}/4 =====")
  for y in years:
    sr=[r for r in rows if r["s"]==s and r["d"].year==y]
    if not sr: continue
    parts=[]
    for cap in caps:
      vals=[r["p"] for r in sr if r["g"]<=cap]
      z=st(vals)
      if not z: continue
      n,wr,avg,pf=z
      label="none" if cap==99 else f"{cap:+.2f}"
      parts.append(f"{label}:n{n}/avg{avg:+.2f}/PF{fmt(pf)}")
    lines.append(str(y)+" | "+" | ".join(parts))
  lines.append("")

# fixed-policy chosen cap versus nearby alternatives by score, pooled yearly wins
current={0:99.0,1:0.0,2:0.25,3:0.0,4:1.0}
lines.append("===== CURRENT CAP ROBUSTNESS =====")
for s in range(5):
  testcaps=caps
  scores=[]
  for cap in testcaps:
    yearly=[]
    for y in years:
      vals=[r["p"] for r in rows if r["s"]==s and r["d"].year==y and r["g"]<=cap]
      z=st(vals)
      if z: yearly.append((y,z[0],z[2],z[3]))
    pos=sum(1 for _,_,avg,_ in yearly if avg>0)
    pf1=sum(1 for _,_,_,pf in yearly if pf is not None and pf>1)
    n=sum(x[1] for x in yearly)
    avg=sum(x[1]*x[2] for x in yearly)/n if n else 0
    scores.append((cap,pos,pf1,n,avg))
  scores.sort(key=lambda x:(x[1],x[2],x[4]),reverse=True)
  lines.append(f"SCORE {s}/4 current={current[s] if current[s]!=99 else 'none'}")
  for cap,pos,pf1,n,avg in scores[:5]:
    label="none" if cap==99 else f"{cap:+.2f}%"
    lines.append(f"  {label}: positive-years {pos}/{len(years)} / PF>1 years {pf1}/{len(years)} / n{n} / pooled avg {avg:+.3f}%")
OUT.parent.mkdir(exist_ok=True)
OUT.write_text("\n".join(lines),encoding="utf-8")
print("\n".join(lines))