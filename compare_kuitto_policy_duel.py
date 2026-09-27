import csv, math, datetime as dt
from pathlib import Path
SRC=Path("data/kuitto_score_trades.csv")
OUT=Path("output/kuitto_policy_duel.txt")
CUR={0:None,1:0.0,2:0.25,3:0.0,4:1.0}
ALT={0:0.5,1:0.5,2:0.0,3:0.0,4:0.75}
rows=[]
with SRC.open(encoding="utf-8-sig",newline="") as f:
  for r in csv.DictReader(f):
    try:
      d=dt.datetime.strptime(r["signal_date"],"%Y-%m-%d").date()
      s=int(float(r["runner_score"])); g=float(r["next_open_gap_pct"]); p=float(r["exit_pnl_pct"])
    except: continue
    if any(math.isnan(x) for x in (g,p)): continue
    rows.append({"d":d,"s":s,"g":g,"p":p})

def keep(r,rule):
  cap=rule[r["s"]]
  return True if cap is None else r["g"]<=cap
def st(vals):
  n=len(vals); gp=sum(v for v in vals if v>0); gl=-sum(v for v in vals if v<0)
  vv=sorted(vals); med=vv[n//2] if n%2 else (vv[n//2-1]+vv[n//2])/2
  return {"n":n,"wr":sum(v>0 for v in vals)/n*100,"avg":sum(vals)/n,"med":med,"pf":gp/gl if gl else None}
def line(name,z): return f"{name}: n {z['n']} / win {z['wr']:.2f}% / avg {z['avg']:+.3f}% / median {z['med']:+.3f}% / PF {z['pf']:.3f}"

periods=[("全期間",None,None),("前半2021-2023",dt.date(2021,1,1),dt.date(2023,12,31)),("後半2024-2026",dt.date(2024,1,1),dt.date(2026,12,31))]
lines=["くいっと policy duel","現行: 0=none /1=0 /2=+0.25 /3=0 /4=+1.0","安定案: 0=+0.5 /1=+0.5 /2=0 /3=0 /4=+0.75",""]
for label,a,b in periods:
  sub=rows if a is None else [r for r in rows if a<=r["d"]<=b]
  lines.append(label)
  for name,rule in [("現行",CUR),("安定案",ALT)]:
    vals=[r["p"] for r in sub if keep(r,rule)]
    lines.append(line(name,st(vals)))
  lines.append("")
for y in sorted({r["d"].year for r in rows}):
  sub=[r for r in rows if r["d"].year==y]
  lines.append(str(y))
  for name,rule in [("現行",CUR),("安定案",ALT)]:
    vals=[r["p"] for r in sub if keep(r,rule)]
    lines.append(line(name,st(vals)))
  lines.append("")
OUT.parent.mkdir(exist_ok=True)
OUT.write_text("\n".join(lines),encoding="utf-8")
print("\n".join(lines))