import csv, math
from pathlib import Path
SRC=Path("data/kuitto_score_trades.csv")
OUT=Path("output/kuitto_score2_split_gap.txt")
BASE={0:None,1:0.0,3:0.0,4:1.0}
ATR_PT=2.917505; DD_PT=-5.855856; TO_PT=828_434_090
rows=[]
with SRC.open(encoding="utf-8-sig",newline="") as f:
  for r in csv.DictReader(f):
    try:
      s=int(float(r["runner_score"])); g=float(r["next_open_gap_pct"]); p=float(r["exit_pnl_pct"]); atr=float(r["atr14_pct"]); dd=float(r["dd20_pct"]); to=float(r["avg_turnover_20"])
    except: continue
    if any(math.isnan(x) for x in (g,p,atr,dd,to)): continue
    subtype=""
    if s==2:
      if atr>=ATR_PT and not (dd<=DD_PT and to>=TO_PT): subtype="ATR_ONLY"
      elif atr<ATR_PT and dd<=DD_PT and to>=TO_PT: subtype="DD_TO"
      else: subtype="OTHER2"
    rows.append({"s":s,"g":g,"p":p,"sub":subtype})

def st(vals):
  if not vals: return None
  n=len(vals); gp=sum(v for v in vals if v>0); gl=-sum(v for v in vals if v<0)
  vv=sorted(vals); med=vv[n//2] if n%2 else (vv[n//2-1]+vv[n//2])/2
  return n,sum(v>0 for v in vals)/n*100,sum(vals)/n,med,(gp/gl if gl else None)
def fmt(v): return "-" if v is None else f"{v:.3f}"

gaps=[-0.25,0.0,0.25,0.5,0.75,1.0,1.5,2.0,99.0]
lines=["score2 subtype gap test",""]
for sub in ["ATR_ONLY","DD_TO","OTHER2"]:
  lines.append(sub)
  sr=[r for r in rows if r["s"]==2 and r["sub"]==sub]
  for cap in gaps:
    vals=[r["p"] for r in sr if r["g"]<=cap]
    z=st(vals)
    if not z: continue
    n,wr,avg,med,pf=z
    label="none" if cap==99 else f"<={cap:+.2f}%"
    lines.append(f"{label}: n {n} / win {wr:.2f}% / avg {avg:+.3f}% / median {med:+.3f}% / PF {fmt(pf)}")
  lines.append("")

# 全体合成: ATR_ONLY cap x DD_TO cap
cands=[]
caps=[-0.25,0.0,0.25,0.5,0.75,1.0,1.5,2.0,99.0]
for ca in caps:
  for cd in caps:
    vals=[]
    for r in rows:
      if r["s"]==2:
        cap = ca if r["sub"]=="ATR_ONLY" else cd if r["sub"]=="DD_TO" else 0.25
        if r["g"]<=cap: vals.append(r["p"])
      else:
        th=BASE.get(r["s"])
        if th is None or r["g"]<=th: vals.append(r["p"])
    n,wr,avg,med,pf=st(vals)
    cands.append((pf,avg,n,wr,ca,cd,med))
cands.sort(reverse=True)
lines.append("combined top 20")
for pf,avg,n,wr,ca,cd,med in cands[:20]:
  la="none" if ca==99 else f"{ca:+.2f}%"
  ld="none" if cd==99 else f"{cd:+.2f}%"
  lines.append(f"ATR_ONLY {la} / DD_TO {ld}: n {n} / win {wr:.2f}% / avg {avg:+.3f}% / median {med:+.3f}% / PF {pf:.3f}")
OUT.parent.mkdir(exist_ok=True)
OUT.write_text("\n".join(lines),encoding="utf-8")
print("\n".join(lines))