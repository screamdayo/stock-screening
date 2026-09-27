import csv, math, itertools
from pathlib import Path
SRC=Path("data/kuitto_score_trades.csv")
OUT=Path("output/kuitto_rejected_rescue_grid.txt")
BASE={0:None,1:0.0,2:0.25,3:0.0,4:1.0}
rows=[]
with SRC.open(encoding="utf-8-sig",newline="") as f:
  for r in csv.DictReader(f):
    try:
      s=int(float(r["runner_score"])); g=float(r["next_open_gap_pct"]); p=float(r["exit_pnl_pct"]); atr=float(r["atr14_pct"]); dd=float(r["dd20_pct"]); r5=float(r["return_5d_pct"]); r10=float(r["return_10d_pct"]); r15=float(r["return_15d_pct"])
    except: continue
    if any(math.isnan(x) for x in (g,p,atr,dd,r5,r10,r15)): continue
    th=BASE.get(s); keep=True if th is None else g<=th
    if keep: continue
    rows.append({"s":s,"g":g,"p":p,"atr":atr,"dd":dd,"big":max(r5,r10,r15)>=10})

def stats(xs):
  if not xs: return None
  vals=[r["p"] for r in xs]; gp=sum(v for v in vals if v>0); gl=-sum(v for v in vals if v<0)
  return {"n":len(xs),"wr":sum(v>0 for v in vals)/len(vals)*100,"avg":sum(vals)/len(vals),"pf":gp/gl if gl else 99.0,"big":sum(r["big"] for r in xs)}

cands=[]
for minscore in [1,2,3,4]:
  for atrcut in [None,2.7,2.9,3.1,3.3]:
    for ddcut in [None,-4.0,-5.0,-5.86,-7.0]:
      for gmax in [0.5,0.75,1.0,1.5,2.0,3.0,10.0]:
        xs=[]
        for r in rows:
          if r["s"]<minscore: continue
          if atrcut is not None and r["atr"]<atrcut: continue
          if ddcut is not None and r["dd"]>ddcut: continue
          if r["g"]>gmax: continue
          xs.append(r)
        st=stats(xs)
        if not st or st["n"]<10: continue
        cands.append((st["pf"],st["avg"],st["n"],st["big"],minscore,atrcut,ddcut,gmax,st["wr"]))

cands.sort(reverse=True)
lines=[f"除外群 rescue grid / total rejected {len(rows)}","上位は PF優先、最低10件",""]
for pf,avg,n,big,ms,atr,dd,gm,wr in cands[:30]:
  lines.append(f"score>={ms}, ATR>={atr if atr is not None else '-'}, DD20<={dd if dd is not None else '-'}, gap<={gm}%: n {n} / big10 {big} / win {wr:.1f}% / avg {avg:+.3f}% / PF {pf:.3f}")
OUT.parent.mkdir(exist_ok=True)
OUT.write_text("\n".join(lines),encoding="utf-8")
print("\n".join(lines))