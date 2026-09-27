import csv, math
from pathlib import Path
SRC=Path("data/kuitto_score_trades.csv")
OUT=Path("output/kuitto_rescue_finegrid.txt")
BASE={0:None,1:0.0,2:0.25,3:0.0,4:1.0}
rows=[]
with SRC.open(encoding="utf-8-sig",newline="") as f:
  for r in csv.DictReader(f):
    try:
      s=int(float(r["runner_score"])); g=float(r["next_open_gap_pct"]); p=float(r["exit_pnl_pct"]); atr=float(r["atr14_pct"]); dd=float(r["dd20_pct"])
    except: continue
    if any(math.isnan(x) for x in (g,p,atr,dd)): continue
    th=BASE.get(s); base=True if th is None else g<=th
    rows.append({"s":s,"g":g,"p":p,"atr":atr,"dd":dd,"base":base})

def st(vals):
  if not vals: return None
  gp=sum(v for v in vals if v>0); gl=-sum(v for v in vals if v<0)
  return {"n":len(vals),"wr":sum(v>0 for v in vals)/len(vals)*100,"avg":sum(vals)/len(vals),"pf":gp/gl if gl else 99.0}

atrs=[3.0,3.05,3.1,3.15,3.2]
dds=[-5.5,-5.6,-5.7,-5.8,-5.86,-5.9,-6.0,-6.1,-6.2]
gaps=[0.6,0.65,0.7,0.75,0.8,0.85,0.9]
cands=[]
basevals=[r["p"] for r in rows if r["base"]]
bst=st(basevals)
for a in atrs:
  for d in dds:
    for gm in gaps:
      rescue=[r for r in rows if (not r["base"]) and r["atr"]>=a and r["dd"]<=d and r["g"]<=gm]
      rs=st([r["p"] for r in rescue])
      if not rs or rs["n"]<8: continue
      comb=st(basevals+[r["p"] for r in rescue])
      cands.append({"a":a,"d":d,"g":gm,"rn":rs["n"],"ravg":rs["avg"],"rpf":rs["pf"],"cn":comb["n"],"cavg":comb["avg"],"cpf":comb["pf"],"cwr":comb["wr"]})

cands.sort(key=lambda x:(x["cpf"],x["cavg"],x["rn"]),reverse=True)
lines=[f"ベース: n {bst['n']} / avg {bst['avg']:+.3f}% / PF {bst['pf']:.3f}","", "全体PF上位30"]
for x in cands[:30]:
  lines.append(f"ATR>={x['a']:.2f}, DD20<={x['d']:.2f}, gap<={x['g']:.2f}% | rescue n{x['rn']} avg{x['ravg']:+.3f}% PF{x['rpf']:.3f} | combined n{x['cn']} win{x['cwr']:.2f}% avg{x['cavg']:+.3f}% PF{x['cpf']:.3f}")
OUT.parent.mkdir(exist_ok=True)
OUT.write_text("\n".join(lines),encoding="utf-8")
print("\n".join(lines))