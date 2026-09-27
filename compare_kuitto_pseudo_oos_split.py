import csv, math, datetime as dt
from pathlib import Path
SRC=Path("data/kuitto_score_trades.csv")
OUT=Path("output/kuitto_pseudo_oos_split.txt")
BASE={0:None,1:0.0,2:0.25,3:0.0,4:1.0}
RESCUE_ATR=3.10; RESCUE_DD=-6.10; RESCUE_GAP=0.75
rows=[]
with SRC.open(encoding="utf-8-sig",newline="") as f:
  for r in csv.DictReader(f):
    try:
      d=dt.datetime.strptime(r["signal_date"],"%Y-%m-%d").date()
      s=int(float(r["runner_score"])); g=float(r["next_open_gap_pct"]); p=float(r["exit_pnl_pct"]); atr=float(r["atr14_pct"]); dd=float(r["dd20_pct"])
    except: continue
    if any(math.isnan(x) for x in (g,p,atr,dd)): continue
    rows.append({"d":d,"s":s,"g":g,"p":p,"atr":atr,"dd":dd})

def st(vals):
  if not vals: return None
  n=len(vals); wins=sum(v>0 for v in vals); gp=sum(v for v in vals if v>0); gl=-sum(v for v in vals if v<0)
  vals2=sorted(vals); med=vals2[n//2] if n%2 else (vals2[n//2-1]+vals2[n//2])/2
  return n,wins/n*100,sum(vals)/n,med,(gp/gl if gl else None)
def base_keep(s,g):
  th=BASE.get(s); return True if th is None else g<=th
def fmt(x): return "-" if x is None else f"{x:.3f}"

periods=[("前半2021-2023",dt.date(2021,1,1),dt.date(2023,12,31)),("後半2024-2026",dt.date(2024,1,1),dt.date(2026,12,31))]
lines=["くいっと 疑似OOS期間分割","固定条件: score別=0無制限/1<=0/2<=+0.25/3<=0/4<=+1.0; rescue=ATR>=3.10, DD20<=-6.10, gap<=+0.75",""]
for label,start,end in periods:
  sub=[r for r in rows if start<=r["d"]<=end]
  cases=[]
  vals=[r["p"] for r in sub if r["g"]<=0.5]; cases.append(("一律+0.5%",vals))
  vals=[r["p"] for r in sub if base_keep(r["s"],r["g"])]; cases.append(("スコア別",vals))
  vals=[r["p"] for r in sub if base_keep(r["s"],r["g"]) or (r["atr"]>=RESCUE_ATR and r["dd"]<=RESCUE_DD and r["g"]<=RESCUE_GAP)]; cases.append(("スコア別+救済",vals))
  lines.append(label)
  for name,vals in cases:
    n,wr,avg,med,pf=st(vals)
    lines.append(f"{name}: n {n} / win {wr:.2f}% / avg {avg:+.3f}% / median {med:+.3f}% / PF {fmt(pf)}")
  lines.append("")

# 後半年別
for y in [2024,2025,2026]:
  sub=[r for r in rows if r["d"].year==y]
  lines.append(str(y))
  for name,fn in [
    ("一律+0.5%", lambda r:r["g"]<=0.5),
    ("スコア別", lambda r:base_keep(r["s"],r["g"])),
    ("スコア別+救済", lambda r:base_keep(r["s"],r["g"]) or (r["atr"]>=RESCUE_ATR and r["dd"]<=RESCUE_DD and r["g"]<=RESCUE_GAP))
  ]:
    vals=[r["p"] for r in sub if fn(r)]
    n,wr,avg,med,pf=st(vals)
    lines.append(f"{name}: n {n} / win {wr:.2f}% / avg {avg:+.3f}% / PF {fmt(pf)}")
  lines.append("")
OUT.parent.mkdir(exist_ok=True)
OUT.write_text("\n".join(lines),encoding="utf-8")
print("\n".join(lines))