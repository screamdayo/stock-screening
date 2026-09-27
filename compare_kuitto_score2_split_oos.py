import csv, math, datetime as dt
from pathlib import Path
SRC=Path("data/kuitto_score_trades.csv")
OUT=Path("output/kuitto_score2_split_oos.txt")
ATR_PT=2.917505; DD_PT=-5.855856; TO_PT=828_434_090
BASE={0:None,1:0.0,3:0.0,4:1.0}
rows=[]
with SRC.open(encoding="utf-8-sig",newline="") as f:
  for r in csv.DictReader(f):
    try:
      d=dt.datetime.strptime(r["signal_date"],"%Y-%m-%d").date()
      s=int(float(r["runner_score"])); g=float(r["next_open_gap_pct"]); p=float(r["exit_pnl_pct"]); atr=float(r["atr14_pct"]); dd=float(r["dd20_pct"]); to=float(r["avg_turnover_20"])
    except: continue
    if any(math.isnan(x) for x in (g,p,atr,dd,to)): continue
    sub=""
    if s==2:
      if atr>=ATR_PT and not (dd<=DD_PT and to>=TO_PT): sub="ATR_ONLY"
      elif atr<ATR_PT and dd<=DD_PT and to>=TO_PT: sub="DD_TO"
      else: sub="OTHER2"
    rows.append({"d":d,"s":s,"g":g,"p":p,"sub":sub})

def st(vals):
  if not vals: return None
  n=len(vals); gp=sum(v for v in vals if v>0); gl=-sum(v for v in vals if v<0)
  vv=sorted(vals); med=vv[n//2] if n%2 else (vv[n//2-1]+vv[n//2])/2
  return n,sum(v>0 for v in vals)/n*100,sum(vals)/n,med,(gp/gl if gl else None)
def keep_current(r):
  if r["s"]==2: return r["g"]<=0.25
  th=BASE.get(r["s"]); return True if th is None else r["g"]<=th
def keep_split(r):
  if r["s"]==2:
    if r["sub"]=="ATR_ONLY": return r["g"]<=0.25
    if r["sub"]=="DD_TO": return r["g"]<=0.0
    return r["g"]<=0.25
  th=BASE.get(r["s"]); return True if th is None else r["g"]<=th

periods=[("前半2021-2023",dt.date(2021,1,1),dt.date(2023,12,31)),("後半2024-2026",dt.date(2024,1,1),dt.date(2026,12,31))]
lines=["score2 split pseudo OOS","current: score2<=+0.25 / split: ATR_ONLY<=+0.25, DD_TO<=0.00",""]
for label,a,b in periods:
  sub=[r for r in rows if a<=r["d"]<=b]
  lines.append(label)
  for name,fn in [("現行スコア別",keep_current),("2点分解版",keep_split)]:
    vals=[r["p"] for r in sub if fn(r)]
    n,wr,avg,med,pf=st(vals)
    lines.append(f"{name}: n {n} / win {wr:.2f}% / avg {avg:+.3f}% / median {med:+.3f}% / PF {pf:.3f}")
  lines.append("")
for y in [2024,2025,2026]:
  sub=[r for r in rows if r["d"].year==y]
  lines.append(str(y))
  for name,fn in [("現行スコア別",keep_current),("2点分解版",keep_split)]:
    vals=[r["p"] for r in sub if fn(r)]
    n,wr,avg,med,pf=st(vals)
    lines.append(f"{name}: n {n} / win {wr:.2f}% / avg {avg:+.3f}% / median {med:+.3f}% / PF {pf:.3f}")
  lines.append("")
OUT.parent.mkdir(exist_ok=True)
OUT.write_text("\n".join(lines),encoding="utf-8")
print("\n".join(lines))