import csv, math
from pathlib import Path
SRC=Path("data/kuitto_score_trades.csv")
OUT=Path("output/kuitto_rescue_policy.txt")
BASE={0:None,1:0.0,2:0.25,3:0.0,4:1.0}
ATR_CUT=2.9
rows=[]
with SRC.open(encoding="utf-8-sig",newline="") as f:
  for r in csv.DictReader(f):
    try:
      s=int(float(r["runner_score"])); g=float(r["next_open_gap_pct"]); p=float(r["exit_pnl_pct"]); atr=float(r["atr14_pct"])
    except: continue
    if any(math.isnan(x) for x in (g,p,atr)): continue
    rows.append((s,g,p,atr))

def stats(vals):
  if not vals: return (0,None,None,None,None)
  vals=sorted(vals); n=len(vals); wins=sum(v>0 for v in vals); avg=sum(vals)/n
  med=vals[n//2] if n%2 else (vals[n//2-1]+vals[n//2])/2
  gp=sum(v for v in vals if v>0); gl=-sum(v for v in vals if v<0)
  return n,wins/n*100,avg,med,(gp/gl if gl else None)
def fmt(v,n=3): return "-" if v is None else f"{v:.{n}f}"

def base_keep(s,g):
  th=BASE.get(s)
  return True if th is None else g<=th

def collect(mode):
  vals=[]
  for s,g,p,atr in rows:
    keep=base_keep(s,g)
    if mode=="none": keep=True
    elif mode=="flat05": keep=g<=0.5
    elif mode=="base": pass
    elif mode=="rescue4": keep=keep or s==4
    elif mode=="rescueatr": keep=keep or atr>=ATR_CUT
    elif mode=="rescueeither": keep=keep or s==4 or atr>=ATR_CUT
    if keep: vals.append(p)
  return stats(vals)

cases=[("制限なし","none"),("一律+0.5%","flat05"),("スコア別ベース","base"),("4/4救済","rescue4"),("ATR>=2.9救済","rescueatr"),("4/4 or ATR>=2.9救済","rescueeither")]
lines=["くいっと 除外群救済ルール比較",""]
for label,mode in cases:
  n,wr,avg,med,pf=collect(mode)
  lines.append(f"{label}: 件数 {n} / 勝率 {fmt(wr,2)}% / 平均 {fmt(avg)}% / 中央値 {fmt(med)}% / PF {fmt(pf)}")
OUT.parent.mkdir(exist_ok=True)
OUT.write_text("\n".join(lines),encoding="utf-8")
print("\n".join(lines))