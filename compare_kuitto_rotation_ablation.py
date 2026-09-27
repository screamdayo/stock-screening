import csv, math
from pathlib import Path
import pandas as pd
import download
SRC=Path("data/kuitto_score_trades.csv"); OUT=Path("output/kuitto_rotation_ablation.txt")
INITIAL=1_000_000.0; LOT=100; MAX_POS=8; CAP={0:None,1:0.0,2:0.25,3:0.0,4:1.0}; OPEN_CACHE={}
def load():
  a=[]
  with SRC.open(encoding="utf-8-sig",newline="") as f:
    for r in csv.DictReader(f):
      try:s=int(float(r["runner_score"]));g=float(r["next_open_gap_pct"]);e=float(r["entry_open"]);x=float(r["exit_price"]);m=float(r["ma5_vs_ma25_pct"]);ed=pd.Timestamp(r["entry_date"]);xd=pd.Timestamp(r["exit_date"])
      except:continue
      if any(math.isnan(v) for v in (g,e,x,m)):continue
      c=CAP[s]
      if c is not None and g>c:continue
      a.append({"code":str(r["code"]),"score":s,"entry":e,"exit":x,"ma":m,"ed":ed,"xd":xd})
  return a
def get_open(code,d):
  k=(code,d)
  if k in OPEN_CACHE:return OPEN_CACHE[k]
  ds=d.strftime("%Y%m%d");rows=download._get_bars_for_code(code,from_date=ds,to_date=ds)
  if not rows:OPEN_CACHE[k]=None;return None
  df=download._finalize_df(rows);df["Date"]=pd.to_datetime(df["Date"]);h=df[df["Date"]==d]
  if h.empty:OPEN_CACHE[k]=None;return None
  OPEN_CACHE[k]=float(h.iloc[0]["O"]);return OPEN_CACHE[k]
def weak(pos):return min(pos,key=lambda p:(p["score"],-abs(p["ma"]),p["ed"],p["code"]))
def allowed(n,o,mode):
  if n["score"]<=o["score"]:return False
  pair=(o["score"],n["score"])
  if mode=="any":return True
  if mode=="no_bad3":return pair not in {(0,2),(0,3),(1,2)}
  if mode=="no_12":return pair!=(1,2)
  if mode=="no_02":return pair!=(0,2)
  if mode=="no_03":return pair!=(0,3)
  if mode=="pos_pairs":return pair in {(0,1),(0,4),(1,3),(1,4),(2,3),(3,4)}
  return False
def sim(tr,mode):
  by={};dates=set()
  for t in tr:by.setdefault(t["ed"],[]).append(t);dates.update([t["ed"],t["xd"]])
  cash=INITIAL;pos=[];closed=[];rots=0
  for d in sorted(dates):
    rem=[]
    for p in pos:
      if p["xd"]==d:cash+=p["exit"]*LOT;closed.append((p["exit"]/p["entry"]-1)*100)
      else:rem.append(p)
    pos=rem;rot=False
    for n in sorted(by.get(d,[]),key=lambda z:(-z["score"],abs(z["ma"]),z["code"])):
      if any(p["code"]==n["code"] for p in pos):continue
      if len(pos)>=MAX_POS:
        if rot:continue
        o=weak(pos)
        if not allowed(n,o,mode):continue
        px=get_open(o["code"],d)
        if px is None:continue
        cash+=px*LOT;closed.append((px/o["entry"]-1)*100);pos.remove(o);rots+=1;rot=True
      cost=n["entry"]*LOT
      if cost>cash+1e-9:continue
      cash-=cost;pos.append(n)
  for p in pos:cash+=p["exit"]*LOT;closed.append((p["exit"]/p["entry"]-1)*100)
  gp=sum(v for v in closed if v>0);gl=-sum(v for v in closed if v<0)
  years=(max(t["xd"] for t in tr)-min(t["ed"] for t in tr)).days/365.2425
  return (cash/INITIAL-1)*100,((cash/INITIAL)**(1/years)-1)*100,(gp/gl if gl else None),rots,cash
def main():
  tr=load();modes=[("any","全部"),("no_12","1→2除外"),("no_02","0→2除外"),("no_03","0→3除外"),("no_bad3","0→2/0→3/1→2除外"),("pos_pairs","孤立比較プラス遷移だけ")]
  lines=["高score入替 アブレーション",""]
  for mode,label in modes:
    r,c,p,n,f=sim(tr,mode);lines.append(f"{label}: 最終 {f:,.0f}円 / 累積 {r:+.2f}% / CAGR {c:+.2f}% / PF {p:.3f} / 入替{n}")
  lines.append("")
  for y in sorted({t["ed"].year for t in tr}):
    sub=[t for t in tr if t["ed"].year==y];lines.append(str(y))
    for mode,label in modes:
      r,c,p,n,f=sim(sub,mode);lines.append(f"{label}: {r:+.2f}% / PF {p:.3f} / 入替{n}")
  OUT.parent.mkdir(exist_ok=True);OUT.write_text("\n".join(lines),encoding="utf-8");print("\n".join(lines))
if __name__=="__main__":main()