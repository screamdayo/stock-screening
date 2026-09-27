import csv, math
from pathlib import Path
import pandas as pd
import download

SRC=Path("data/kuitto_score_trades.csv"); OUT=Path("output/kuitto_rotation_score4_test.txt")
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
  ds=d.strftime("%Y%m%d"); rows=download._get_bars_for_code(code,from_date=ds,to_date=ds)
  if not rows:OPEN_CACHE[k]=None;return None
  df=download._finalize_df(rows);df["Date"]=pd.to_datetime(df["Date"]);h=df[df["Date"]==d]
  if h.empty:OPEN_CACHE[k]=None;return None
  OPEN_CACHE[k]=float(h.iloc[0]["O"]);return OPEN_CACHE[k]
def weak(pos):return min(pos,key=lambda p:(p["score"],-abs(p["ma"]),p["ed"],p["code"]))
def allowed(n,o,mode):
  if mode=="none":return False
  if mode=="any":return n["score"]>o["score"]
  if mode=="score4":return n["score"]==4 and o["score"]<4
  if mode=="score4_old01":return n["score"]==4 and o["score"]<=1
  if mode=="score3plus":return n["score"]>=3 and n["score"]>o["score"]
  return False
def sim(tr,mode):
  by={};dates=set()
  for t in tr:by.setdefault(t["ed"],[]).append(t);dates.update([t["ed"],t["xd"]])
  cash=INITIAL;pos=[];rots=0;closed=[]
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
  return cash,(cash/INITIAL-1)*100,((cash/INITIAL)**(1/years)-1)*100,(gp/gl if gl else None),rots
def main():
  tr=load();lines=["高score入れ替えの中身再検証",""]
  for mode,label in [("none","入替なし"),("any","今の高scoreなら入替"),("score3plus","新score3以上だけ"),("score4","新score4だけ"),("score4_old01","新score4かつ旧score0/1だけ")]:
    f,r,c,p,n=sim(tr,mode);lines.append(f"{label}: 最終 {f:,.0f}円 / 累積 {r:+.2f}% / CAGR {c:+.2f}% / PF {p:.3f} / 入替{n}")
  lines.append("")
  for y in sorted({t["ed"].year for t in tr}):
    sub=[t for t in tr if t["ed"].year==y];lines.append(str(y))
    for mode,label in [("none","なし"),("any","高score"),("score4","score4のみ"),("score4_old01","score4←旧0/1")]:
      f,r,c,p,n=sim(sub,mode);lines.append(f"{label}: {r:+.2f}% / PF {p:.3f} / 入替{n}")
  OUT.parent.mkdir(exist_ok=True);OUT.write_text("\n".join(lines),encoding="utf-8");print("\n".join(lines))
if __name__=="__main__":main()