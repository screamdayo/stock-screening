import csv, math, pandas as pd
from pathlib import Path
import download
SRC=Path("data/kuitto_score_trades.csv")
OUT=Path("output/kuitto_rotation_2022_detail.txt")
LOT=100; MAX_POS=8; CAP={0:None,1:0.0,2:0.25,3:0.0,4:1.0}
OPEN_CACHE={}
def load_trades():
  out=[]
  with SRC.open(encoding="utf-8-sig",newline="") as f:
    for r in csv.DictReader(f):
      try:
        s=int(float(r["runner_score"])); gap=float(r["next_open_gap_pct"]); entry=float(r["entry_open"]); exitp=float(r["exit_price"]); ma=float(r["ma5_vs_ma25_pct"])
        ed=pd.Timestamp(r["entry_date"]); xd=pd.Timestamp(r["exit_date"])
      except: continue
      if any(math.isnan(x) for x in (gap,entry,exitp,ma)): continue
      cap=CAP[s]
      if cap is not None and gap>cap: continue
      out.append({"code":str(r["code"]),"score":s,"entry":entry,"exit":exitp,"ma":ma,"ed":ed,"xd":xd,"exit_reason":r.get("exit_reason","")})
  return out
def get_open(code,d):
  key=(str(code),pd.Timestamp(d))
  if key in OPEN_CACHE: return OPEN_CACHE[key]
  ds=pd.Timestamp(d).strftime("%Y%m%d")
  rows=download._get_bars_for_code(str(code),from_date=ds,to_date=ds)
  if not rows: OPEN_CACHE[key]=None; return None
  df=download._finalize_df(rows)
  if df.empty: OPEN_CACHE[key]=None; return None
  df["Date"]=pd.to_datetime(df["Date"])
  hit=df[df["Date"]==pd.Timestamp(d)]
  if hit.empty: OPEN_CACHE[key]=None; return None
  px=float(hit.iloc[0]["O"]); OPEN_CACHE[key]=px; return px
def weaker(a,b):
  return min([a,b],key=lambda p:(p["score"],-abs(p["ma"]),p["ed"],p["code"]))
def weakest(pos): return min(pos,key=lambda p:(p["score"],-abs(p["ma"]),p["ed"],p["code"]))
tr=[t for t in load_trades() if t["ed"].year==2022]
by={}; dates=set()
for t in tr: by.setdefault(t["ed"],[]).append(t); dates.update([t["ed"],t["xd"]])
cash=1_000_000.0; pos=[]; rots=[]
for d in sorted(dates):
  rem=[]
  for p in pos:
    if p["xd"]==d: cash+=p["exit"]*LOT
    else: rem.append(p)
  pos=rem
  rotated=False
  for x in sorted(by.get(d,[]),key=lambda z:(-z["score"],abs(z["ma"]),z["code"])):
    if any(p["code"]==x["code"] for p in pos): continue
    if len(pos)>=MAX_POS:
      if rotated: continue
      v=weakest(pos)
      if x["score"]<=v["score"]: continue
      px=get_open(v["code"],d)
      if px is None: continue
      cash+=px*LOT; pos.remove(v); rotated=True
      # compare: sold now vs hypothetical hold to its original exit
      sold_now=(px/v["entry"]-1)*100
      sold_hold=(v["exit"]/v["entry"]-1)*100
      new_full=(x["exit"]/x["entry"]-1)*100
      edge=new_full-sold_hold
      rots.append({"date":d,"sold":v,"sold_now":sold_now,"sold_hold":sold_hold,"new":x,"new_full":new_full,"edge":edge})
    cost=x["entry"]*LOT
    if cost>cash+1e-9: continue
    cash-=cost; pos.append(x)
lines=[f"2022 高score入れ替え分解 / {len(rots)}件","edge = 入れ替え先の最終% - 売却銘柄を元の出口まで持った最終%",""]
better=0; worse=0; edges=[]
for r in rots:
  e=r["edge"]; edges.append(e)
  if e>0: better+=1
  elif e<0: worse+=1
  lines.append(f"{r['date'].date()} | 売 {r['sold']['code']} score{r['sold']['score']} 今売{r['sold_now']:+.2f}% / 持続{r['sold_hold']:+.2f}% -> 買 {r['new']['code']} score{r['new']['score']} 最終{r['new_full']:+.2f}% | edge {e:+.2f}pt")
lines += ["",f"入れ替え先が上回った: {better}/{len(rots)}",f"下回った: {worse}/{len(rots)}",f"平均edge: {sum(edges)/len(edges):+.2f}pt" if edges else ""]
OUT.parent.mkdir(exist_ok=True); OUT.write_text("\n".join(lines),encoding="utf-8"); print("\n".join(lines))