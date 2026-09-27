import csv, math, time
from pathlib import Path
import pandas as pd
import download

SRC=Path("data/kuitto_score_trades.csv"); OUT=Path("output/kuitto_rotation_fast.txt")
INITIAL=1_000_000.0; LOT=100; MAX_POS=8; CAP={0:None,1:0.0,2:0.25,3:0.0,4:1.0}
def load():
 out=[]
 with SRC.open(encoding="utf-8-sig",newline="") as f:
  for r in csv.DictReader(f):
   try:
    s=int(float(r["runner_score"])); gap=float(r["next_open_gap_pct"]); entry=float(r["entry_open"]); exitp=float(r["exit_price"]); ma=float(r["ma5_vs_ma25_pct"]); ed=pd.Timestamp(r["entry_date"]); xd=pd.Timestamp(r["exit_date"])
   except: continue
   if any(math.isnan(x) for x in (gap,entry,exitp,ma)): continue
   cap=CAP[s]
   if cap is not None and gap>cap: continue
   out.append({"code":str(r["code"]),"score":s,"gap":gap,"entry":entry,"exit":exitp,"ma":ma,"ed":ed,"xd":xd})
 return out
def fetch_opens(dates):
 m={}
 for i,d in enumerate(sorted(dates),1):
  ds=d.strftime("%Y%m%d"); rows=download._get_bars_for_date(ds)
  for r in rows:
   code=str(r.get("Code") or r.get("code") or "")[:4]
   op=r.get("O",r.get("Open",r.get("open")))
   if code and op is not None:
    try:m[(code,d)]=float(op)
    except:pass
  if i%50==0: print(f"dates {i}/{len(dates)}")
  time.sleep(0.03)
 return m
def weakest(pos): return min(pos,key=lambda p:(p["score"],-abs(p["ma"]),p["ed"],p["code"]))
def strong(n,o,mode):
 if n["score"]>o["score"]: return True
 return mode=="score_ma" and n["score"]==o["score"] and abs(n["ma"])<abs(o["ma"])-1e-12
def sim(tr,om,mode):
 by={}; dates=set()
 for t in tr: by.setdefault(t["ed"],[]).append(t); dates|={t["ed"],t["xd"]}
 cash=INITIAL; pos=[]; done=[]; rots=0; ss=sc=sd=0; buys=0; peak=INITIAL; maxdd=0
 for d in sorted(dates):
  rem=[]
  for p in pos:
   if p["xd"]==d: cash+=p["exit"]*LOT; done.append((p,(p["exit"]/p["entry"]-1)*100))
   else: rem.append(p)
  pos=rem; rotated=False
  for x in sorted(by.get(d,[]),key=lambda z:(-z["score"],abs(z["ma"]),z["code"])):
   if any(p["code"]==x["code"] for p in pos): sd+=1; continue
   if len(pos)>=MAX_POS:
    if mode=="none" or rotated: ss+=1; continue
    v=weakest(pos)
    if not strong(x,v,mode): ss+=1; continue
    px=om.get((v["code"],d))
    if px is None: ss+=1; continue
    cash+=px*LOT; done.append((v,(px/v["entry"]-1)*100)); pos.remove(v); rots+=1; rotated=True
   cost=x["entry"]*LOT
   if cost>cash+1e-9: sc+=1; continue
   cash-=cost; pos.append(x); buys+=1
  eq=cash+sum(p["entry"]*LOT for p in pos); peak=max(peak,eq); maxdd=min(maxdd,(eq/peak-1)*100)
 for p in pos: cash+=p["exit"]*LOT; done.append((p,(p["exit"]/p["entry"]-1)*100))
 gp=sum(p["entry"]*LOT*pct/100 for p,pct in done if pct>0); gl=-sum(p["entry"]*LOT*pct/100 for p,pct in done if pct<0)
 yrs=(max(t["xd"] for t in tr)-min(t["ed"] for t in tr)).days/365.2425; cagr=((cash/INITIAL)**(1/yrs)-1)*100
 return cash,(cash/INITIAL-1)*100,cagr,gp/gl if gl else 0,buys,rots,ss,sc,maxdd
def main():
 tr=load(); entry_dates={t["ed"] for t in tr}; print("entry dates",len(entry_dates)); om=fetch_opens(entry_dates)
 lines=["くいっと8枠 入れ替え比較 高速版","100万円/100株/8枠/現行gap/入替は1日最大1回/入替売却=当日始値/税手数料なし",""]
 for mode,label in [("none","入替なし"),("score","新候補scoreが高い時のみ"),("score_ma","score高い or 同scoreでMA25に近い")]:
  z=sim(tr,om,mode); lines.append(f"{label}: 最終 {z[0]:,.0f}円 / 累積 {z[1]:+.2f}% / CAGR {z[2]:+.2f}% / PF {z[3]:.3f} / 買付{z[4]} / 入替{z[5]} / 枠見送り{z[6]} / 資金見送り{z[7]} / DD参考{z[8]:.2f}%")
 OUT.parent.mkdir(exist_ok=True); OUT.write_text("\n".join(lines),encoding="utf-8"); print("\n".join(lines))
if __name__=="__main__": main()