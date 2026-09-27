import csv, math, time, datetime as dt
from pathlib import Path
import download
SRC=Path("data/kuitto_score_trades.csv"); OUT=Path("output/kuitto_rotation_lazy.txt")
INITIAL=1_000_000.0; LOT=100; MAX_POS=8; CAP={0:None,1:0.0,2:0.25,3:0.0,4:1.0}
def D(s): return dt.datetime.strptime(s,"%Y-%m-%d").date()
def load():
 out=[]
 with SRC.open(encoding="utf-8-sig",newline="") as f:
  for r in csv.DictReader(f):
   try:s=int(float(r["runner_score"])); gap=float(r["next_open_gap_pct"]); ep=float(r["entry_open"]); xp=float(r["exit_price"]); ma=float(r["ma5_vs_ma25_pct"]); ed=D(r["entry_date"]); xd=D(r["exit_date"])
   except:continue
   cap=CAP[s]
   if cap is not None and gap>cap:continue
   out.append({"code":str(r["code"]),"score":s,"entry":ep,"exit":xp,"ma":ma,"ed":ed,"xd":xd})
 return out
_oc={}
def open_at(code,d):
 k=(code,d)
 if k in _oc:return _oc[k]
 ds=d.strftime("%Y%m%d"); rows=download._get_bars_for_code(code,from_date=ds,to_date=ds)
 px=None
 for r in rows:
  v=r.get("O",r.get("Open",r.get("open")))
  if v is not None: px=float(v); break
 _oc[k]=px; time.sleep(0.03); return px
def weakest(pos):return min(pos,key=lambda p:(p["score"],-abs(p["ma"]),p["ed"],p["code"]))
def stronger(n,o,mode):return n["score"]>o["score"] or (mode=="score_ma" and n["score"]==o["score"] and abs(n["ma"])<abs(o["ma"])-1e-12)
def sim(tr,mode):
 by={}; dates=set()
 for t in tr:by.setdefault(t["ed"],[]).append(t);dates|={t["ed"],t["xd"]}
 cash=INITIAL;pos=[];done=[];rots=slots=cashskip=dup=buys=0;peak=INITIAL;maxdd=0
 for d in sorted(dates):
  rem=[]
  for p in pos:
   if p["xd"]==d:cash+=p["exit"]*LOT;done.append((p,(p["exit"]/p["entry"]-1)*100))
   else:rem.append(p)
  pos=rem;rotated=False
  for x in sorted(by.get(d,[]),key=lambda z:(-z["score"],abs(z["ma"]),z["code"])):
   if any(p["code"]==x["code"] for p in pos):dup+=1;continue
   if len(pos)>=MAX_POS:
    if mode=="none" or rotated:slots+=1;continue
    v=weakest(pos)
    if not stronger(x,v,mode):slots+=1;continue
    px=open_at(v["code"],d)
    if px is None:slots+=1;continue
    cash+=px*LOT;done.append((v,(px/v["entry"]-1)*100));pos.remove(v);rots+=1;rotated=True
   cost=x["entry"]*LOT
   if cost>cash+1e-9:cashskip+=1;continue
   cash-=cost;pos.append(x);buys+=1
  eq=cash+sum(p["entry"]*LOT for p in pos);peak=max(peak,eq);maxdd=min(maxdd,(eq/peak-1)*100)
 for p in pos:cash+=p["exit"]*LOT;done.append((p,(p["exit"]/p["entry"]-1)*100))
 gp=sum(p["entry"]*LOT*q/100 for p,q in done if q>0);gl=-sum(p["entry"]*LOT*q/100 for p,q in done if q<0)
 yrs=(max(t["xd"] for t in tr)-min(t["ed"] for t in tr)).days/365.2425;cagr=((cash/INITIAL)**(1/yrs)-1)*100
 return cash,(cash/INITIAL-1)*100,cagr,gp/gl if gl else 0,buys,rots,slots,cashskip,maxdd
def main():
 tr=load(); lines=["くいっと8枠 入替比較 lazy","100万円/100株/8枠/1日最大1入替/当日始値売却/税手数料なし",""]
 for mode,label in [("none","入替なし"),("score","score高い時のみ"),("score_ma","score高い or 同scoreでMA25近い")]:
  z=sim(tr,mode);lines.append(f"{label}: 最終 {z[0]:,.0f}円 / 累積 {z[1]:+.2f}% / CAGR {z[2]:+.2f}% / PF {z[3]:.3f} / 買付{z[4]} / 入替{z[5]} / 枠見送り{z[6]} / 資金見送り{z[7]} / DD参考{z[8]:.2f}%")
 lines.append(f"入替価格API取得 {len(_oc)}件");OUT.parent.mkdir(exist_ok=True);OUT.write_text("\n".join(lines),encoding="utf-8");print("\n".join(lines))
if __name__=="__main__":main()