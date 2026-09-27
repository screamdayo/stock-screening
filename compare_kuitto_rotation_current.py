import csv, math, os, datetime as dt
from pathlib import Path
import pandas as pd
import config, download

SRC=Path("data/kuitto_score_trades.csv")
OUT=Path("output/kuitto_rotation_current.txt")
INITIAL=1_000_000.0; LOT=100; MAX_POS=8
CAP={0:None,1:0.0,2:0.25,3:0.0,4:1.0}

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
      out.append({"code":str(r["code"]),"score":s,"gap":gap,"entry":entry,"exit":exitp,"ma":ma,"ed":ed,"xd":xd})
  return out

OPEN_CACHE = {}

def get_open(code, d):
  key=(str(code),pd.Timestamp(d))
  if key in OPEN_CACHE: return OPEN_CACHE[key]
  ds=pd.Timestamp(d).strftime("%Y%m%d")
  rows=download._get_bars_for_code(str(code),from_date=ds,to_date=ds)
  if not rows:
    OPEN_CACHE[key]=None
    return None
  df=download._finalize_df(rows)
  if df.empty:
    OPEN_CACHE[key]=None
    return None
  df["Date"]=pd.to_datetime(df["Date"])
  hit=df[df["Date"]==pd.Timestamp(d)]
  if hit.empty:
    OPEN_CACHE[key]=None
    return None
  px=float(hit.iloc[0]["O"])
  OPEN_CACHE[key]=px
  return px

def stronger(new,old,mode):
  if new["score"]>old["score"]: return True
  if mode=="score_ma" and new["score"]==old["score"] and abs(new["ma"])<abs(old["ma"])-1e-12: return True
  return False

def weakest(pos):
  return min(pos,key=lambda p:(p["score"],-abs(p["ma"]),p["ed"],p["code"]))

def simulate(trades,mode):
  by={}
  dates=set()
  for t in trades:
    by.setdefault(t["ed"],[]).append(t); dates.add(t["ed"]); dates.add(t["xd"])
  cash=INITIAL; pos=[]; done=[]; rotations=[]; skips_slot=skips_cash=skips_dup=0; buys=0
  peak=INITIAL; maxdd=0.0
  for d in sorted(dates):
    # normal exits at that date open
    rem=[]
    for p in pos:
      if p["xd"]==d:
        cash += p["exit"]*LOT; done.append((p,(p["exit"]/p["entry"]-1)*100,"normal"))
      else: rem.append(p)
    pos=rem
    cands=sorted(by.get(d,[]),key=lambda x:(-x["score"],abs(x["ma"]),x["code"]))
    rotated=False
    for x in cands:
      if any(p["code"]==x["code"] for p in pos): skips_dup+=1; continue
      if len(pos)>=MAX_POS:
        if mode=="none" or rotated:
          skips_slot+=1; continue
        v=weakest(pos)
        if not stronger(x,v,mode):
          skips_slot+=1; continue
        px=get_open(v["code"],d)
        if px is None:
          skips_slot+=1; continue
        cash += px*LOT
        pnl=(px/v["entry"]-1)*100
        done.append((v,pnl,"rotation"))
        pos.remove(v)
        rotations.append((d,v["code"],v["score"],v["ma"],pnl,x["code"],x["score"],x["ma"]))
        rotated=True
      cost=x["entry"]*LOT
      if cost>cash+1e-9:
        skips_cash+=1; continue
      cash-=cost; pos.append(x); buys+=1
    # cost-basis realized-equity approximation
    eq=cash+sum(p["entry"]*LOT for p in pos)
    peak=max(peak,eq)
    dd=(eq/peak-1)*100
    maxdd=min(maxdd,dd)
  # close leftovers at cached exits
  for p in pos:
    cash+=p["exit"]*LOT; done.append((p,(p["exit"]/p["entry"]-1)*100,"normal"))
  gains=sum(p["entry"]*LOT*(pct/100) for p,pct,_ in done if pct>0)
  losses=-sum(p["entry"]*LOT*(pct/100) for p,pct,_ in done if pct<0)
  years=(max(t["xd"] for t in trades)-min(t["ed"] for t in trades)).days/365.2425
  cagr=(cash/INITIAL)**(1/years)-1 if years>0 else 0
  return {"final":cash,"ret":(cash/INITIAL-1)*100,"cagr":cagr*100,"buys":buys,"rot":len(rotations),"slots":skips_slot,"cashskip":skips_cash,"dups":skips_dup,"pf":gains/losses if losses else None,"maxdd":maxdd},rotations

def main():
  tr=load_trades()
  modes=[("none","入れ替えなし"),("score","新候補のscoreが高い時だけ"),("score_ma","score高い or 同scoreでMA25に近い")]
  lines=["くいっと8枠 入れ替え比較","100万円 / 100株 / 最大8枠 / 現行スコア別gap / 1日最大1回入れ替え / 入替売却は当日始値 / 手数料税金なし",""]
  for mode,label in modes:
    s,rots=simulate(tr,mode)
    lines.append(f"{label}: 最終 {s['final']:,.0f}円 / 累積 {s['ret']:+.2f}% / CAGR {s['cagr']:+.2f}% / PF {s['pf']:.3f} / 買付{s['buys']} / 入替{s['rot']} / 枠見送り{s['slots']} / 資金見送り{s['cashskip']} / DD参考{s['maxdd']:.2f}%")
    if rots:
      pd.DataFrame(rots,columns=["date","sold_code","sold_score","sold_ma_gap","sold_pnl_pct","new_code","new_score","new_ma_gap"]).to_csv(f"output/kuitto_rotation_{mode}.csv",index=False,encoding="utf-8-sig")
  OUT.parent.mkdir(exist_ok=True)
  OUT.write_text("\n".join(lines),encoding="utf-8")
  print("\n".join(lines))
if __name__=="__main__": main()