import csv, math
from pathlib import Path
import pandas as pd
import config, download

SRC=Path("data/kuitto_score_trades.csv")
OUT=Path("output/kuitto_rotation_yearly.txt")
INITIAL=1_000_000.0; LOT=100; MAX_POS=8
CAP={0:None,1:0.0,2:0.25,3:0.0,4:1.0}
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
      out.append({"code":str(r["code"]),"score":s,"entry":entry,"exit":exitp,"ma":ma,"ed":ed,"xd":xd})
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

def stronger(n,o,mode):
  if n["score"]>o["score"]: return True
  return mode=="score_ma" and n["score"]==o["score"] and abs(n["ma"])<abs(o["ma"])-1e-12
def weakest(pos): return min(pos,key=lambda p:(p["score"],-abs(p["ma"]),p["ed"],p["code"]))

def simulate(trades,mode):
  by={}; dates=set()
  for t in trades: by.setdefault(t["ed"],[]).append(t); dates.update([t["ed"],t["xd"]])
  cash=INITIAL; pos=[]; done=[]; rotations=0
  for d in sorted(dates):
    rem=[]
    for p in pos:
      if p["xd"]==d:
        pnl=(p["exit"]/p["entry"]-1)*100; cash+=p["exit"]*LOT; done.append(pnl)
      else: rem.append(p)
    pos=rem
    rotated=False
    for x in sorted(by.get(d,[]),key=lambda z:(-z["score"],abs(z["ma"]),z["code"])):
      if any(p["code"]==x["code"] for p in pos): continue
      if len(pos)>=MAX_POS:
        if mode=="none" or rotated: continue
        v=weakest(pos)
        if not stronger(x,v,mode): continue
        px=get_open(v["code"],d)
        if px is None: continue
        pnl=(px/v["entry"]-1)*100; cash+=px*LOT; done.append(pnl); pos.remove(v); rotations+=1; rotated=True
      cost=x["entry"]*LOT
      if cost>cash+1e-9: continue
      cash-=cost; pos.append(x)
  for p in pos:
    cash+=p["exit"]*LOT; done.append((p["exit"]/p["entry"]-1)*100)
  gp=sum(v for v in done if v>0); gl=-sum(v for v in done if v<0)
  return {"final":cash,"ret":(cash/INITIAL-1)*100,"pf":gp/gl if gl else None,"rot":rotations,"n":len(done)}

def main():
  alltr=load_trades()
  lines=["くいっと8枠 入れ替え年別比較","各年を100万円から独立スタート / 100株 / 最大8枠 / 1日最大1回入替 / 入替売却は当日始値",""]
  for y in sorted({t["ed"].year for t in alltr}):
    # entry yearで年別を切り、年をまたぐ保有はそのトレードの実出口まで含める
    tr=[t for t in alltr if t["ed"].year==y]
    if not tr: continue
    lines.append(str(y))
    for mode,label in [("none","入替なし"),("score","高scoreのみ"),("score_ma","同score+MA25も")]:
      s=simulate(tr,mode)
      lines.append(f"{label}: 最終 {s['final']:,.0f}円 / return {s['ret']:+.2f}% / PF {s['pf']:.3f} / 決済{s['n']} / 入替{s['rot']}")
    lines.append("")
  OUT.parent.mkdir(exist_ok=True); OUT.write_text("\n".join(lines),encoding="utf-8"); print("\n".join(lines))
if __name__=="__main__": main()