import csv, math
from collections import defaultdict
from pathlib import Path
import pandas as pd
import download

SRC=Path("data/kuitto_score_trades.csv")
OUT=Path("output/kuitto_rotation_reason.txt")
LOT=100; MAX_POS=8
CAP={0:None,1:0.0,2:0.25,3:0.0,4:1.0}
OPEN_CACHE={}

def load_trades():
  out=[]
  with SRC.open(encoding="utf-8-sig",newline="") as f:
    for r in csv.DictReader(f):
      try:
        s=int(float(r["runner_score"])); gap=float(r["next_open_gap_pct"]); entry=float(r["entry_open"]); exitp=float(r["exit_price"]); ma=float(r["ma5_vs_ma25_pct"])
        atr=float(r["atr14_pct"]); dd=float(r["dd20_pct"]); bull=float(r["bull_candle_pct"]); vol=float(r["volume_ratio"])
        ed=pd.Timestamp(r["entry_date"]); xd=pd.Timestamp(r["exit_date"]); hold=float(r["hold_days"])
      except: continue
      if any(math.isnan(x) for x in (gap,entry,exitp,ma,atr,dd,bull,vol)): continue
      cap=CAP[s]
      if cap is not None and gap>cap: continue
      out.append({"code":str(r["code"]),"score":s,"gap":gap,"entry":entry,"exit":exitp,"ma":ma,"atr":atr,"dd":dd,"bull":bull,"vol":vol,"ed":ed,"xd":xd,"hold":hold,"reason":r.get("exit_reason","")})
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

def weakest(pos): return min(pos,key=lambda p:(p["score"],-abs(p["ma"]),p["ed"],p["code"]))

def build_rotations(trades):
  by=defaultdict(list); dates=set()
  for t in trades: by[t["ed"]].append(t); dates.update([t["ed"],t["xd"]])
  cash=1_000_000.0; pos=[]; rots=[]
  for d in sorted(dates):
    rem=[]
    for p in pos:
      if p["xd"]==d: cash += p["exit"]*LOT
      else: rem.append(p)
    pos=rem; rotated=False
    for x in sorted(by.get(d,[]),key=lambda z:(-z["score"],abs(z["ma"]),z["code"])):
      if any(p["code"]==x["code"] for p in pos): continue
      if len(pos)>=MAX_POS:
        if rotated: continue
        v=weakest(pos)
        if x["score"]<=v["score"]: continue
        px=get_open(v["code"],d)
        if px is None: continue
        cash += px*LOT; pos.remove(v); rotated=True
        old_future=(v["exit"]-px)*LOT
        new_future=(x["exit"]-x["entry"])*LOT
        rots.append({
          "date":d,"year":d.year,"old":v,"new":x,"old_rot_open":px,
          "old_unreal_pct":(px/v["entry"]-1)*100,
          "old_future_yen":old_future,"new_future_yen":new_future,
          "delta_yen":new_future-old_future,
          "score_jump":x["score"]-v["score"],
          "old_age_days":(d-v["ed"]).days,
        })
      cost=x["entry"]*LOT
      if cost>cash+1e-9: continue
      cash-=cost; pos.append(x)
  return rots

def summarize(rs,label):
  if not rs: return [label+": none"]
  wins=[r for r in rs if r["delta_yen"]>0]
  delta=sum(r["delta_yen"] for r in rs)
  return [f"{label}: n{len(rs)} / 入替勝ち {len(wins)/len(rs)*100:.1f}% / 差額合計 {delta:+,.0f}円 / 1回平均 {delta/len(rs):+,.0f}円"]

def main():
  tr=load_trades(); rs=build_rotations(tr)
  lines=["くいっと 高score入れ替え 理由分析","比較: 入替日に既存を売らず本来の出口まで持つ場合 vs 新候補を100株買って本来の出口まで持つ場合",""]
  for y in sorted({r["year"] for r in rs}): lines += summarize([r for r in rs if r["year"]==y],str(y))
  lines += ["", "score jump別"]
  for j in sorted({r["score_jump"] for r in rs}): lines += summarize([r for r in rs if r["score_jump"]==j],f"+{j}点")
  lines += ["","既存含み損益別"]
  bands=[(-999,-3,"既存<-3%"),(-3,0,"-3〜0%"),(0,3,"0〜+3%"),(3,999,"既存>+3%")]
  for lo,hi,name in bands: lines += summarize([r for r in rs if lo<=r["old_unreal_pct"]<hi],name)
  lines += ["","既存score→新score別"]
  pairs=sorted({(r["old"]["score"],r["new"]["score"]) for r in rs})
  for a,b in pairs: lines += summarize([r for r in rs if r["old"]["score"]==a and r["new"]["score"]==b],f"{a}→{b}")
  lines += ["","2022詳細"]
  r22=[r for r in rs if r["year"]==2022]
  for r in sorted(r22,key=lambda x:x["delta_yen"]):
    lines.append(f"{r['date'].date()} {r['old']['code']} score{r['old']['score']}→{r['new']['code']} score{r['new']['score']} / 既存含み {r['old_unreal_pct']:+.2f}% / 旧その後 {r['old_future_yen']:+,.0f}円 / 新その後 {r['new_future_yen']:+,.0f}円 / 差 {r['delta_yen']:+,.0f}円 / 旧出口 {r['old']['reason']} / 新出口 {r['new']['reason']}")
  OUT.parent.mkdir(exist_ok=True); OUT.write_text("\n".join(lines),encoding="utf-8"); print("\n".join(lines))
if __name__=="__main__": main()