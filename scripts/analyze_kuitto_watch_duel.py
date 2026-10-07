import json
from pathlib import Path
import numpy as np
import pandas as pd

B=Path("data/batches")
def load():
    fs=sorted(B.glob("batch_*.parquet")); a=[]
    for p in fs:
        d=pd.read_parquet(p)
        cols=[c for c in ["Code","Date","O","H","L","C","Vo","AdjO","AdjH","AdjL","AdjC","AdjVo","ArchiveMarket"] if c in d]
        a.append(d[cols])
    d=pd.concat(a,ignore_index=True)
    if "ArchiveMarket" in d: d=d[d.ArchiveMarket=="プライム"].copy()
    for x,y in [("O","AdjO"),("H","AdjH"),("L","AdjL"),("C","AdjC"),("Vo","AdjVo")]:
        if y in d: d[x]=d[y].where(d[y].notna(),d.get(x))
    for c in ["O","H","L","C","Vo"]: d[c]=pd.to_numeric(d[c],errors="coerce")
    d["Code"]=d.Code.astype(str); d["Date"]=pd.to_datetime(d.Date)
    return d.drop_duplicates(["Code","Date"]).sort_values(["Code","Date"])

def prep(g):
    g=g.dropna(subset=["O","H","L","C"]).copy().reset_index(drop=True)
    g["MA5"]=g.C.rolling(5).mean(); g["MA25"]=g.C.rolling(25).mean()
    g["ATR14"]=pd.concat([(g.H-g.L),(g.H-g.C.shift()).abs(),(g.L-g.C.shift()).abs()],axis=1).max(axis=1).rolling(14).mean()/g.C*100
    g["DD20"]=g.C/g.C.rolling(20).max()*100-100\n    g["MA25S5"]=(g.MA25/g.MA25.shift(5)-1)*100\n    g["RET20"]=(g.C/g.C.shift(20)-1)*100
    return g

def common(g,i):
    r=g.iloc[i]; bull=(r.C/r.O-1)*100
    gap=(r.MA5/r.MA25-1)*100
    dec=(g.MA5.iloc[i-1]/g.MA5.iloc[i-6]-1)*100
    return 1.5<=bull<=3.5 and -5<=gap<=0 and -5.5<=dec<=-2 and (r.C/r.MA5-1)*100<=4 and r.ATR14>=3.4 and r.DD20<=-5.5

def normal(g,i):
    if i<25 or not common(g,i): return False
    return all(g.MA5.iloc[j]<=g.MA5.iloc[j-1] for j in [i-2,i-1]) and g.MA5.iloc[i]>g.MA5.iloc[i-1] and g.Vo.iloc[i]/g.Vo.iloc[i-1]>=1.25

# watch seed: same quality setup and strong candle, but MA5 has NOT turned up yet.
# Confirm within 1-2 sessions when MA5 first turns up. No confirmation-volume requirement.
def seed(g,i):
    if i<25 or not common(g,i): return False
    return all(g.MA5.iloc[j]<=g.MA5.iloc[j-1] for j in [i-2,i-1,i])

def metric(rows):
    if not rows:return {"n":0}
    x=pd.Series([r["ret10"] for r in rows]); gp=x[x>0].sum(); gl=-x[x<0].sum()
    return {"n":len(x),"win":round((x>0).mean()*100,2),"avg":round(x.mean(),3),"med":round(x.median(),3),"pf":round(gp/gl,3) if gl else None,"p5":round((x>=5).mean()*100,2),"p10":round((x>=10).mean()*100,2)}

d=load(); out={"normal":[],"watch1":[],"watch2":[]}\nfeatures=[]
for code,z in d.groupby("Code"):
    g=prep(z)
    for i in range(25,len(g)-13):
        if normal(g,i):
            e=i+1; x=i+11
            out["normal"].append({"year":g.Date.iloc[i].year,"ret10":(g.O.iloc[x]/g.O.iloc[e]-1)*100})
        if seed(g,i):
            for lag,key in [(1,"watch1"),(2,"watch2")]:
                j=i+lag
                if j>=len(g)-11: continue
                if g.MA5.iloc[j]>g.MA5.iloc[j-1] and all(g.MA5.iloc[k]<=g.MA5.iloc[k-1] for k in range(i+1,j)):
                    e=j+1; x=e+10
                    ret=(g.O.iloc[x]/g.O.iloc[e]-1)*100\n                    out[key].append({"year":g.Date.iloc[j].year,"ret10":ret})\n                    if key=="watch1":\n                        features.append({"ret10":ret,"bull":(g.C.iloc[i]/g.O.iloc[i]-1)*100,"vol_seed":g.Vo.iloc[i]/g.Vo.iloc[i-1],"vol_confirm":g.Vo.iloc[j]/g.Vo.iloc[j-1],"ma5_slope_seed":(g.MA5.iloc[i]/g.MA5.iloc[i-1]-1)*100,"ma5_slope_confirm":(g.MA5.iloc[j]/g.MA5.iloc[j-1]-1)*100,"dd20":g.DD20.iloc[i],"ma25s5":g.MA25S5.iloc[i],"ret20":g.RET20.iloc[i],"close_ma5":(g.C.iloc[i]/g.MA5.iloc[i]-1)*100})
                    break
res={}
for k,v in out.items():
    res[k]={"all":metric(v),"yearly":{str(y):metric([r for r in v if r["year"]==y]) for y in sorted(set(r["year"] for r in v))}}
Path("results").mkdir(exist_ok=True); Path("results/kuitto_watch_duel.json").write_text(json.dumps(res,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(res,ensure_ascii=False,indent=2))
