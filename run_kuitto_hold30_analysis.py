import glob, json, os
import pandas as pd

SL=5.0
GAP_MAX=0.5
HOLDS=[15,20,25,30]


def load_prices():
    frames=[]
    for path in glob.glob('docs/prices/*.json'):
        code=os.path.basename(path).replace('.json','')
        if code.startswith('_'): continue
        try: rows=json.load(open(path,encoding='utf-8'))
        except Exception: continue
        if not rows: continue
        df=pd.DataFrame(rows)
        if not {'t','o','h','l','c'}.issubset(df.columns): continue
        ren={'t':'Date','o':'O','h':'H','l':'L','c':'C','v':'Vo'}
        df=df.rename(columns={k:v for k,v in ren.items() if k in df.columns})
        if 'Vo' not in df.columns: df['Vo']=pd.NA
        df['Code']=code
        frames.append(df[['Code','Date','O','H','L','C','Vo']])
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()


def entry_signal(g,i):
    r=g.iloc[i]
    if pd.isna(r.MA5) or pd.isna(r.MA25) or r.O<=0: return False
    bull=(r.C/r.O-1)*100
    if not (1.5 <= bull <= 3.5): return False
    gap=(r.MA5/r.MA25-1)*100
    if not (-5.0 <= gap <= 0.0): return False
    base=i-1; past=base-5
    if past<0 or pd.isna(g.MA5.iloc[past]) or not g.MA5.iloc[past]>0: return False
    decline=(g.MA5.iloc[base]/g.MA5.iloc[past]-1)*100
    if not (-5.5 <= decline <= -2.0): return False
    if any(g.MA5.iloc[j] > g.MA5.iloc[j-1] for j in range(i-2,i)): return False
    if not g.MA5.iloc[i] > g.MA5.iloc[i-1]: return False
    if (r.C/r.MA5-1)*100 > 4.0: return False
    if i<1 or pd.isna(r.Vo) or pd.isna(g.Vo.iloc[i-1]) or not float(g.Vo.iloc[i-1])>0: return False
    return float(r.Vo)/float(g.Vo.iloc[i-1]) >= 1.25


def strict_gakutto(g,j):
    if j<3: return False
    if pd.isna(g.MA5.iloc[j]) or pd.isna(g.MA5.iloc[j-1]): return False
    if not g.MA5.iloc[j] < g.MA5.iloc[j-1]: return False
    if g.MA5.iloc[j-1] < g.MA5.iloc[j-2]: return False
    if g.MA5.iloc[j-2] < g.MA5.iloc[j-3]: return False
    return float(g.C.iloc[j]) < float(g.O.iloc[j])


def simulate(g,i,hold_days):
    ent=i+1
    if ent>=len(g): return None
    prev_close=float(g.C.iloc[i]); entry=float(g.O.iloc[ent])
    if not prev_close>0 or not entry>0: return None
    entry_gap=(entry/prev_close-1)*100
    if entry_gap>GAP_MAX: return None
    sl=entry*(1-SL/100)
    end=min(ent+hold_days-1,len(g)-1)
    pending=False
    for j in range(ent,end+1):
        if pending:
            px=float(g.O.iloc[j])
            return (px-entry)/entry*100,'gakutto_strict_next_open',j-ent+1,entry_gap
        if float(g.L.iloc[j])<=sl:
            return -SL,'stop_loss',j-ent+1,entry_gap
        if j<end and strict_gakutto(g,j):
            pending=True
    if end+1 < len(g):
        px=float(g.O.iloc[end+1])
        return (px-entry)/entry*100,'time_exit_next_open',end-ent+2,entry_gap
    px=float(g.C.iloc[end])
    return (px-entry)/entry*100,'time_exit',end-ent+1,entry_gap


def stats(rows):
    df=pd.DataFrame(rows)
    if df.empty: return {'count':0}
    s=df.pnl.astype(float); pos=s[s>0]; neg=s[s<=0]; gl=-neg.sum(); pf=pos.sum()/gl if gl>0 else None
    reason_avg_hold={k:round(float(v),2) for k,v in df.groupby('reason')['hold'].mean().to_dict().items()}
    reason_median_hold={k:round(float(v),2) for k,v in df.groupby('reason')['hold'].median().to_dict().items()}
    return {'count':len(df),'win_rate':round((s>0).mean()*100,2),'avg_pnl':round(s.mean(),3),
            'median_pnl':round(s.median(),3),'pf':round(pf,3) if pf is not None else None,
            'avg_hold':round(df.hold.mean(),2),'reasons':df.reason.value_counts().to_dict(),
            'reason_avg_hold':reason_avg_hold,'reason_median_hold':reason_median_hold,
            'date_min':str(df.date.min()),'date_max':str(df.date.max())}


def three_way(rows):
    df=pd.DataFrame(rows)
    if df.empty: return []
    df['date_dt']=pd.to_datetime(df.date)
    dates=sorted(df.date_dt.dropna().unique())
    cut1=dates[len(dates)//3]; cut2=dates[(2*len(dates))//3]
    return [stats(df[df.date_dt<cut1]),stats(df[(df.date_dt>=cut1)&(df.date_dt<cut2)]),stats(df[df.date_dt>=cut2])]


def main():
    prices=load_prices(); out={h:[] for h in HOLDS}
    for code,g in prices.groupby('Code'):
        g=g.sort_values('Date').reset_index(drop=True).copy()
        for c in ['O','H','L','C','Vo']: g[c]=pd.to_numeric(g[c],errors='coerce')
        g['MA5']=g.C.rolling(5).mean(); g['MA25']=g.C.rolling(25).mean()
        for i in range(25,len(g)-1):
            if not entry_signal(g,i): continue
            for h in HOLDS:
                sim=simulate(g,i,h)
                if not sim: continue
                pnl,reason,hold,gap=sim
                out[h].append({'code':code,'date':str(g.Date.iloc[i]),'pnl':pnl,'reason':reason,'hold':hold,'gap':gap})
    result={'rule':'entry signal + volume>=1.25 + next-open gap<=+0.5% + SL -5% + strict gakutto next-open',
            'holds':{str(h):{'all':stats(rows),'three_way':three_way(rows)} for h,rows in out.items()}}
    print('KUITTO_HOLD30='+json.dumps(result,ensure_ascii=False))
    os.makedirs('output',exist_ok=True)
    json.dump(result,open('output/kuitto_hold30_analysis.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)

if __name__=='__main__': main()
