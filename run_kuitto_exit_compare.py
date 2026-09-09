import glob, json, os
import pandas as pd

SL=3.0
TP=5.0
HOLD=10


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
    close_ma5=(r.C/r.MA5-1)*100
    if close_ma5 > 4.0: return False
    if i<1 or pd.isna(r.Vo) or pd.isna(g.Vo.iloc[i-1]) or not float(g.Vo.iloc[i-1])>0: return False
    return float(r.Vo)/float(g.Vo.iloc[i-1]) >= 1.25


def first_down(g,j):
    return j>=1 and pd.notna(g.MA5.iloc[j]) and pd.notna(g.MA5.iloc[j-1]) and g.MA5.iloc[j] < g.MA5.iloc[j-1]


def strict_gakutto(g,j):
    if j<3 or not first_down(g,j): return False
    if g.MA5.iloc[j-1] < g.MA5.iloc[j-2]: return False
    if g.MA5.iloc[j-2] < g.MA5.iloc[j-3]: return False
    return float(g.C.iloc[j]) < float(g.O.iloc[j])


def simulate(g,i,mode):
    ent=i+1
    if ent>=len(g): return None
    entry=float(g.O.iloc[ent])
    if not entry>0: return None
    sl=entry*(1-SL/100); tp=entry*(1+TP/100)
    end=min(ent+HOLD-1,len(g)-1)
    pending_exit=None

    for j in range(ent,end+1):
        # 前日引けでがくっとを確認した場合は、この日の始値で手仕舞い。
        if pending_exit is not None:
            px=float(g.O.iloc[j])
            return (px-entry)/entry*100,pending_exit,j-ent+1

        lo=float(g.L.iloc[j]); hi=float(g.H.iloc[j])
        if lo<=sl:
            return -SL,'stop_loss',j-ent+1
        if mode in ('fixed5','hybrid_firstdown','hybrid_strict') and hi>=tp:
            return TP,'take_profit',j-ent+1

        # 最終保有日は引けでタイムアウトするので、翌朝売却予約はしない。
        if j < end:
            if mode in ('firstdown','hybrid_firstdown') and first_down(g,j):
                pending_exit='gakutto_firstdown_next_open'
            if mode in ('strict','hybrid_strict') and strict_gakutto(g,j):
                pending_exit='gakutto_strict_next_open'

    close=float(g.C.iloc[end])
    return (close-entry)/entry*100,'time_exit',end-ent+1


def stats(rows):
    df=pd.DataFrame(rows)
    if df.empty: return {'count':0}
    s=df.pnl.astype(float); pos=s[s>0]; neg=s[s<=0]; gl=-neg.sum(); pf=pos.sum()/gl if gl>0 else None
    return {
        'count':len(df), 'win_rate':round((s>0).mean()*100,2), 'avg_pnl':round(s.mean(),3),
        'median_pnl':round(s.median(),3), 'pf':round(pf,3) if pf is not None else None,
        'avg_hold':round(df.hold.mean(),2), 'reasons':df.reason.value_counts().to_dict()
    }


def main():
    prices=load_prices()
    modes=['fixed5','firstdown','strict','hybrid_firstdown','hybrid_strict']
    out={m:[] for m in modes}
    signal_count=0
    for code,g in prices.groupby('Code'):
        g=g.sort_values('Date').reset_index(drop=True).copy()
        for c in ['O','H','L','C','Vo']: g[c]=pd.to_numeric(g[c],errors='coerce')
        g['MA5']=g.C.rolling(5).mean(); g['MA25']=g.C.rolling(25).mean()
        for i in range(25,len(g)-1):
            if not entry_signal(g,i): continue
            signal_count+=1
            for m in modes:
                sim=simulate(g,i,m)
                if not sim: continue
                pnl,reason,hold=sim
                out[m].append({'code':code,'date':str(g.Date.iloc[i]),'pnl':pnl,'reason':reason,'hold':hold})
    result={'signals':signal_count,'modes':{m:stats(rows) for m,rows in out.items()}}
    print('KUITTO_EXIT_COMPARE='+json.dumps(result,ensure_ascii=False))
    os.makedirs('output',exist_ok=True)
    json.dump(result,open('output/kuitto_exit_compare.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)

if __name__=='__main__': main()
