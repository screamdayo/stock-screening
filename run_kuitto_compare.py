import json, os, glob
import pandas as pd

TP=5.0; SL=3.0; HOLD=10


def load_prices():
    frames=[]
    for path in glob.glob('docs/prices/*.json'):
        code=os.path.basename(path).replace('.json','')
        if code.startswith('_'): continue
        try:
            rows=json.load(open(path,encoding='utf-8'))
        except Exception:
            continue
        if not rows: continue
        df=pd.DataFrame(rows)
        if not {'t','o','h','l','c'}.issubset(df.columns): continue
        df=df.rename(columns={'t':'Date','o':'O','h':'H','l':'L','c':'C'})
        df['Code']=code
        frames.append(df[['Code','Date','O','H','L','C']])
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()


def base_signal(g,i):
    r=g.iloc[i]
    if pd.isna(r.MA5) or pd.isna(r.MA25) or r.O<=0: return None
    bull=(r.C/r.O-1)*100
    if bull < 1.5: return None
    if r.MA5 > r.MA25: return None
    base=i-1; past=base-5
    if past<0 or pd.isna(g.MA5.iloc[past]) or g.MA5.iloc[past]==0: return None
    decline=(g.MA5.iloc[base]/g.MA5.iloc[past]-1)*100
    if decline > -0.5: return None
    if any(g.MA5.iloc[j] > g.MA5.iloc[j-1] for j in range(i-2,i)): return None
    if not (g.MA5.iloc[i] > g.MA5.iloc[i-1]): return None
    return {
        'bull':bull,
        'decline':decline,
        'ma5_gap':(r.MA5/r.MA25-1)*100 if r.MA25 else None,
        'close_ma5':(r.C/r.MA5-1)*100 if r.MA5 else None,
    }


def simulate(g,i):
    ent=i+1; entry=float(g.iloc[ent].O)
    if not entry>0: return None
    tp=entry*(1+TP/100); sl=entry*(1-SL/100)
    end=min(ent+HOLD-1,len(g)-1)
    for j in range(ent,end+1):
        rr=g.iloc[j]
        if float(rr.L)<=sl: return -SL,'stop_loss'
        if float(rr.H)>=tp: return TP,'take_profit'
    pnl=(float(g.iloc[end].C)-entry)/entry*100
    return pnl,'time_exit'


def stats(rows):
    if not rows: return {'count':0}
    df=pd.DataFrame(rows); s=df.pnl.astype(float); w=s[s>0]; l=s[s<=0]; gl=-l.sum(); pf=w.sum()/gl if gl>0 else None
    return {
        'count':len(df),'win_rate':round((s>0).mean()*100,2),'avg_pnl':round(s.mean(),3),
        'median_pnl':round(s.median(),3),'pf':round(pf,3) if pf is not None else None,
        'tp':int((df.reason=='take_profit').sum()),'sl':int((df.reason=='stop_loss').sum()),'time':int((df.reason=='time_exit').sum())
    }


def main():
    prices=load_prices(); buckets={'base':[],'wide':[],'narrow':[]}
    for code,g in prices.groupby('Code'):
        g=g.sort_values('Date').reset_index(drop=True).copy()
        for c in ['O','H','L','C']: g[c]=pd.to_numeric(g[c],errors='coerce')
        g['MA5']=g.C.rolling(5).mean(); g['MA25']=g.C.rolling(25).mean()
        for i in range(25,len(g)-1):
            f=base_signal(g,i)
            if not f: continue
            sim=simulate(g,i)
            if not sim: continue
            pnl,reason=sim; rec={'code':code,'date':str(g.iloc[i].Date),'pnl':pnl,'reason':reason}
            buckets['base'].append(rec)
            # wide: decline -5.5..-2, ma5 gap -5..0, close vs ma5 <=4, bull 1.5..3.5
            if -5.5 <= f['decline'] <= -2.0 and -5.0 <= f['ma5_gap'] <= 0 and f['close_ma5'] <= 4.0 and 1.5 <= f['bull'] <= 3.5:
                buckets['wide'].append(rec)
            # narrow: strongest reverse-analysis zones combined
            if -3.428 <= f['decline'] <= -2.112 and -1.575 <= f['ma5_gap'] <= 0 and f['close_ma5'] <= 1.587 and 1.5 <= f['bull'] <= 1.911:
                buckets['narrow'].append(rec)
    out={k:stats(v) for k,v in buckets.items()}
    out['period']={'min':str(prices.Date.min()),'max':str(prices.Date.max())}
    print('KUITTO_COMPARE='+json.dumps(out,ensure_ascii=False))
    os.makedirs('output',exist_ok=True)
    json.dump(out,open('output/kuitto_compare.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)

if __name__=='__main__': main()
