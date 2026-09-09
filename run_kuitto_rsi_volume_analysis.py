import glob, json, os
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
        ren={'t':'Date','o':'O','h':'H','l':'L','c':'C','v':'Vo'}
        df=df.rename(columns={k:v for k,v in ren.items() if k in df.columns})
        if 'Vo' not in df.columns:
            df['Vo']=pd.NA
        df['Code']=code
        frames.append(df[['Code','Date','O','H','L','C','Vo']])
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()


def add_rsi(g, period=14):
    d=g['C'].diff()
    gain=d.clip(lower=0)
    loss=-d.clip(upper=0)
    avg_gain=gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss=loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs=avg_gain/avg_loss.replace(0,pd.NA)
    g['RSI14']=100-(100/(1+rs))
    g.loc[(avg_loss==0)&(avg_gain>0),'RSI14']=100.0
    g.loc[(avg_loss==0)&(avg_gain==0),'RSI14']=50.0
    return g


def signal_features(g,i):
    r=g.iloc[i]
    if pd.isna(r.MA5) or pd.isna(r.MA25) or r.O<=0: return None
    bull=(r.C/r.O-1)*100
    if not (1.5 <= bull <= 3.5): return None
    if not r.MA25>0: return None
    gap=(r.MA5/r.MA25-1)*100
    if not (-5.0 <= gap <= 0.0): return None
    base=i-1; past=base-5
    if past<0 or pd.isna(g.MA5.iloc[past]) or not g.MA5.iloc[past]>0: return None
    decline=(g.MA5.iloc[base]/g.MA5.iloc[past]-1)*100
    if not (-5.5 <= decline <= -2.0): return None
    if any(g.MA5.iloc[j] > g.MA5.iloc[j-1] for j in range(i-2,i)): return None
    if not g.MA5.iloc[i] > g.MA5.iloc[i-1]: return None
    if not r.MA5>0: return None
    close_ma5=(r.C/r.MA5-1)*100
    if close_ma5 > 4.0: return None
    vr=None
    if i>=1 and pd.notna(r.Vo) and pd.notna(g.Vo.iloc[i-1]) and float(g.Vo.iloc[i-1])>0:
        vr=float(r.Vo)/float(g.Vo.iloc[i-1])
    return {'rsi14': None if pd.isna(r.RSI14) else float(r.RSI14),'volume_ratio':vr}


def simulate(g,i):
    ent=i+1
    if ent>=len(g): return None
    entry=float(g.iloc[ent].O)
    if not entry>0: return None
    tp=entry*(1+TP/100); sl=entry*(1-SL/100)
    end=min(ent+HOLD-1,len(g)-1)
    for j in range(ent,end+1):
        rr=g.iloc[j]
        if float(rr.L)<=sl: return -SL,'stop_loss'
        if float(rr.H)>=tp: return TP,'take_profit'
    pnl=(float(g.iloc[end].C)-entry)/entry*100
    return pnl,'time_exit'


def qstats(df,col):
    x=df[[col,'pnl']].dropna().copy()
    if x.empty: return []
    x['bin']=pd.qcut(x[col],4,duplicates='drop')
    out=[]
    for b,g in x.groupby('bin',observed=True):
        s=g.pnl.astype(float); w=s[s>0]; l=s[s<=0]; gl=-l.sum(); pf=w.sum()/gl if gl>0 else None
        out.append({'bin':str(b),'count':len(g),'win_rate':round((s>0).mean()*100,2),'avg_pnl':round(s.mean(),3),'median_pnl':round(s.median(),3),'pf':round(pf,3) if pf is not None else None})
    return out


def threshold_stats(df,col,thresholds,op):
    out=[]
    for t in thresholds:
        if op=='ge': g=df[df[col]>=t]
        elif op=='le': g=df[df[col]<=t]
        else: continue
        if g.empty: continue
        s=g.pnl.astype(float); w=s[s>0]; l=s[s<=0]; gl=-l.sum(); pf=w.sum()/gl if gl>0 else None
        out.append({'threshold':t,'op':op,'count':len(g),'win_rate':round((s>0).mean()*100,2),'avg_pnl':round(s.mean(),3),'pf':round(pf,3) if pf is not None else None})
    return out


def main():
    prices=load_prices(); rows=[]
    for code,g in prices.groupby('Code'):
        g=g.sort_values('Date').reset_index(drop=True).copy()
        for c in ['O','H','L','C','Vo']: g[c]=pd.to_numeric(g[c],errors='coerce')
        g['MA5']=g.C.rolling(5).mean(); g['MA25']=g.C.rolling(25).mean(); g=add_rsi(g)
        for i in range(25,len(g)-1):
            f=signal_features(g,i)
            if not f: continue
            sim=simulate(g,i)
            if not sim: continue
            pnl,reason=sim
            rows.append({'code':code,'date':str(g.iloc[i].Date),'pnl':pnl,'reason':reason,**f})
    df=pd.DataFrame(rows)
    s=df.pnl.astype(float); w=s[s>0]; l=s[s<=0]; gl=-l.sum(); pf=w.sum()/gl if gl>0 else None
    base={'count':len(df),'win_rate':round((s>0).mean()*100,2),'avg_pnl':round(s.mean(),3),'pf':round(pf,3) if pf is not None else None}
    analysis={
        'base':base,
        'rsi_quartiles':qstats(df,'rsi14'),
        'volume_quartiles':qstats(df,'volume_ratio'),
        'rsi_thresholds_ge':threshold_stats(df,'rsi14',[30,35,40,45,50,55,60],'ge'),
        'rsi_thresholds_le':threshold_stats(df,'rsi14',[40,45,50,55,60,65,70],'le'),
        'volume_thresholds_ge':threshold_stats(df,'volume_ratio',[0.5,0.75,1.0,1.25,1.5,2.0],'ge'),
        'non_null':{'rsi':int(df.rsi14.notna().sum()),'volume':int(df.volume_ratio.notna().sum())},
    }
    print('KUITTO_RSI_VOLUME='+json.dumps(analysis,ensure_ascii=False))
    os.makedirs('output',exist_ok=True)
    df.to_csv('output/kuitto_rsi_volume_trades.csv',index=False)
    json.dump(analysis,open('output/kuitto_rsi_volume_analysis.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)

if __name__=='__main__': main()
