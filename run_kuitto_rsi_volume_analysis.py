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
        if 'Vo' not in df.columns: df['Vo']=pd.NA
        df['Code']=code
        frames.append(df[['Code','Date','O','H','L','C','Vo']])
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()


def add_rsi(g, period=14):
    d=g['C'].diff(); gain=d.clip(lower=0); loss=-d.clip(upper=0)
    avg_gain=gain.ewm(alpha=1/period,adjust=False,min_periods=period).mean()
    avg_loss=loss.ewm(alpha=1/period,adjust=False,min_periods=period).mean()
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
    return {
        'rsi14':None if pd.isna(r.RSI14) else float(r.RSI14),
        'volume_ratio':vr,
        'bull_candle_pct':float(bull),
        'ma5_prior5d_decline_pct':float(decline),
        'ma5_vs_ma25_pct':float(gap),
        'close_vs_ma5_pct':float(close_ma5),
    }


def simulate(g,i):
    ent=i+1
    if ent>=len(g): return None
    entry=float(g.iloc[ent].O)
    if not entry>0: return None
    tp=entry*(1+TP/100); sl=entry*(1-SL/100); end=min(ent+HOLD-1,len(g)-1)
    for j in range(ent,end+1):
        rr=g.iloc[j]
        if float(rr.L)<=sl: return -SL,'stop_loss'
        if float(rr.H)>=tp: return TP,'take_profit'
    return (float(g.iloc[end].C)-entry)/entry*100,'time_exit'


def stats(g):
    if g.empty: return {'count':0}
    s=g.pnl.astype(float); w=s[s>0]; l=s[s<=0]; gl=-l.sum(); pf=w.sum()/gl if gl>0 else None
    return {'count':len(g),'win_rate':round((s>0).mean()*100,2),'avg_pnl':round(s.mean(),3),'median_pnl':round(s.median(),3),'pf':round(pf,3) if pf is not None else None,'date_min':str(g.date.min()),'date_max':str(g.date.max())}


def three_way_parts(df):
    x=df.copy(); x['date_dt']=pd.to_datetime(x['date']); dates=sorted(x.date_dt.dropna().unique())
    if len(dates)<3: return [x]
    cut1=dates[len(dates)//3]; cut2=dates[(2*len(dates))//3]
    return [x[x.date_dt<cut1],x[(x.date_dt>=cut1)&(x.date_dt<cut2)],x[x.date_dt>=cut2]]


def rank_day(g, method):
    x=g.copy()
    if method=='volume_desc':
        return x.sort_values(['volume_ratio','code'],ascending=[False,True])
    if method=='ma25_near':
        x['_score']=x.ma5_vs_ma25_pct.abs(); return x.sort_values(['_score','code'])
    if method=='close_ma5_low':
        return x.sort_values(['close_vs_ma5_pct','code'])
    if method=='decline_mid':
        x['_score']=(x.ma5_prior5d_decline_pct+2.75).abs(); return x.sort_values(['_score','code'])
    if method=='bull_small':
        return x.sort_values(['bull_candle_pct','code'])
    if method=='composite':
        # 日ごとの相対順位を均等加重。小さいほど上位。
        x['_r_vol']=x.volume_ratio.rank(ascending=False,method='average',pct=True)
        x['_r_gap']=x.ma5_vs_ma25_pct.abs().rank(ascending=True,method='average',pct=True)
        x['_r_close']=x.close_vs_ma5_pct.rank(ascending=True,method='average',pct=True)
        x['_r_decline']=(x.ma5_prior5d_decline_pct+2.75).abs().rank(ascending=True,method='average',pct=True)
        x['_r_bull']=x.bull_candle_pct.rank(ascending=True,method='average',pct=True)
        x['_score']=x[['_r_vol','_r_gap','_r_close','_r_decline','_r_bull']].mean(axis=1)
        return x.sort_values(['_score','code'])
    raise ValueError(method)


def select_topn(df,method,n):
    if n is None: return df.copy()
    picked=[]
    for _,g in df.groupby('date',sort=True):
        picked.append(rank_day(g,method).head(n))
    return pd.concat(picked,ignore_index=True) if picked else df.iloc[0:0].copy()


def multi_day_analysis(df):
    x=df[df.volume_ratio>=1.25].copy()
    counts=x.groupby('date').size()
    overview={'signals':len(x),'signal_days':int(counts.size),'days_ge2':int((counts>=2).sum()),'days_ge5':int((counts>=5).sum()),'days_ge10':int((counts>=10).sum()),'max_per_day':int(counts.max()) if not counts.empty else 0,'avg_per_signal_day':round(float(counts.mean()),2) if not counts.empty else 0}
    methods=['volume_desc','ma25_near','close_ma5_low','decline_mid','bull_small','composite']
    ns=[10,5,3,1]
    out={'overview':overview,'all':stats(x),'methods':{}}
    for method in methods:
        out['methods'][method]={}
        for n in ns:
            sel=select_topn(x,method,n)
            out['methods'][method][f'top{n}']=stats(sel)
        # 実運用候補としてtop3の時系列安定性も確認
        top3=select_topn(x,method,3)
        out['methods'][method]['top3_three_way']=[stats(p) for p in three_way_parts(top3)]
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
    analysis={'base':stats(df),'multi_day':multi_day_analysis(df)}
    print('KUITTO_MULTI_DAY='+json.dumps(analysis,ensure_ascii=False))
    os.makedirs('output',exist_ok=True)
    df.to_csv('output/kuitto_multi_day_trades.csv',index=False)
    json.dump(analysis,open('output/kuitto_multi_day_analysis.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)

if __name__=='__main__': main()
