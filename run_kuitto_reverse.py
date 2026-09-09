import json, os, glob
import pandas as pd

TP_PCT=5.0; SL_PCT=3.0; HOLD_DAYS=10
DECLINE_LOOKBACK=5; MIN_DECLINE=-0.5; TURN_LOOKBACK=2; MIN_BULL=1.5

def load_prices():
    frames=[]
    for path in glob.glob('docs/prices/*.json'):
        code=os.path.basename(path).replace('.json','')
        if code.startswith('_'): continue
        try:
            with open(path,encoding='utf-8') as f: rows=json.load(f)
        except Exception:
            continue
        if not rows: continue
        df=pd.DataFrame(rows)
        if not {'t','o','h','l','c'}.issubset(df.columns): continue
        df=df.rename(columns={'t':'Date','o':'O','h':'H','l':'L','c':'C'})
        df['Code']=code
        frames.append(df[['Code','Date','O','H','L','C']])
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

def summarize_group(df):
    if df.empty: return {'count':0}
    out={'count':len(df)}
    cols=['bull_candle_pct','ma5_prior5d_decline_pct','ma25_slope_pct','ma5_vs_ma25_pct','close_vs_ma25_pct','close_vs_ma5_pct','prior_5d_return_pct']
    for c in cols:
        s=pd.to_numeric(df[c],errors='coerce').dropna()
        out[c]={'mean':round(s.mean(),3),'median':round(s.median(),3)} if len(s) else {'mean':None,'median':None}
    return out

def qstats(df, col):
    x=df[[col,'profit_pct']].dropna().copy()
    if x.empty: return []
    try:
        x['bin']=pd.qcut(x[col],4,duplicates='drop')
    except Exception:
        return []
    rows=[]
    for b,g in x.groupby('bin',observed=True):
        p=g['profit_pct'].astype(float); w=p[p>0]; l=p[p<=0]; gl=-l.sum(); pf=(w.sum()/gl) if gl>0 else None
        rows.append({'bin':str(b),'count':len(g),'win_rate':round((p>0).mean()*100,2),'avg_pnl':round(p.mean(),3),'pf':round(pf,3) if pf is not None else None})
    return rows

def main():
    prices=load_prices(); trades=[]; signals=0
    for code,g in prices.groupby('Code'):
        g=g.sort_values('Date').reset_index(drop=True).copy()
        for c in ['O','H','L','C']: g[c]=pd.to_numeric(g[c],errors='coerce')
        g['MA5']=g['C'].rolling(5).mean(); g['MA25']=g['C'].rolling(25).mean()
        for i in range(25,len(g)-1):
            r=g.iloc[i]
            if pd.isna(r['MA5']) or pd.isna(r['MA25']) or r['O']<=0: continue
            bull=(r['C']/r['O']-1)*100
            if bull < MIN_BULL: continue
            if r['MA5'] > r['MA25']: continue
            base=i-1; past=base-DECLINE_LOOKBACK
            if past<0 or pd.isna(g['MA5'].iloc[past]) or g['MA5'].iloc[past]==0: continue
            decline=(g['MA5'].iloc[base]/g['MA5'].iloc[past]-1)*100
            if decline > MIN_DECLINE: continue
            if any(g['MA5'].iloc[j] > g['MA5'].iloc[j-1] for j in range(i-TURN_LOOKBACK,i)): continue
            if not (g['MA5'].iloc[i] > g['MA5'].iloc[i-1]): continue
            signals+=1
            ma25_slope=(g['MA25'].iloc[i]/g['MA25'].iloc[i-1]-1)*100 if g['MA25'].iloc[i-1] else None
            ma5_vs_ma25=(r['MA5']/r['MA25']-1)*100 if r['MA25'] else None
            close_vs_ma25=(r['C']/r['MA25']-1)*100 if r['MA25'] else None
            close_vs_ma5=(r['C']/r['MA5']-1)*100 if r['MA5'] else None
            prior5=(r['C']/g['C'].iloc[i-5]-1)*100 if i>=5 and g['C'].iloc[i-5] else None
            ent=i+1; entry=float(g.iloc[ent]['O'])
            if not entry>0: continue
            tp=entry*(1+TP_PCT/100); sl=entry*(1-SL_PCT/100)
            end=min(ent+HOLD_DAYS-1,len(g)-1); exitp=None; reason=None
            for j in range(ent,end+1):
                rr=g.iloc[j]; hit_tp=float(rr['H'])>=tp; hit_sl=float(rr['L'])<=sl
                if hit_sl: exitp=sl; reason='stop_loss'; break
                if hit_tp: exitp=tp; reason='take_profit'; break
            if exitp is None: exitp=float(g.iloc[end]['C']); reason='time_exit'
            pnl=(exitp-entry)/entry*100
            trades.append({'code':code,'signal_date':str(r['Date']),'profit_pct':round(pnl,3),'exit_reason':reason,'bull_candle_pct':bull,'ma5_prior5d_decline_pct':decline,'ma25_slope_pct':ma25_slope,'ma5_vs_ma25_pct':ma5_vs_ma25,'close_vs_ma25_pct':close_vs_ma25,'close_vs_ma5_pct':close_vs_ma5,'prior_5d_return_pct':prior5})
    df=pd.DataFrame(trades); os.makedirs('output',exist_ok=True)
    s=df['profit_pct'].astype(float); w=s[s>0]; l=s[s<=0]; gl=-l.sum(); pf=(w.sum()/gl) if gl>0 else None
    summary={'signals':signals,'total_trades':len(df),'win_rate':round((s>0).mean()*100,2),'avg_profit_pct_all':round(s.mean(),3),'median_profit_pct':round(s.median(),3),'profit_factor':round(pf,3) if pf is not None else None,'take_profit_count':int((df.exit_reason=='take_profit').sum()),'stop_loss_count':int((df.exit_reason=='stop_loss').sum()),'time_exit_count':int((df.exit_reason=='time_exit').sum()),'date_min':str(prices['Date'].min()),'date_max':str(prices['Date'].max())}
    groups={'take_profit':summarize_group(df[df.exit_reason=='take_profit']),'stop_loss':summarize_group(df[df.exit_reason=='stop_loss']),'all_wins':summarize_group(df[df.profit_pct>0]),'all_losses':summarize_group(df[df.profit_pct<=0])}
    cols=['bull_candle_pct','ma5_prior5d_decline_pct','ma25_slope_pct','ma5_vs_ma25_pct','close_vs_ma25_pct','close_vs_ma5_pct','prior_5d_return_pct']
    analysis={'groups':groups,'quartiles':{c:qstats(df,c) for c in cols}}
    df.to_csv('output/kuitto_reverse_trades.csv',index=False)
    with open('output/kuitto_reverse_summary.json','w',encoding='utf-8') as f: json.dump(summary,f,ensure_ascii=False,indent=2)
    with open('output/kuitto_reverse_analysis.json','w',encoding='utf-8') as f: json.dump(analysis,f,ensure_ascii=False,indent=2)
    print('KUITTO_REVERSE_SUMMARY='+json.dumps(summary,ensure_ascii=False))
    print('KUITTO_REVERSE_ANALYSIS='+json.dumps(analysis,ensure_ascii=False))

if __name__=='__main__': main()
