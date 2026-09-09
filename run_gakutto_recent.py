import json, os, glob
import pandas as pd

TP_PCT=5.0; SL_PCT=3.0; HOLD_DAYS=10
UP_LOOKBACK=5; MIN_RISE=0.5; TURN_LOOKBACK=2; MIN_BEAR=1.5

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
        need={'t','o','h','l','c'}
        if not need.issubset(df.columns): continue
        df=df.rename(columns={'t':'Date','o':'O','h':'H','l':'L','c':'C'})
        df['Code']=code
        frames.append(df[['Code','Date','O','H','L','C']])
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

def detect_and_trade(prices):
    trades=[]; signals=0
    for code,g in prices.groupby('Code'):
        g=g.sort_values('Date').reset_index(drop=True).copy()
        for c in ['O','H','L','C']: g[c]=pd.to_numeric(g[c],errors='coerce')
        g['MA5']=g['C'].rolling(5).mean(); g['MA25']=g['C'].rolling(25).mean()
        for i in range(25,len(g)-1):
            r=g.iloc[i]
            if pd.isna(r['MA5']) or pd.isna(r['MA25']) or r['O']<=0: continue
            bear=(r['C']/r['O']-1)*100
            if bear > -MIN_BEAR: continue
            if r['MA5'] < r['MA25']: continue
            base=i-1; past=base-UP_LOOKBACK
            if past<0 or pd.isna(g['MA5'].iloc[past]) or g['MA5'].iloc[past]==0: continue
            rise=(g['MA5'].iloc[base]/g['MA5'].iloc[past]-1)*100
            if rise<MIN_RISE: continue
            if any(g['MA5'].iloc[j] < g['MA5'].iloc[j-1] for j in range(i-TURN_LOOKBACK,i)): continue
            if not (g['MA5'].iloc[i] < g['MA5'].iloc[i-1]): continue
            signals+=1
            ent=i+1; entry=float(g.iloc[ent]['O'])
            if not entry>0: continue
            tp=entry*(1-TP_PCT/100); sl=entry*(1+SL_PCT/100)
            end=min(ent+HOLD_DAYS-1,len(g)-1)
            exitp=None; reason=None
            for j in range(ent,end+1):
                rr=g.iloc[j]
                hit_tp=float(rr['L'])<=tp; hit_sl=float(rr['H'])>=sl
                if hit_sl:
                    exitp=sl; reason='stop_loss'; break
                if hit_tp:
                    exitp=tp; reason='take_profit'; break
            if exitp is None:
                exitp=float(g.iloc[end]['C']); reason='time_exit'
            pnl=(entry-exitp)/entry*100
            ma25_slope=(g['MA25'].iloc[i]/g['MA25'].iloc[i-1]-1)*100 if g['MA25'].iloc[i-1] else None
            ma_gap=(r['MA5']/r['MA25']-1)*100 if r['MA25'] else None
            close_vs_ma25=(r['C']/r['MA25']-1)*100 if r['MA25'] else None
            close_vs_ma5=(r['C']/r['MA5']-1)*100 if r['MA5'] else None
            prior_5d_ret=(r['C']/g['C'].iloc[max(0,i-5)]-1)*100 if g['C'].iloc[max(0,i-5)] else None
            trades.append({
                'code':code,'signal_date':str(r['Date']),'entry_date':str(g.iloc[ent]['Date']),
                'profit_pct':round(pnl,3),'exit_reason':reason,
                'bear_candle_pct':round(float(bear),3),'ma5_prior5d_rise_pct':round(float(rise),3),
                'ma25_slope_pct':round(float(ma25_slope),4) if ma25_slope is not None else None,
                'ma5_vs_ma25_pct':round(float(ma_gap),3) if ma_gap is not None else None,
                'close_vs_ma25_pct':round(float(close_vs_ma25),3) if close_vs_ma25 is not None else None,
                'close_vs_ma5_pct':round(float(close_vs_ma5),3) if close_vs_ma5 is not None else None,
                'prior_5d_return_pct':round(float(prior_5d_ret),3) if prior_5d_ret is not None else None,
            })
    return signals,trades

def summarize_features(df):
    feats=['bear_candle_pct','ma5_prior5d_rise_pct','ma25_slope_pct','ma5_vs_ma25_pct','close_vs_ma25_pct','close_vs_ma5_pct','prior_5d_return_pct']
    out={}
    groups={
        'take_profit': df[df.exit_reason=='take_profit'],
        'stop_loss': df[df.exit_reason=='stop_loss'],
        'all_wins': df[df.profit_pct>0],
        'all_losses': df[df.profit_pct<=0],
    }
    for name,g in groups.items():
        out[name]={'count':int(len(g))}
        for f in feats:
            s=pd.to_numeric(g[f],errors='coerce').dropna()
            out[name][f]={'mean':round(float(s.mean()),3),'median':round(float(s.median()),3)} if len(s) else None
    bins={}
    for f in feats:
        s=pd.to_numeric(df[f],errors='coerce')
        try:
            q=pd.qcut(s,4,duplicates='drop')
            tmp=df.assign(_bin=q).dropna(subset=['_bin'])
            rows=[]
            for b,g in tmp.groupby('_bin',observed=True):
                pnl=pd.to_numeric(g.profit_pct,errors='coerce')
                wins=pnl[pnl>0]; losses=pnl[pnl<=0]
                gl=-losses.sum(); pf=wins.sum()/gl if gl>0 else None
                rows.append({'bin':str(b),'count':int(len(g)),'win_rate':round(float((pnl>0).mean()*100),2),'avg_pnl':round(float(pnl.mean()),3),'pf':round(float(pf),3) if pf is not None else None})
            bins[f]=rows
        except Exception:
            bins[f]=[]
    return {'groups':out,'quartiles':bins}

def main():
    p=load_prices(); signals,trades=detect_and_trade(p)
    os.makedirs('output',exist_ok=True)
    df=pd.DataFrame(trades)
    if df.empty:
        summary={'signals':signals,'total_trades':0}; analysis={}
    else:
        s=df['profit_pct'].astype(float); w=s[s>0]; l=s[s<=0]
        gl=-l.sum(); pf=(w.sum()/gl) if gl>0 else None
        summary={'signals':signals,'total_trades':len(s),'win_rate':round((s>0).mean()*100,2),'avg_profit_pct_all':round(s.mean(),3),'median_profit_pct':round(s.median(),3),'profit_factor':round(pf,3) if pf is not None else None,'take_profit_count':int((df.exit_reason=='take_profit').sum()),'stop_loss_count':int((df.exit_reason=='stop_loss').sum()),'time_exit_count':int((df.exit_reason=='time_exit').sum()),'date_min':str(p['Date'].min()),'date_max':str(p['Date'].max())}
        analysis=summarize_features(df)
    df.to_csv('output/gakutto_recent_trades.csv',index=False)
    with open('output/gakutto_recent_summary.json','w',encoding='utf-8') as f: json.dump(summary,f,ensure_ascii=False,indent=2)
    with open('output/gakutto_reverse_analysis.json','w',encoding='utf-8') as f: json.dump(analysis,f,ensure_ascii=False,indent=2)
    print('GAKUTTO_RECENT_SUMMARY='+json.dumps(summary,ensure_ascii=False))
    print('GAKUTTO_REVERSE_ANALYSIS='+json.dumps(analysis,ensure_ascii=False))

if __name__=='__main__': main()
