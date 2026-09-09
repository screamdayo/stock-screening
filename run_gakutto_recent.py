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
            if (r['C']/r['O']-1)*100 > -MIN_BEAR: continue
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
            trades.append({'code':code,'signal_date':str(r['Date']),'entry_date':str(g.iloc[ent]['Date']),'profit_pct':round(pnl,3),'exit_reason':reason})
    return signals,trades

def main():
    p=load_prices(); signals,trades=detect_and_trade(p)
    os.makedirs('output',exist_ok=True)
    df=pd.DataFrame(trades)
    if df.empty:
        summary={'signals':signals,'total_trades':0}
    else:
        s=df['profit_pct'].astype(float); w=s[s>0]; l=s[s<=0]
        gl=-l.sum(); pf=(w.sum()/gl) if gl>0 else None
        summary={'signals':signals,'total_trades':len(s),'win_rate':round((s>0).mean()*100,2),'avg_profit_pct_all':round(s.mean(),3),'median_profit_pct':round(s.median(),3),'profit_factor':round(pf,3) if pf is not None else None,'take_profit_count':int((df.exit_reason=='take_profit').sum()),'stop_loss_count':int((df.exit_reason=='stop_loss').sum()),'time_exit_count':int((df.exit_reason=='time_exit').sum()),'date_min':str(p['Date'].min()),'date_max':str(p['Date'].max())}
    df.to_csv('output/gakutto_recent_trades.csv',index=False)
    with open('output/gakutto_recent_summary.json','w',encoding='utf-8') as f: json.dump(summary,f,ensure_ascii=False,indent=2)
    print('GAKUTTO_RECENT_SUMMARY='+json.dumps(summary,ensure_ascii=False))

if __name__=='__main__': main()
