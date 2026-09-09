import json, os
import pandas as pd

INPUT='output/kuitto_reverse_trades.csv'
OUT='output/kuitto_timesplit.json'

def stats(df):
    if df.empty:
        return {'count':0}
    p=pd.to_numeric(df['profit_pct'],errors='coerce').dropna()
    w=p[p>0]; l=p[p<=0]; gl=-l.sum(); pf=(w.sum()/gl) if gl>0 else None
    return {
        'count':int(len(p)),
        'win_rate':round((p>0).mean()*100,2),
        'avg_pnl':round(p.mean(),3),
        'median_pnl':round(p.median(),3),
        'pf':round(pf,3) if pf is not None else None,
        'tp':int((df.loc[p.index,'exit_reason']=='take_profit').sum()),
        'sl':int((df.loc[p.index,'exit_reason']=='stop_loss').sum()),
        'time':int((df.loc[p.index,'exit_reason']=='time_exit').sum()),
    }

def main():
    df=pd.read_csv(INPUT)
    df['signal_date']=pd.to_datetime(df['signal_date'])
    # Wide pullback version already selected from reverse analysis.
    wide=df[
        df['ma5_prior5d_decline_pct'].between(-5.5,-2.0, inclusive='both') &
        df['ma5_vs_ma25_pct'].between(-5.0,0.0, inclusive='both') &
        (df['close_vs_ma5_pct'] <= 4.0) &
        df['bull_candle_pct'].between(1.5,3.5, inclusive='both')
    ].copy().sort_values('signal_date')

    # Split chronologically by the median signal date of qualifying trades.
    split_date=wide['signal_date'].median()
    first=wide[wide['signal_date'] <= split_date]
    second=wide[wide['signal_date'] > split_date]

    # Also report quarterly-ish thirds as a stability sanity check.
    thirds=[]
    if len(wide):
        q1=wide['signal_date'].quantile(1/3)
        q2=wide['signal_date'].quantile(2/3)
        chunks=[wide[wide['signal_date']<=q1], wide[(wide['signal_date']>q1)&(wide['signal_date']<=q2)], wide[wide['signal_date']>q2]]
        for i,g in enumerate(chunks,1):
            x=stats(g)
            if len(g):
                x['date_min']=str(g['signal_date'].min().date()); x['date_max']=str(g['signal_date'].max().date())
            thirds.append({'part':i,**x})

    result={
        'wide_all':stats(wide),
        'split_date':str(split_date.date()) if pd.notna(split_date) else None,
        'first_half':{**stats(first),'date_min':str(first['signal_date'].min().date()) if len(first) else None,'date_max':str(first['signal_date'].max().date()) if len(first) else None},
        'second_half':{**stats(second),'date_min':str(second['signal_date'].min().date()) if len(second) else None,'date_max':str(second['signal_date'].max().date()) if len(second) else None},
        'thirds':thirds,
    }
    os.makedirs('output',exist_ok=True)
    with open(OUT,'w',encoding='utf-8') as f: json.dump(result,f,ensure_ascii=False,indent=2)
    print('KUITTO_TIMESPLIT='+json.dumps(result,ensure_ascii=False))

if __name__=='__main__': main()
