import glob, json, os
import pandas as pd

SL = 5.0
GAP_MAX = 0.5
MAX_HOLD = 15
DAILY_ORDER_CAP = 5
PORTFOLIO_CAPS = [5, 8, 10, 15]


def load_prices():
    frames = []
    for path in glob.glob('docs/prices/*.json'):
        code = os.path.basename(path).replace('.json', '')
        if code.startswith('_'):
            continue
        try:
            rows = json.load(open(path, encoding='utf-8'))
        except Exception:
            continue
        if not rows:
            continue
        df = pd.DataFrame(rows)
        if not {'t', 'o', 'h', 'l', 'c'}.issubset(df.columns):
            continue
        ren = {'t':'Date','o':'O','h':'H','l':'L','c':'C','v':'Vo'}
        df = df.rename(columns={k:v for k,v in ren.items() if k in df.columns})
        if 'Vo' not in df.columns:
            df['Vo'] = pd.NA
        df['Code'] = code
        frames.append(df[['Code','Date','O','H','L','C','Vo']])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def entry_signal(g, i):
    r = g.iloc[i]
    if pd.isna(r.MA5) or pd.isna(r.MA25) or r.O <= 0:
        return False
    bull = (r.C / r.O - 1) * 100
    if not (1.5 <= bull <= 3.5):
        return False
    gap = (r.MA5 / r.MA25 - 1) * 100
    if not (-5.0 <= gap <= 0.0):
        return False
    base = i - 1
    past = base - 5
    if past < 0 or pd.isna(g.MA5.iloc[past]) or not g.MA5.iloc[past] > 0:
        return False
    decline = (g.MA5.iloc[base] / g.MA5.iloc[past] - 1) * 100
    if not (-5.5 <= decline <= -2.0):
        return False
    if any(g.MA5.iloc[j] > g.MA5.iloc[j-1] for j in range(i-2, i)):
        return False
    if not g.MA5.iloc[i] > g.MA5.iloc[i-1]:
        return False
    if (r.C / r.MA5 - 1) * 100 > 4.0:
        return False
    if i < 1 or pd.isna(r.Vo) or pd.isna(g.Vo.iloc[i-1]) or not float(g.Vo.iloc[i-1]) > 0:
        return False
    return float(r.Vo) / float(g.Vo.iloc[i-1]) >= 1.25


def strict_gakutto(g, j):
    if j < 3:
        return False
    if pd.isna(g.MA5.iloc[j]) or pd.isna(g.MA5.iloc[j-1]):
        return False
    if not g.MA5.iloc[j] < g.MA5.iloc[j-1]:
        return False
    if g.MA5.iloc[j-1] < g.MA5.iloc[j-2]:
        return False
    if g.MA5.iloc[j-2] < g.MA5.iloc[j-3]:
        return False
    return float(g.C.iloc[j]) < float(g.O.iloc[j])


def simulate(g, i):
    ent = i + 1
    if ent >= len(g):
        return None
    prev_close = float(g.C.iloc[i])
    entry = float(g.O.iloc[ent])
    if not prev_close > 0 or not entry > 0:
        return None
    entry_gap = (entry / prev_close - 1) * 100
    if entry_gap > GAP_MAX:
        return None
    sl = entry * (1 - SL / 100)
    end = min(ent + MAX_HOLD - 1, len(g) - 1)
    pending = False
    for j in range(ent, end + 1):
        if pending:
            px = float(g.O.iloc[j])
            return {'pnl': (px-entry)/entry*100, 'reason':'gakutto_strict_next_open', 'hold':j-ent+1,
                    'entry_date':str(g.Date.iloc[ent]), 'exit_date':str(g.Date.iloc[j]), 'entry_gap':entry_gap}
        if float(g.L.iloc[j]) <= sl:
            return {'pnl':-SL, 'reason':'stop_loss', 'hold':j-ent+1,
                    'entry_date':str(g.Date.iloc[ent]), 'exit_date':str(g.Date.iloc[j]), 'entry_gap':entry_gap}
        if j < end and strict_gakutto(g, j):
            pending = True
    if end + 1 < len(g):
        px = float(g.O.iloc[end+1])
        return {'pnl':(px-entry)/entry*100, 'reason':'time_exit_next_open', 'hold':end-ent+2,
                'entry_date':str(g.Date.iloc[ent]), 'exit_date':str(g.Date.iloc[end+1]), 'entry_gap':entry_gap}
    px = float(g.C.iloc[end])
    return {'pnl':(px-entry)/entry*100, 'reason':'time_exit', 'hold':end-ent+1,
            'entry_date':str(g.Date.iloc[ent]), 'exit_date':str(g.Date.iloc[end]), 'entry_gap':entry_gap}


def stats(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return {'count':0}
    s = df.pnl.astype(float)
    pos = s[s > 0]
    neg = s[s <= 0]
    gross_loss = -neg.sum()
    pf = pos.sum() / gross_loss if gross_loss > 0 else None
    return {
        'count':len(df),
        'win_rate':round((s > 0).mean()*100, 2),
        'avg_pnl':round(s.mean(), 3),
        'median_pnl':round(s.median(), 3),
        'pf':round(pf, 3) if pf is not None else None,
        'avg_hold':round(df.hold.mean(), 2),
        'reasons':df.reason.value_counts().to_dict(),
    }


def apply_portfolio_cap(filled, cap):
    selected = []
    active = []
    skipped_full = 0
    by_entry = {}
    for r in filled:
        by_entry.setdefault(r['entry_date'], []).append(r)
    for entry_date in sorted(by_entry):
        active = [p for p in active if p['exit_date'] > entry_date]
        slots = max(0, cap - len(active))
        todays = sorted(by_entry[entry_date], key=lambda x: (x['ma25_distance'], x['code']))
        take = todays[:slots]
        skipped_full += max(0, len(todays) - len(take))
        selected.extend(take)
        active.extend(take)
    return selected, skipped_full


def main():
    prices = load_prices()
    raw = []
    for code, g in prices.groupby('Code'):
        g = g.sort_values('Date').reset_index(drop=True).copy()
        for c in ['O','H','L','C','Vo']:
            g[c] = pd.to_numeric(g[c], errors='coerce')
        g['MA5'] = g.C.rolling(5).mean()
        g['MA25'] = g.C.rolling(25).mean()
        for i in range(25, len(g)-1):
            if not entry_signal(g, i):
                continue
            sim = simulate(g, i)
            raw.append({'code':code, 'signal_date':str(g.Date.iloc[i]),
                        'ma25_distance':abs((float(g.MA5.iloc[i])/float(g.MA25.iloc[i])-1)*100),
                        'gap_ok':sim is not None, **(sim or {})})

    raw_df = pd.DataFrame(raw)
    ordered = []
    if not raw_df.empty:
        for _, day in raw_df.groupby('signal_date'):
            day = day.sort_values(['ma25_distance','code']).head(DAILY_ORDER_CAP)
            ordered.extend(day.to_dict('records'))
    filled = [r for r in ordered if r.get('gap_ok')]

    comparisons = {}
    for cap in PORTFOLIO_CAPS:
        selected, skipped = apply_portfolio_cap(filled, cap)
        comparisons[str(cap)] = {'stats':stats(selected), 'skipped_due_portfolio_full':skipped}

    result = {
        'rule':'daily MA25-near top5 orders -> next-open gap<=+0.5% fills; SL -5%; strict gakutto next-open; max15 next-open',
        'raw_signals':len(raw),
        'ordered_top5':len(ordered),
        'filled_without_portfolio_cap':stats(filled),
        'portfolio_caps':comparisons,
    }
    print('KUITTO_PORTFOLIO_CAP_COMPARE=' + json.dumps(result, ensure_ascii=False))
    os.makedirs('output', exist_ok=True)
    json.dump(result, open('output/kuitto_portfolio_cap_analysis.json','w',encoding='utf-8'), ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
