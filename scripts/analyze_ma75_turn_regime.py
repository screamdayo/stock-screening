"""Research: can the good years of the MA75 turn rule be identified in advance? Two market-state filters on buys.

Base trades: MA75 slope20 turns up -> buy next open; turns down -> sell next open (results/ma75_turn_trend.json).
Filters (known at the buy signal close; exits unchanged):
  own_track  share of base-rule trades entered in the previous 120 sessions that are in profit at today's close
             (closed trades count at their realized return). Needs >= 20 such trades, otherwise no buy.
  breadth    share of the universe (point-in-time Prime, turnover >= 100M) closing above its MA75.
Out-of-sample check: each filter's threshold is chosen on entries 2016-2021 only (best excess return per holding
day vs the equal-weight universe, keeping >= 30% of trades), then applied unchanged to entries 2022-2026.
All thresholds are reported for both halves. Output: results/ma75_turn_regime.json (aggregate only).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_ma75_turn_trend import LIQ_MIN_YEN, MIN_PRICE, MA_LEN, SLOPE_LAG, build_trades, load_prices, prime_mask, slope_of

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/ma75_turn_regime.json"
LOOKBACK = 120
MIN_TRADES = 20
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
SPLIT = pd.Timestamp("2022-01-01")
MIN_KEEP = 0.30


def own_track(close, tr, n_days):
    """share_in_profit[d] over trades with entry in [d-LOOKBACK, d-1]; mark-to-market at close d, realized after exit."""
    C = close.to_numpy()
    pos = np.zeros(n_days)
    cnt = np.zeros(n_days)
    for j, e, x, p_in, p_out in tr[["j", "entry", "exit", "p_in", "p_out"]].itertuples(index=False):
        d0, d1 = e + 1, min(e + LOOKBACK, n_days - 1)
        if d0 > d1:
            continue
        d = np.arange(d0, d1 + 1)
        val = C[d, j] / p_in - 1
        val = np.where(d >= x, p_out / p_in - 1, val)  # exit happens at the open of day x
        ok = ~np.isnan(val)
        pos[d[ok]] += val[ok] > 0
        cnt[d[ok]] += 1
    share = np.where(cnt >= MIN_TRADES, pos / np.maximum(cnt, 1), np.nan)
    return pd.Series(share, index=close.index)


def daily_series(close, tr, universe):
    ret = close.pct_change(fill_method=None)
    held = np.zeros(close.shape, dtype=bool)
    for j, e, x in tr[["j", "entry", "exit"]].itertuples(index=False):
        held[e + 1:x + 1, j] = True
    strat = ret.where(held).mean(axis=1).fillna(0.0)
    bench = ret.where(universe.shift(1, fill_value=False)).mean(axis=1).fillna(0.0)
    return strat, bench, pd.Series(held.sum(axis=1), index=close.index)


def pstats(s):
    if len(s) < 50:
        return None
    eq = (1 + s).cumprod()
    return {"cagr_pct": round(float(eq.iloc[-1] ** (245 / len(s)) - 1) * 100, 2),
            "max_dd_pct": round(float((eq / eq.cummax() - 1).min()) * 100, 1)}


def trade_stats(t, exc):
    if t.empty:
        return {"n": 0}
    e = exc[t.index]
    return {"n": int(len(t)), "mean_pct": round(t.ret.mean() * 100, 2), "win_rate": round(float((t.ret > 0).mean()), 3),
            "mean_days": round(float(t.days.mean()), 1),
            "excess_per_day_bp": round(float(e.sum() / t.days.clip(lower=1).sum() * 1e4), 2)}


def evaluate(close, tr, universe, idx, start):
    """Stats per half for a trade set (closed + delisted only for trade stats)."""
    done = tr[tr.status != "open"]
    if done.empty:
        return {"2016_2021": {"trades": {"n": 0}}, "2022_2026": {"trades": {"n": 0}}, "by_entry_year": {}}
    exc = (1 + done.ret) / (idx[done.exit.to_numpy()] / idx[done.entry.to_numpy() - 1]) - 1
    strat, bench, npos = daily_series(close, tr, universe)
    out = {}
    for name, m_tr, m_day in (("2016_2021", done.entry_date < SPLIT, close.index < SPLIT),
                              ("2022_2026", done.entry_date >= SPLIT, close.index >= SPLIT)):
        m_day = m_day & (np.arange(len(close)) >= start)
        out[name] = {"trades": trade_stats(done[m_tr], exc),
                     "portfolio": pstats(strat[m_day]), "universe": pstats(bench[m_day]),
                     "share_of_days_invested": round(float((npos[m_day] > 0).mean()), 3)}
    out["by_entry_year"] = {str(y): {"n": int(len(g)), "mean_pct": round(g.ret.mean() * 100, 2)}
                            for y, g in done.groupby(done.entry_date.dt.year)}
    return out


def tests():
    idx = pd.bdate_range("2020-01-01", periods=300)
    close = pd.DataFrame({"A": np.linspace(100, 130, 300), "B": np.linspace(100, 70, 300)}, index=idx)
    tr = pd.DataFrame([(0, 10, 200, close.A.iloc[10], close.A.iloc[200]) for _ in range(12)]
                      + [(1, 10, 200, close.B.iloc[10], close.B.iloc[200]) for _ in range(12)],
                      columns=["j", "entry", "exit", "p_in", "p_out"])
    s = own_track(close, tr, len(close))
    assert np.isnan(s.iloc[10]) and abs(s.iloc[50] - 0.5) < 1e-12 and np.isnan(s.iloc[131])
    # realized value after exit
    tr2 = pd.DataFrame([(0, 10, 20, 100.0, 90.0)] * 20, columns=["j", "entry", "exit", "p_in", "p_out"])
    s2 = own_track(close, tr2, len(close))
    assert s2.iloc[15] == 1.0 and s2.iloc[25] == 0.0
    print("own-track window, realized-exit and minimum-count tests passed", flush=True)


def main():
    tests()
    a = load_prices()
    close = a.pivot(index="Date", columns="Code", values="P").sort_index()
    opn = a.pivot(index="Date", columns="Code", values="PO").reindex_like(close)
    to = a.pivot(index="Date", columns="Code", values="to").reindex_like(close)
    del a
    universe = prime_mask(close.index, close.columns) & (to.rolling(20, min_periods=15).median() >= LIQ_MIN_YEN) \
        & (close >= MIN_PRICE)
    slope = slope_of(close)
    turn = (slope.shift(1) <= 0) & (slope > 0)
    base = build_trades(close, opn, slope, universe, turn)
    bench = close.pct_change(fill_method=None).where(universe.shift(1, fill_value=False)).mean(axis=1).fillna(0.0)
    idx = (1 + bench).cumprod().to_numpy()
    start = MA_LEN + SLOPE_LAG + 1

    ma = close.rolling(MA_LEN, min_periods=MA_LEN).mean()
    breadth = ((close > ma) & universe & ma.notna()).sum(axis=1) / (universe & ma.notna()).sum(axis=1).replace(0, np.nan)
    track = own_track(close, base, len(close))
    indicators = {"own_track": track, "breadth": breadth}

    out = {"settings": {"base": "MA75 slope20 turn-up buy / turn-down sell, next open",
                        "own_track": f"share of base trades from the last {LOOKBACK} sessions in profit (>= {MIN_TRADES} trades)",
                        "breadth": "share of universe closing above MA75",
                        "selection": f"threshold chosen on 2016-2021 entries by excess/day, keeping >= {int(MIN_KEEP*100)}% of trades",
                        "thresholds": THRESHOLDS},
           "indicator_by_year_mean": {k: {str(y): round(float(v), 3) for y, v in s.groupby(s.index.year).mean().items()}
                                      for k, s in indicators.items()},
           "base": evaluate(close, base, universe, idx, start), "filters": {}, "selected": {}}
    base_first = (base[(base.status != "open") & (base.entry_date < SPLIT)])
    for name, ind in indicators.items():
        res = {}
        for th in THRESHOLDS:
            ok = pd.DataFrame(np.repeat((ind > th).to_numpy()[:, None], close.shape[1], axis=1),
                              index=close.index, columns=close.columns)
            t = build_trades(close, opn, slope, universe, turn & ok)
            res[f"{th:.1f}"] = evaluate(close, t, universe, idx, start)
            f1 = res[f"{th:.1f}"]["2016_2021"]["trades"]
            f2 = res[f"{th:.1f}"]["2022_2026"]["trades"]
            print(name, th, "early", f1.get("n"), f1.get("excess_per_day_bp"), res[f"{th:.1f}"]["2016_2021"].get("portfolio"),
                  "| late", f2.get("n"), f2.get("excess_per_day_bp"), res[f"{th:.1f}"]["2022_2026"].get("portfolio"), flush=True)
        out["filters"][name] = res
        cands = {k: v for k, v in res.items() if v["2016_2021"]["trades"].get("n", 0) >= MIN_KEEP * len(base_first)}
        best = max(cands, key=lambda k: cands[k]["2016_2021"]["trades"]["excess_per_day_bp"]) if cands else None
        out["selected"][name] = {"threshold": best,
                                 "chosen_on_2016_2021": res[best]["2016_2021"] if best else None,
                                 "applied_2022_2026": res[best]["2022_2026"] if best else None}
        print("selected", name, best, flush=True)
    for h in ("2016_2021", "2022_2026"):
        b = out["base"][h]
        print("base", h, b["trades"], b["portfolio"], "universe", b["universe"], flush=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
