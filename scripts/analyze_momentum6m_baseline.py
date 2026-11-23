"""Research: the live Momentum6M rule rebuilt without survivor bias, as the baseline for improving it.

Live rule (scripts in stock-screening-data: monthly_momentum_candidates.py)
- Signal at each month-end close. Universe: Prime with 20-session median trading value >= 100M yen.
- Rank by 126-session adjusted-close return, after excluding
  volume surge: 20-session average trading value >= 2x the 120-session average (or too short a history)
  heavy short interest: margin short balance / 20-session average volume above the 80th percentile of the
  liquid universe; margin data usable 4 sessions after its as-of date, ignored when older than 21 days.
- Top 5; buy at the next session's open; sell at the first open after the next month-end.
Here: point-in-time Prime (TSE 1st before 2022-04) incl. delisted issues (closed at the last price), trading
value approximated by raw close x volume, 0.3% round-trip cost (as in new_condition_scan_summary.json).
Variants: filters off / volume filter only / live; top 5 / 10 / 20; the same live rule restricted to issues
still listed today (survivor universe, what the earlier research used); 200 random 5-stock picks from the
liquid universe (how much luck alone moves a 5-stock result).
Output: results/momentum6m_baseline.json (aggregate statistics only).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_ma75_turn_trend import LIQ_MIN_YEN, load_prices, prime_mask, read_dir
from price_adjust import normalize_code

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results/momentum6m_baseline.json"
LOOKBACK = 126
VOLSURGE_MAX = 2.0
SHORT_PCT_MAX = 0.80
MARGIN_LAG = 4
MARGIN_MAX_AGE = 21
COST = 0.003
SPLIT = pd.Period("2022-01", "M")
N_RANDOM = 200
SEED = 6


def margin_short(dates):
    m = read_dir(DATA / "margin_interest", ["Date", "Code", "ShrtVol"])
    m["Code"] = m.Code.astype(str).map(normalize_code)
    m["Date"] = pd.to_datetime(m.Date, errors="coerce")
    m["ShrtVol"] = pd.to_numeric(m.ShrtVol, errors="coerce")
    m = m.dropna(subset=["Date"])
    pos = dates.searchsorted(m.Date.values) + MARGIN_LAG
    m["avail"] = [dates[p] if p < len(dates) else pd.Timestamp.max for p in pos]
    return m.sort_values("avail")


def short_days_at(m, d, codes, adv20):
    x = m[(m.avail <= d) & m.Code.isin(codes)]
    if x.empty:
        return None, None
    latest = x.sort_values("Date").drop_duplicates("Code", keep="last")
    asof = latest.Date.max()
    if (d - asof).days > MARGIN_MAX_AGE:
        return None, asof
    s = latest.set_index("Code").ShrtVol.reindex(codes) / adv20.reindex(codes)
    return s, asof


def month_trade(PO, P, e, x, ci):
    """Next-open entry e, exit open x; an issue without data at x is closed at its last close before x."""
    pin = PO[e, ci]
    pout = PO[x, ci] if x < len(PO) else np.full(len(ci), np.nan)
    last = np.array([P[e:x + 1, j][~np.isnan(P[e:x + 1, j])][-1] if (~np.isnan(P[e:x + 1, j])).any() else np.nan
                     for j in ci]) if len(ci) else np.array([])
    pout = np.where(np.isnan(pout), last, pout)
    return pout / pin - 1


def pstats(r):
    r = pd.Series(r).dropna()
    if len(r) < 12:
        return None
    eq = (1 + r).cumprod()
    return {"months": int(len(r)), "cagr_pct": round(float(eq.iloc[-1] ** (12 / len(r)) - 1) * 100, 2),
            "max_dd_pct": round(float((eq / eq.cummax() - 1).min()) * 100, 1),
            "ann_vol_pct": round(float(r.std() * np.sqrt(12) * 100), 1)}


def excess_stats(r, b):
    e = (pd.Series(r) - pd.Series(b)).dropna()
    if len(e) < 12:
        return None
    return {"mean_excess_pct_per_month": round(float(e.mean() * 100), 3),
            "t": round(float(e.mean() / (e.std() / np.sqrt(len(e)))), 2),
            "months_beating_pct": round(float((e > 0).mean() * 100), 1)}


def tests():
    P = np.array([[10, 20], [11, np.nan], [12, np.nan], [13, np.nan]], dtype=float)
    PO = P.copy()
    r = month_trade(PO, P, 0, 3, np.array([0, 1]))
    assert abs(r[0] - 0.3) < 1e-12 and abs(r[1] - 0.0) < 1e-12   # issue 1 stopped trading: last close 20
    s = pstats([0.01] * 24)
    assert abs(s["cagr_pct"] - round((1.01 ** 12 - 1) * 100, 2)) < 1e-9 and s["max_dd_pct"] == 0
    print("trade return with delisting and stats tests passed", flush=True)


def main():
    tests()
    a = load_prices()
    P = a.pivot(index="Date", columns="Code", values="P").sort_index()
    PO = a.pivot(index="Date", columns="Code", values="PO").reindex_like(P)
    to = a.pivot(index="Date", columns="Code", values="to").reindex_like(P)
    vo = a.pivot(index="Date", columns="Code", values="Vo").reindex_like(P)
    del a
    dates = P.index
    survivors = set(pd.read_parquet(DATA / "master/master_latest.parquet", columns=["Code"]).Code.astype(str).map(normalize_code))
    prime = prime_mask(dates, P.columns)
    liq = to.rolling(20, min_periods=20).median() >= LIQ_MIN_YEN
    mom = P / P.shift(LOOKBACK) - 1
    volsurge = to.rolling(20).mean() / to.rolling(120, min_periods=100).mean()
    adv20 = vo.rolling(20, min_periods=15).mean()
    margin = margin_short(dates)
    Pa, POa = P.to_numpy(), PO.to_numpy()
    col = {c: j for j, c in enumerate(P.columns)}

    ym = pd.Series(dates.to_period("M"), index=dates)
    month_ends = dates[ym.ne(ym.shift(-1)).to_numpy()]
    first_of_month = {p: dates[ym.eq(p).to_numpy()][0] for p in ym.unique()}
    variants = {"raw_top5": (5, False, False, False), "volume_filter_top5": (5, True, False, False),
                "live_top5": (5, True, True, False), "live_top10": (10, True, True, False),
                "live_top20": (20, True, True, False), "live_top5_survivors_only": (5, True, True, True)}
    rets = {k: {} for k in variants}
    bench, bench_surv = {}, {}
    rnd = [dict() for _ in range(N_RANDOM)]
    rng = np.random.default_rng(SEED)
    short_used = 0
    for d in month_ends:
        i = dates.get_loc(d)
        p = pd.Period(d, "M")
        x_date = first_of_month.get(p + 2)
        if i < LOOKBACK + 20 or x_date is None:
            continue
        e, x = i + 1, dates.get_loc(x_date)
        univ = (prime.loc[d] & liq.loc[d] & mom.loc[d].notna())
        codes = univ.index[univ.to_numpy()]
        if len(codes) < 100:
            continue
        sd, asof = short_days_at(margin, d, codes, adv20.loc[d])
        if sd is not None:
            short_used += 1
            short_bad = sd.rank(pct=True) > SHORT_PCT_MAX
        else:
            short_bad = pd.Series(False, index=codes)
        vs = volsurge.loc[d, codes]
        vol_bad = vs.isna() | (vs >= VOLSURGE_MAX)
        m = mom.loc[d, codes]
        ci_all = np.array([col[c] for c in codes])
        r_all = month_trade(POa, Pa, e, x, ci_all)
        bench[p] = np.nanmean(r_all)
        surv_mask = np.array([c in survivors for c in codes])
        bench_surv[p] = np.nanmean(r_all[surv_mask])
        rmap = dict(zip(codes, r_all))
        for name, (n, use_vol, use_short, surv_only) in variants.items():
            ok = pd.Series(True, index=codes)
            if use_vol:
                ok &= ~vol_bad
            if use_short:
                ok &= ~short_bad.reindex(codes).fillna(False)
            if surv_only:
                ok &= pd.Series(surv_mask, index=codes)
            pick = m[ok].sort_values(ascending=False).head(n).index
            rets[name][p] = np.nanmean([rmap[c] for c in pick]) - COST
        for k in range(N_RANDOM):
            rnd[k][p] = np.nanmean(rng.choice(r_all[~np.isnan(r_all)], size=5, replace=False)) - COST
    print(f"months {len(bench)}; short filter usable in {short_used}", flush=True)

    b = pd.Series(bench)
    out = {"settings": {"universe": "point-in-time Prime/TSE1 incl. delisted, 20d median value (C*Vo) >= 1e8",
                        "rule": "126-session return rank, volume-surge and short-interest exclusions, top N, next open to first open after next month-end",
                        "cost_roundtrip_pct": COST * 100, "split": str(SPLIT), "months": [str(b.index.min()), str(b.index.max())],
                        "short_filter_usable_months": short_used},
           "benchmark_equal_weight": {"all": pstats(b), "before_2022": pstats(b[b.index < SPLIT]), "from_2022": pstats(b[b.index >= SPLIT]),
                                      "by_year_pct": {str(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in b.groupby(b.index.year)}},
           "benchmark_survivors_only": {"all": pstats(pd.Series(bench_surv))},
           "variants": {}}
    for name, r in rets.items():
        s = pd.Series(r)
        out["variants"][name] = {"all": pstats(s), "before_2022": pstats(s[s.index < SPLIT]), "from_2022": pstats(s[s.index >= SPLIT]),
                                 "excess_all": excess_stats(s, b), "excess_before_2022": excess_stats(s[s.index < SPLIT], b),
                                 "excess_from_2022": excess_stats(s[s.index >= SPLIT], b),
                                 "by_year_pct": {str(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in s.groupby(s.index.year)}}
        v = out["variants"][name]
        print(name, v["all"], "| <2022", v["before_2022"]["cagr_pct"], ">=2022", v["from_2022"]["cagr_pct"],
              "| excess", v["excess_all"], flush=True)
    cagr = [pstats(pd.Series(x))["cagr_pct"] for x in rnd]
    live = out["variants"]["live_top5"]["all"]["cagr_pct"]
    out["random5_from_liquid_universe"] = {
        "cagr_pct_percentiles": {q: round(float(np.percentile(cagr, q)), 2) for q in (5, 25, 50, 75, 95)},
        "share_of_random_runs_beating_live_top5": round(float(np.mean(np.array(cagr) >= live)), 3)}
    print("benchmark", out["benchmark_equal_weight"]["all"], "| <2022", out["benchmark_equal_weight"]["before_2022"]["cagr_pct"],
          ">=2022", out["benchmark_equal_weight"]["from_2022"]["cagr_pct"], "| survivors-only bench", out["benchmark_survivors_only"]["all"], flush=True)
    print("random5", out["random5_from_liquid_universe"], flush=True)
    print("live_top5 by year", out["variants"]["live_top5"]["by_year_pct"], "bench", out["benchmark_equal_weight"]["by_year_pct"], flush=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
