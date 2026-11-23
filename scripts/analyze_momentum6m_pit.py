"""Research: Momentum6M monthly rule re-checked without survivor bias, plus a few pre-declared variants.

Production rule (stock-screening-data scripts/monthly_momentum_candidates.py, 2026-11):
  signal = last session of month; universe = Prime with 20-session median trading value (Va) >= 100M yen
  rank   = 126-session adjusted-close return, descending
  exclude volume surge (20d avg Va >= 2x the 120d avg) and short_days (margin ShrtVol, as-of + 4 sessions,
          / 20d avg raw volume) above the 80th percentile of the liquid universe (only if margin data <= 21 days old)
  top 5, buy limit = signal close on the next session (fills at the open if open <= limit, at the limit if the low
  reaches it, otherwise the slot stays in cash), sell at the first open after the next month-end.
The earlier study (results/new_condition_scan_summary.json) used current Prime survivors only. Here the universe is
point-in-time Prime (TSE 1st before 2022-04) from monthly master snapshots and includes delisted issues; a holding
that stops trading is closed at its last close. 0.3% round-trip cost per position. Monthly equal-weight slots
(1/N each, empty slot = cash). Benchmark: equal-weight liquid point-in-time Prime universe, open-to-open, no cost.
Variants (fixed before running): production; buy at the open instead of a limit; survivor universe (to measure the
bias); no filters; top 10; production but no new buys when TOPIX is below its 200-day average; production ranked by
the 12-month return skipping the last month. Periods: signals 2017-2021 (A) and 2022-2026 (B), as before.
Output: results/momentum6m_pit.json (aggregate statistics only).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_ma75_turn_trend import PRIME_NAMES, prime_mask, read_dir
from price_adjust import normalize_code, stale_basis_multiplier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results/momentum6m_pit.json"
COLS = ["Date", "Code", "O", "L", "C", "Vo", "Va", "AdjO", "AdjL", "AdjC", "AdjFactor"]
LIQ_MIN, VOLSURGE_MAX, SHORT_PCT_MAX, MARGIN_LAG, MARGIN_MAX_AGE = 1e8, 2.0, 0.80, 4, 21
COST = 0.003


def read_cols(path, cols):
    d = pd.read_parquet(path)
    for c in cols:
        if c not in d.columns:
            d[c] = np.nan
    return d[cols]


def load_bars():
    dm = pd.read_parquet(DATA / "delisted/delisted_master.parquet")
    dm = dm[(dm.Code.astype(str).str.len() == 4) & (dm.ProdCat.astype(str) == "011")]
    dl = pd.concat([read_cols(f, COLS) for f in sorted((DATA / "delisted").glob("batch_*.parquet"))], ignore_index=True)
    dl["Code"] = dl.Code.astype(str).map(normalize_code)
    dl = dl[dl.Code.isin(set(dm.Code.astype(str)))].assign(src=2)
    a = pd.concat([read_cols(f, COLS).assign(src=0) for f in sorted((DATA / "batches").glob("*.parquet"))]
                  + [read_cols(f, COLS).assign(src=1) for f in sorted((DATA / "daily_updates").glob("*.parquet"))],
                  ignore_index=True)
    a["Code"] = a.Code.astype(str).map(normalize_code)
    a = a[~a.Code.isin(set(dl.Code))]
    a = pd.concat([a, dl], ignore_index=True)
    a["Date"] = pd.to_datetime(a.Date)
    for c in COLS[2:]:
        a[c] = pd.to_numeric(a[c], errors="coerce")
    a = a.sort_values(["Code", "Date", "src"]).drop_duplicates(["Code", "Date"], keep="last").reset_index(drop=True)
    m = stale_basis_multiplier(a)
    a["P"], a["PO"], a["PL"] = a.AdjC * m, a.AdjO * m, a.AdjL * m
    a["Va"] = a.Va.fillna(a.C * a.Vo)
    a = a[a.P > 0].reset_index(drop=True)
    print(f"rows {len(a):,}, issues {a.Code.nunique():,}, delisted {dl.Code.nunique():,}", flush=True)
    return a


def survivors():
    mh = read_dir(DATA / "master_history", ["Code", "MktNm", "ProdCat", "SnapshotDate"])
    mh["Code"] = mh.Code.astype(str).map(normalize_code)
    last = mh.SnapshotDate.max()
    cur = mh[(mh.SnapshotDate == last) & mh.MktNm.isin(PRIME_NAMES) & (mh.ProdCat.astype(str) == "011")]
    return set(cur.Code)


def add_avail(margin, tdates):
    pos = tdates.searchsorted(margin.Date.values) + MARGIN_LAG
    far = pd.Timestamp("2262-01-01")
    margin["avail"] = [tdates[p] if p < len(tdates) else far for p in pos]
    return margin


def short_days_at(margin, signal, codes, adv20):
    m = margin[margin.avail <= signal]
    if m.empty:
        return None, None
    asof = m.Date.max()
    if (signal - asof).days > MARGIN_MAX_AGE:
        return None, asof
    latest = m[m.Date > asof - pd.Timedelta(days=MARGIN_MAX_AGE)].sort_values("Date").drop_duplicates("Code", keep="last")
    sv = latest.set_index("Code").ShrtVol.reindex(codes)
    return sv / adv20.reindex(codes), asof


def simulate(sel, P, PO, PL, month_info, limit=True):
    """sel: list per month of selected code-index arrays (len N, may be shorter). Returns monthly returns, trade list."""
    rets, trades = [], []
    n = len(P)
    for (sig, e, x, slots), ci in zip(month_info, sel):
        r = 0.0
        for j in ci:
            lim, op, lo = P[sig, j], PO[e, j], PL[e, j]
            if np.isnan(op):
                continue
            if limit:
                if op <= lim:
                    px = op
                elif not np.isnan(lo) and lo <= lim:
                    px = lim
                else:
                    trades.append(("unfilled", 0.0))
                    continue
            else:
                px = op
            col = P[:, j]
            if x < n and not np.isnan(PO[x, j]):
                out = PO[x, j]
            else:  # stopped trading before the exit open: last close up to the exit
                seg = col[e:min(x, n - 1) + 1]
                out = seg[~np.isnan(seg)][-1]
            tr = out / px - 1 - COST
            trades.append(("filled", tr))
            r += tr / slots
        rets.append(r)
    return np.array(rets), trades


def stats(rets, months, bench=None):
    def one(m):
        x = rets[m]
        if len(x) < 6:
            return None
        eq = np.cumprod(1 + x)
        o = {"months": int(len(x)), "cagr_pct": round(float(eq[-1] ** (12 / len(x)) - 1) * 100, 2),
             "max_dd_pct_monthly": round(float((eq / np.maximum.accumulate(eq) - 1).min()) * 100, 1),
             "monthly_mean_pct": round(float(x.mean() * 100), 3), "monthly_win_pct": round(float((x > 0).mean() * 100), 1)}
        if bench is not None:
            ex = x - bench[m]
            o["monthly_excess_pct"] = round(float(ex.mean() * 100), 3)
            o["excess_t"] = round(float(ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex)))), 2) if ex.std() > 0 else None
        return o
    yrs = months.year
    out = {"all": one(np.ones(len(rets), bool)), "A_2017_2021": one(yrs <= 2021), "B_2022_2026": one(yrs >= 2022),
           "by_year_pct": {str(y): round(float((1 + rets[yrs == y]).prod() - 1) * 100, 1) for y in sorted(set(yrs))}}
    if bench is not None:
        by = {y: (1 + rets[yrs == y]).prod() - (1 + bench[yrs == y]).prod() for y in sorted(set(yrs))}
        out["years_beating_benchmark"] = f"{sum(v > 0 for v in by.values())}/{len(by)}"
    return out


def tests():
    P = np.array([[100.0], [101.0], [102.0], [103.0]])
    PO = np.array([[100.0], [99.0], [102.0], [104.0]])
    PL = np.array([[99.0], [98.0], [101.0], [102.0]])
    r, t = simulate([np.array([0])], P, PO, PL, [(0, 1, 3, 1)])
    assert abs(r[0] - (104 / 99 - 1 - COST)) < 1e-12              # open below limit -> fill at open
    PO2 = PO.copy(); PO2[1, 0] = 101.0
    r, _ = simulate([np.array([0])], P, PO2, PL, [(0, 1, 3, 1)])
    assert abs(r[0] - (104 / 100 - 1 - COST)) < 1e-12             # low touches limit -> fill at limit
    PL2 = PL.copy(); PL2[1, 0] = 100.5
    r, t = simulate([np.array([0])], P, PO2, PL2, [(0, 1, 3, 1)])
    assert r[0] == 0.0 and t[0][0] == "unfilled"                   # never reaches limit -> cash
    P3 = P.copy(); P3[2:, 0] = np.nan; PO3 = PO.copy(); PO3[2:, 0] = np.nan
    r, _ = simulate([np.array([0])], P3, PO3, PL, [(0, 1, 3, 1)])
    assert abs(r[0] - (101 / 99 - 1 - COST)) < 1e-12              # delisted -> last close
    print("limit fill, vacancy and delisting exit tests passed", flush=True)


def main():
    tests()
    a = load_bars()
    piv = lambda v: a.pivot(index="Date", columns="Code", values=v).sort_index()
    Pd = piv("P")
    PO, PL, Va, Vo = (piv(v).reindex_like(Pd) for v in ("PO", "PL", "Va", "Vo"))
    del a
    idx, codes = Pd.index, Pd.columns
    prime = prime_mask(idx, codes)
    surv = survivors()
    liq = Va.rolling(20, min_periods=20).median()
    mom126 = Pd / Pd.shift(126) - 1
    mom12_1 = Pd.shift(21) / Pd.shift(252) - 1
    volsurge = Va.rolling(20, min_periods=20).mean() / Va.rolling(120, min_periods=100).mean()
    adv20 = Vo.rolling(20, min_periods=15).mean()
    margin = read_dir(DATA / "margin_interest", ["Date", "Code", "ShrtVol"])
    margin["Code"] = margin.Code.astype(str).map(normalize_code)
    margin["Date"] = pd.to_datetime(margin.Date, errors="coerce")
    margin["ShrtVol"] = pd.to_numeric(margin.ShrtVol, errors="coerce")
    margin = add_avail(margin.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True), idx)
    tpx = pd.read_parquet(DATA / "topix/topix_daily.parquet", columns=["Date", "C"])
    tc = pd.to_numeric(tpx.set_index(pd.to_datetime(tpx.Date)).C, errors="coerce").sort_index()
    t_ok = (tc >= tc.rolling(200, min_periods=200).mean()).reindex(idx).ffill()

    ym = pd.Series(idx.to_period("M"), index=idx)
    sig_days = np.flatnonzero(ym.ne(ym.shift(-1)).to_numpy())
    Pn, POn, PLn = Pd.to_numpy(), PO.to_numpy(), PL.to_numpy()
    variants = {k: [] for k in ("production", "production_open_buy", "production_survivors", "no_filters_top5",
                                "top10_filtered", "production_topix_ma200", "production_mom12_1")}
    info5, info10, bench, months, short_on = [], [], [], [], []
    col = {c: j for j, c in enumerate(codes)}
    for k, s in enumerate(sig_days[:-1]):
        d = idx[s]
        if s < 252 or d.year < 2017:
            continue
        nxt = sig_days[k + 1]          # last session of the holding month
        e, x = s + 1, nxt + 1          # buy next open; sell first open after the next month-end
        uni = codes[(prime.iloc[s] & (liq.iloc[s] >= LIQ_MIN) & mom126.iloc[s].notna()).to_numpy()]
        sd, asof = short_days_at(margin, d, uni, adv20.iloc[s])
        on = sd is not None
        short_on.append(on)
        frame = pd.DataFrame({"code": uni, "mom": mom126.iloc[s][uni].to_numpy(), "mom12_1": mom12_1.iloc[s][uni].to_numpy(),
                              "vs": volsurge.iloc[s][uni].to_numpy()})
        frame["spct"] = sd.rank(pct=True).to_numpy() if on else np.nan
        bad = frame.vs.isna() | (frame.vs >= VOLSURGE_MAX)
        if on:
            bad |= frame.spct > SHORT_PCT_MAX
        ok = frame[~bad]
        top = lambda f, n, key="mom": np.array([col[c] for c in f.sort_values([key, "code"], ascending=[False, True]).code.head(n)], int)
        variants["production"].append(top(ok, 5))
        variants["production_open_buy"].append(top(ok, 5))
        variants["production_survivors"].append(top(ok[ok.code.isin(surv)], 5))
        variants["no_filters_top5"].append(top(frame, 5))
        variants["top10_filtered"].append(top(ok, 10))
        variants["production_topix_ma200"].append(top(ok, 5) if bool(t_ok.iloc[s]) else np.array([], int))
        variants["production_mom12_1"].append(top(ok.dropna(subset=["mom12_1"]), 5, "mom12_1"))
        info5.append((s, e, x, 5))
        info10.append((s, e, x, 10))
        if x < len(idx):
            ui = np.array([col[c] for c in uni], int)
            o1, o2 = POn[e, ui], POn[x, ui]
            br = o2 / o1 - 1
            bench.append(np.nanmean(br))
        else:
            bench.append(np.nan)
        months.append(d)
    keep = ~np.isnan(np.array(bench))
    bench = np.array(bench)[keep]
    months = pd.DatetimeIndex(months)[keep]
    print(f"months {len(months)}: {months[0].date()}..{months[-1].date()}; short filter on {np.mean(short_on):.0%}", flush=True)

    out = {"settings": {"universe": "point-in-time Prime/TSE1 incl. delisted, 20d median Va >= 1e8",
                        "rule": "mom126 top N after volsurge<2 and short_days<=80th pct; limit buy at signal close next session",
                        "exit": "first open after the next month-end", "cost_roundtrip_pct": COST * 100,
                        "dd_note": "max drawdown measured on month-end equity (intra-month swings not included)"},
           "benchmark_equal_weight": stats(bench, months), "variants": {}}
    for name, sel in variants.items():
        sel = [s for s, k in zip(sel, keep) if k]
        info = [i for i, k in zip(info10 if name.startswith("top10") else info5, keep) if k]
        r, trades = simulate(sel, Pn, POn, PLn, info, limit=(name != "production_open_buy"))
        filled = [t for s, t in trades if s == "filled"]
        res = stats(r, months, bench)
        res["fills"] = {"filled": len(filled), "unfilled": sum(s == "unfilled" for s, _ in trades),
                        "trade_mean_pct": round(float(np.mean(filled) * 100), 2) if filled else None,
                        "trade_win_pct": round(float(np.mean(np.array(filled) > 0) * 100), 1) if filled else None,
                        "trades_down20_pct": round(float(np.mean(np.array(filled) <= -0.2) * 100), 1) if filled else None}
        out["variants"][name] = res
        print(name, res["all"], "| A", res["A_2017_2021"]["monthly_excess_pct"], res["A_2017_2021"]["excess_t"],
              "| B", res["B_2022_2026"]["monthly_excess_pct"], res["B_2022_2026"]["excess_t"], "| yrs", res["years_beating_benchmark"],
              "| fills", res["fills"], flush=True)
    print("benchmark", out["benchmark_equal_weight"]["all"], flush=True)
    print("production by year", out["variants"]["production"]["by_year_pct"], "bench", out["benchmark_equal_weight"]["by_year_pct"], flush=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
