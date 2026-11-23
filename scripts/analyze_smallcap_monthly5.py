"""Research: buy 5 Standard stocks every month from an earnings-based candidate pool, hold 6 or 12 months.

Universe / data / point-in-time rules: scripts/analyze_smallcap_fundamental_entry.py (Standard market from the
monthly snapshot, raw close >= 100 yen, 20-day median turnover >= 10M yen, delisted issues kept, disclosures
strictly before the formation date and <= 150 days old).
Candidate pool at each month-end (fixed in advance, from results/smallcap_fundamental_entry.json):
  pool_sales_top     Standard, stand-alone quarter sales yoy in the top 20% of Standard that month
  pool_sales_top_ex  same, excluding the bottom 20% by quarterly sales size (tiny companies)
Picks from pool_sales_top_ex:
  rule5    OP turnaround first, then the highest sales yoy (5 names)
  random5  5 names at random; 200 independent runs show how much luck decides the result
  pool_all every candidate equally weighted (what random picks earn on average)
Benchmark: every Standard universe stock, same schedule.
Mechanics: buy at the next session's open after month-end, equal weight inside a monthly tranche, buy and hold
for 126 (6m) or 252 (12m) sessions; an issue that stops trading is held at its last price. Tranches overlap
(6 or 12 live at once) and the portfolio averages the live tranches each day. 0.5% round-trip cost per tranche.
Output: results/smallcap_monthly5.json (aggregate statistics only).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_ma75_turn_trend import load_prices
from analyze_smallcap_bigwinners import MIN_HIST, MIN_PRICE, MIN_TURNOVER, market_panel
from analyze_smallcap_fundamental_entry import attach_fundamentals, load_fins, quarterly

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/smallcap_monthly5.json"
HOLDS = {"6m": 126, "12m": 252}
COST = 0.005
N_PICK = 5
N_RANDOM = 200
SPLIT = pd.Timestamp("2021-07-01")
SEED = 20261123


def tranche_returns(Pff, PO, e, codes_idx, hold):
    """Daily returns of an equal-weight buy-and-hold tranche from the open of day e for `hold` sessions."""
    end = min(e + hold, len(Pff) - 1)
    if e >= len(Pff) or len(codes_idx) == 0:
        return None
    entry = PO[e, codes_idx]
    ok = ~np.isnan(entry) & (entry > 0)
    if not ok.any():
        return None
    ci = codes_idx[ok]
    path = Pff[e:end + 1, ci] / entry[ok]
    path = np.where(np.isnan(path), 1.0, path)
    v = path.mean(axis=1)
    r = np.empty(len(v))
    r[0] = v[0] - 1 - COST
    r[1:] = v[1:] / v[:-1] - 1
    return r, v[-1] - 1 - COST, path[-1] - 1


def run_portfolio(n_days, tranches):
    """tranches: list of (start_index, daily_returns). Portfolio return = mean of live tranches each day."""
    s = np.zeros(n_days)
    c = np.zeros(n_days)
    for e, r in tranches:
        s[e:e + len(r)] += r
        c[e:e + len(r)] += 1
    return np.where(c > 0, s / np.maximum(c, 1), 0.0), c


def pstats(r, idx, start, end=None):
    m = (idx >= start) & ((idx < end) if end is not None else True)
    x = r[m]
    if len(x) < 100:
        return None
    eq = np.cumprod(1 + x)
    return {"cagr_pct": round(float(eq[-1] ** (245 / len(x)) - 1) * 100, 2),
            "max_dd_pct": round(float((eq / np.maximum.accumulate(eq) - 1).min()) * 100, 1),
            "ann_vol_pct": round(float(x.std() * np.sqrt(245) * 100), 1)}


def yearly(r, idx, start):
    s = pd.Series(r, index=idx)
    s = s[s.index >= start]
    return {str(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in s.groupby(s.index.year)}


def trade_stats(rets):
    v = pd.Series(rets).dropna()
    if v.empty:
        return {"n": 0}
    return {"n": int(len(v)), "mean_pct": round(float(v.mean() * 100), 1), "median_pct": round(float(v.median() * 100), 1),
            "doubled_pct": round(float((v >= 1).mean() * 100), 2), "down50_pct": round(float((v <= -0.5).mean() * 100), 1),
            "win_pct": round(float((v > 0).mean() * 100), 1)}


def tests():
    P = np.array([[10, 20], [11, 22], [12, 24], [12, 26]], dtype=float)
    PO = P.copy()
    r, total, per = tranche_returns(P, PO, 1, np.array([0, 1]), 2)
    v = np.array([1.0, (12 / 11 + 24 / 22) / 2, (12 / 11 + 26 / 22) / 2])
    assert abs(r[0] - (v[0] - 1 - COST)) < 1e-12 and abs(r[2] - (v[2] / v[1] - 1)) < 1e-12
    assert abs(total - (v[-1] - 1 - COST)) < 1e-12
    port, c = run_portfolio(5, [(0, np.array([0.1, 0.1])), (1, np.array([0.3]))])
    assert abs(port[1] - 0.2) < 1e-12 and port[4] == 0.0 and c[1] == 2
    print("tranche path, cost and overlapping-portfolio tests passed", flush=True)


def main():
    tests()
    a = load_prices()
    P = a.pivot(index="Date", columns="Code", values="P").sort_index()
    PO = a.pivot(index="Date", columns="Code", values="PO").reindex_like(P)
    raw = a.pivot(index="Date", columns="Code", values="C").reindex_like(P)
    to = a.pivot(index="Date", columns="Code", values="to").reindex_like(P)
    del a
    t20 = to.rolling(20, min_periods=15).median()
    hist = P.notna().cumsum()
    mkt = market_panel(set(P.columns))
    fs, _ = load_fins()
    q = quarterly(fs)
    col = {c: j for j, c in enumerate(P.columns)}
    Pff, POa = P.ffill().to_numpy(), PO.to_numpy()
    idx = P.index
    ym = pd.Series(idx.to_period("M"), index=idx)

    months = []
    for d in idx[ym.ne(ym.shift(-1)).to_numpy()]:
        i = idx.get_loc(d)
        if i < 252 or i + 1 >= len(idx):
            continue
        try:
            m = mkt.loc[pd.Period(d, "M")]
        except KeyError:
            continue
        codes = m[m == "Standard"].index.intersection(P.columns)
        ok = (raw.loc[d, codes] >= MIN_PRICE) & (t20.loc[d, codes] >= MIN_TURNOVER) & (hist.loc[d, codes] >= MIN_HIST)
        codes = codes[ok.to_numpy()]
        fr = pd.DataFrame({"Date": d, "Code": codes, "rawC": raw.loc[d, codes].to_numpy()})
        fr = attach_fundamentals(fr, q, None)
        months.append((i, d, fr))
    print(f"months {len(months)}: {months[0][1].date()}..{months[-1][1].date()}", flush=True)

    rng = np.random.default_rng(SEED)
    out = {"settings": {"pool": "Standard, q_sales_yoy top 20% within Standard that month; _ex drops bottom 20% quarterly sales size",
                        "rule5": "op_turnaround first, then highest q_sales_yoy", "random_runs": N_RANDOM,
                        "entry": "next open after month-end", "holds": list(HOLDS), "cost_roundtrip_pct": COST * 100,
                        "split": str(SPLIT.date())},
           "pool_size": {}, "results": {}}
    picks = {"benchmark_all_standard": [], "pool_sales_top": [], "pool_all": [], "rule5": []}
    rand_picks = [[] for _ in range(N_RANDOM)]
    sizes = []
    for i, d, fr in months:
        fy = fr.dropna(subset=["q_sales_yoy"])
        if len(fy) < 50:
            continue
        top = fy[fy.q_sales_yoy.rank(pct=True) > 0.8]
        size_cut = fy.q_sales_size.quantile(0.2)
        top_ex = top[top.q_sales_size > size_cut]
        sizes.append(len(top_ex))
        rule = top_ex.assign(turn=top_ex.op_turnaround.fillna(0)).sort_values(["turn", "q_sales_yoy"], ascending=False).head(N_PICK)
        e = i + 1
        to_idx = lambda s: np.array([col[c] for c in s], dtype=int)
        picks["benchmark_all_standard"].append((e, d, to_idx(fr.Code)))
        picks["pool_sales_top"].append((e, d, to_idx(top.Code)))
        picks["pool_all"].append((e, d, to_idx(top_ex.Code)))
        picks["rule5"].append((e, d, to_idx(rule.Code)))
        pool = to_idx(top_ex.Code)
        for k in range(N_RANDOM):
            rand_picks[k].append((e, d, rng.choice(pool, size=min(N_PICK, len(pool)), replace=False)))
    out["pool_size"] = {"median": float(np.median(sizes)), "min": int(np.min(sizes)), "max": int(np.max(sizes))}
    start = picks["rule5"][0][1]

    def evaluate(plist, hold):
        tr, totals, per_stock = [], [], []
        for e, d, ci in plist:
            res = tranche_returns(Pff, POa, e, ci, hold)
            if res is None:
                continue
            r, total, per = res
            tr.append((e, r))
            if e + hold < len(idx):
                totals.append(total)
                per_stock.extend(per.tolist())
        port, live = run_portfolio(len(idx), tr)
        return port, totals, per_stock

    for hname, hold in HOLDS.items():
        res = {}
        for name, plist in picks.items():
            port, totals, per_stock = evaluate(plist, hold)
            res[name] = {"all": pstats(port, idx, start), "before_2021_07": pstats(port, idx, start, SPLIT),
                         "from_2021_07": pstats(port, idx, SPLIT), "by_year_pct": yearly(port, idx, start),
                         "tranche_return": trade_stats(totals), "per_stock_return": trade_stats(per_stock)}
        cagr, cagr1, cagr2, dd = [], [], [], []
        for k in range(N_RANDOM):
            port, _, _ = evaluate(rand_picks[k], hold)
            s_all, s1, s2 = pstats(port, idx, start), pstats(port, idx, start, SPLIT), pstats(port, idx, SPLIT)
            cagr.append(s_all["cagr_pct"]); dd.append(s_all["max_dd_pct"]); cagr1.append(s1["cagr_pct"]); cagr2.append(s2["cagr_pct"])
        bench = res["benchmark_all_standard"]
        pct = lambda v: {p: round(float(np.percentile(v, p)), 2) for p in (5, 25, 50, 75, 95)}
        res["random5"] = {"cagr_pct_percentiles": pct(cagr), "max_dd_pct_percentiles": pct(dd),
                          "before_2021_07_cagr_percentiles": pct(cagr1), "from_2021_07_cagr_percentiles": pct(cagr2),
                          "share_beating_benchmark_all": round(float(np.mean(np.array(cagr) > bench["all"]["cagr_pct"])), 3),
                          "share_beating_benchmark_both_halves": round(float(np.mean(
                              (np.array(cagr1) > bench["before_2021_07"]["cagr_pct"]) & (np.array(cagr2) > bench["from_2021_07"]["cagr_pct"]))), 3)}
        out["results"][hname] = res
        for name in picks:
            r = res[name]
            print(hname, name, r["all"], "| <2021-07", r["before_2021_07"]["cagr_pct"], "| >=2021-07", r["from_2021_07"]["cagr_pct"],
                  "| stock", {k: r["per_stock_return"][k] for k in ("mean_pct", "median_pct", "doubled_pct", "down50_pct")}, flush=True)
        print(hname, "random5", res["random5"], flush=True)
        print(hname, "rule5 by year", res["rule5"]["by_year_pct"], "bench", bench["by_year_pct"], flush=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
