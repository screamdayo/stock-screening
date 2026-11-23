"""Research: Momentum6M entry method — market at the open vs limit orders above the signal close.

Selection, universe, exit, costs and benchmark are exactly the 'production' rule of
scripts/analyze_momentum6m_pit.py (point-in-time Prime incl. delisted, mom126 top 5 after the volume-surge and
short-interest exclusions, sell at the first open after the next month-end, 0.3% round trip, empty slot = cash).
Only the buy on the session after the signal changes:
  open                    market order at the open
  limit_+X%               limit = signal close x (1+X): fills at the open if the open is at/below it, at the limit if
                          the day's low reaches it, otherwise no buy (X = 0, 1, 2, 3, 5, 10)
  open_if_gap<=X%         market order at the open only if the open is at most X% above the signal close, otherwise
                          no buy (X = 2, 5) -- what a person checking the opening price would do
For every method the trades it skipped are also valued as if bought at the open, to show what was given up.
All methods were fixed before running; 9 methods are compared, so treat the best one's edge with caution.
Output: results/momentum6m_entry.json (aggregate statistics only).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_ma75_turn_trend import prime_mask, read_dir
from analyze_momentum6m_pit import (COST, LIQ_MIN, SHORT_PCT_MAX, VOLSURGE_MAX, add_avail, load_bars, short_days_at,
                                    stats)
from price_adjust import normalize_code

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results/momentum6m_entry.json"
METHODS = [("open", None, None)] + [(f"limit_+{x:g}%", "limit", x / 100) for x in (0, 1, 2, 3, 5, 10)] \
          + [(f"open_if_gap<={x:g}%", "gap", x / 100) for x in (2, 5)]


def exit_price(P, PO, e, x, j):
    n = len(P)
    if x < n and not np.isnan(PO[x, j]):
        return PO[x, j]
    seg = P[e:min(x, n - 1) + 1, j]
    return seg[~np.isnan(seg)][-1]


def simulate_entry(sel, P, PO, PL, info, mode, buf):
    rets, filled, skipped, gaps = [], [], [], []
    for (sig, e, x, slots), ci in zip(info, sel):
        r = 0.0
        for j in ci:
            op, lo, ref = PO[e, j], PL[e, j], P[sig, j]
            if np.isnan(op):
                continue
            gaps.append(op / ref - 1)
            out = exit_price(P, PO, e, x, j)
            if mode is None:
                px = op
            elif mode == "limit":
                lim = ref * (1 + buf)
                px = op if op <= lim else (lim if (not np.isnan(lo) and lo <= lim) else None)
            else:
                px = op if op <= ref * (1 + buf) else None
            if px is None:
                skipped.append(out / op - 1 - COST)
                continue
            tr = out / px - 1 - COST
            filled.append(tr)
            r += tr / slots
        rets.append(r)
    return np.array(rets), np.array(filled), np.array(skipped), np.array(gaps)


def tests():
    P = np.array([[100.0], [0], [0], [0]]); P[1:, 0] = [104, 106, 108]
    PO = np.array([[100.0], [103.0], [105.0], [110.0]])
    PL = np.array([[99.0], [101.5], [104.0], [107.0]])
    info = [(0, 1, 3, 1)]
    r, f, s, g = simulate_entry([np.array([0])], P, PO, PL, info, "limit", 0.02)
    assert abs(f[0] - (110 / 102 - 1 - COST)) < 1e-12           # open 103 > 102, low 101.5 reaches it
    r, f, s, g = simulate_entry([np.array([0])], P, PO, PL, info, "gap", 0.02)
    assert len(f) == 0 and abs(s[0] - (110 / 103 - 1 - COST)) < 1e-12   # gap 3% > 2%: skipped, valued at open
    r, f, s, g = simulate_entry([np.array([0])], P, PO, PL, info, None, None)
    assert abs(f[0] - (110 / 103 - 1 - COST)) < 1e-12 and abs(g[0] - 0.03) < 1e-12
    print("limit buffer, gap rule and skipped-trade valuation tests passed", flush=True)


def main():
    tests()
    a = load_bars()
    piv = lambda v: a.pivot(index="Date", columns="Code", values=v).sort_index()
    Pd = piv("P")
    PO, PL, Va, Vo = (piv(v).reindex_like(Pd) for v in ("PO", "PL", "Va", "Vo"))
    del a
    idx, codes = Pd.index, Pd.columns
    prime = prime_mask(idx, codes)
    liq = Va.rolling(20, min_periods=20).median()
    mom126 = Pd / Pd.shift(126) - 1
    volsurge = Va.rolling(20, min_periods=20).mean() / Va.rolling(120, min_periods=100).mean()
    adv20 = Vo.rolling(20, min_periods=15).mean()
    margin = read_dir(DATA / "margin_interest", ["Date", "Code", "ShrtVol"])
    margin["Code"] = margin.Code.astype(str).map(normalize_code)
    margin["Date"] = pd.to_datetime(margin.Date, errors="coerce")
    margin["ShrtVol"] = pd.to_numeric(margin.ShrtVol, errors="coerce")
    margin = add_avail(margin.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True), idx)

    ym = pd.Series(idx.to_period("M"), index=idx)
    sig_days = np.flatnonzero(ym.ne(ym.shift(-1)).to_numpy())
    Pn, POn, PLn = Pd.to_numpy(), PO.to_numpy(), PL.to_numpy()
    col = {c: j for j, c in enumerate(codes)}
    sel, info, bench, months = [], [], [], []
    for k, s in enumerate(sig_days[:-1]):
        d = idx[s]
        if s < 252 or d.year < 2017:
            continue
        e, x = s + 1, sig_days[k + 1] + 1
        if x >= len(idx):
            continue
        uni = codes[(prime.iloc[s] & (liq.iloc[s] >= LIQ_MIN) & mom126.iloc[s].notna()).to_numpy()]
        sd, _ = short_days_at(margin, d, uni, adv20.iloc[s])
        f = pd.DataFrame({"code": uni, "mom": mom126.iloc[s][uni].to_numpy(), "vs": volsurge.iloc[s][uni].to_numpy()})
        bad = f.vs.isna() | (f.vs >= VOLSURGE_MAX)
        if sd is not None:
            bad |= (sd.rank(pct=True).to_numpy() > SHORT_PCT_MAX)
        ok = f[~bad].sort_values(["mom", "code"], ascending=[False, True]).head(5)
        sel.append(np.array([col[c] for c in ok.code], int))
        info.append((s, e, x, 5))
        ui = np.array([col[c] for c in uni], int)
        bench.append(np.nanmean(POn[x, ui] / POn[e, ui] - 1))
        months.append(d)
    bench, months = np.array(bench), pd.DatetimeIndex(months)
    print(f"months {len(months)}: {months[0].date()}..{months[-1].date()}", flush=True)

    out = {"settings": {"selection": "production rule of analyze_momentum6m_pit.py", "methods": [m[0] for m in METHODS],
                        "warning": "9 entry methods compared on the same data; discount the best one"},
           "benchmark_equal_weight": stats(bench, months), "methods": {}}
    gaps_all = None
    for name, mode, buf in METHODS:
        r, f, sk, g = simulate_entry(sel, Pn, POn, PLn, info, mode, buf)
        if gaps_all is None:
            gaps_all = g
        res = stats(r, months, bench)
        res["trades"] = {"filled": int(len(f)), "skipped": int(len(sk)),
                         "skip_rate_pct": round(float(len(sk) / max(1, len(f) + len(sk)) * 100), 1),
                         "filled_mean_pct": round(float(f.mean() * 100), 2) if len(f) else None,
                         "skipped_mean_if_bought_at_open_pct": round(float(sk.mean() * 100), 2) if len(sk) else None,
                         "filled_down20_pct": round(float((f <= -0.2).mean() * 100), 1) if len(f) else None}
        out["methods"][name] = res
        print(name, res["all"], "| A", res["A_2017_2021"]["cagr_pct"], res["A_2017_2021"]["monthly_excess_pct"],
              "| B", res["B_2022_2026"]["cagr_pct"], res["B_2022_2026"]["monthly_excess_pct"], "| yrs",
              res["years_beating_benchmark"], "|", res["trades"], flush=True)
    q = np.percentile(gaps_all * 100, [10, 25, 50, 75, 90])
    out["opening_gap_vs_signal_close_pct"] = {"p10": round(q[0], 2), "p25": round(q[1], 2), "median": round(q[2], 2),
                                              "p75": round(q[3], 2), "p90": round(q[4], 2),
                                              "share_above_0": round(float((gaps_all > 0).mean()), 3),
                                              "share_above_2pct": round(float((gaps_all > 0.02).mean()), 3),
                                              "share_above_5pct": round(float((gaps_all > 0.05).mean()), 3)}
    print("gaps", out["opening_gap_vs_signal_close_pct"], flush=True)
    print("benchmark", out["benchmark_equal_weight"]["all"], flush=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
