"""Research: buy when MA75 turns up (20-session comparison), sell when it turns down. 10 years, no survivor bias.

Inputs (private J-Quants archive mounted under data/, see .github/workflows/ma75_turn_trend.yml):
  data/batches, data/daily_updates, data/delisted, data/master_history, data/topix
Variants (results/ma75_turn_trend_variants.json): TOPIX MA75 rising filter, and buying when the slope crosses
+0.5/+1/+2% instead of 0; the exit is unchanged.
Output: results/ma75_turn_trend.json (aggregate statistics only; no per-issue data).

Rule
- slope[t] = MA75[t] / MA75[t-20] - 1 (adjusted close, MA includes the signal day)
- Buy signal: slope[t-1] <= 0 and slope[t] > 0. Enter at the next session's open.
- Sell signal: slope[t-1] > 0 and slope[t] <= 0. Exit at the next session's open.
- An issue that stops trading before a sell signal is closed at its last close (delisted issues kept).
- Trades still open at the end are reported separately and marked at the last close.
Universe at the buy signal: point-in-time Prime (TSE 1st before 2022-04), common stock, close >= 100 yen,
20-day median turnover (C*Vo) >= 100M yen.

Portfolio view (approximation): every open trade equally weighted, close-to-close returns from the close
of the entry day to the close of the exit day, cash when nothing is held; compared with the equal-weight
average of the same universe. No fees, dividends or capital limits; round-trip cost sensitivity shown.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from price_adjust import normalize_code, stale_basis_multiplier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results/ma75_turn_trend.json"
OUT_VARIANTS = ROOT / "results/ma75_turn_trend_variants.json"
THRESHOLDS = (0.005, 0.01, 0.02)
PRICE_COLS = ["Date", "Code", "O", "C", "Vo", "AdjO", "AdjC", "AdjFactor"]
PRIME_NAMES = {"東証一部", "プライム"}
MA_LEN, SLOPE_LAG = 75, 20
LIQ_MIN_YEN, MIN_PRICE = 100_000_000, 100
COSTS = (0.002, 0.005)


def read_dir(path, columns=None):
    files = sorted(Path(path).glob("*.parquet"))
    return pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)


def load_prices():
    dm = pd.read_parquet(DATA / "delisted/delisted_master.parquet")
    dm = dm[(dm.Code.astype(str).str.len() == 4) & (dm.ProdCat.astype(str) == "011")]
    dl = pd.concat([pd.read_parquet(f, columns=PRICE_COLS) for f in sorted((DATA / "delisted").glob("batch_*.parquet"))],
                   ignore_index=True)
    dl["Code"] = dl.Code.astype(str).map(normalize_code)
    dl = dl[dl.Code.isin(set(dm.Code.astype(str)))].assign(src=2)
    a = pd.concat([read_dir(DATA / "batches", PRICE_COLS).assign(src=0),
                   read_dir(DATA / "daily_updates", PRICE_COLS).assign(src=1)], ignore_index=True)
    a["Code"] = a.Code.astype(str).map(normalize_code)
    a = a[~a.Code.isin(set(dl.Code))]  # issues delisted after the archive: use their full fresh series
    a = pd.concat([a, dl], ignore_index=True)
    a["Date"] = pd.to_datetime(a.Date)
    for c in ("O", "C", "Vo", "AdjO", "AdjC", "AdjFactor"):
        a[c] = pd.to_numeric(a[c], errors="coerce")
    a = a.sort_values(["Code", "Date", "src"]).drop_duplicates(["Code", "Date"], keep="last").reset_index(drop=True)
    m = stale_basis_multiplier(a)
    a["P"] = a.AdjC * m
    a["PO"] = a.AdjO * m
    a["to"] = a.C * a.Vo
    a = a[(a.P > 0) & (a.PO > 0)].reset_index(drop=True)
    print(f"daily rows {len(a):,}, issues {a.Code.nunique():,}, delisted issues {dl.Code.nunique():,}", flush=True)
    return a


def prime_mask(dates, codes):
    mh = read_dir(DATA / "master_history", ["Code", "MktNm", "ProdCat", "SnapshotDate"])
    mh["Code"] = mh.Code.astype(str).map(normalize_code)
    mh = mh[(mh.ProdCat.astype(str) == "011") & mh.MktNm.isin(PRIME_NAMES)]
    mh["ym"] = pd.to_datetime(mh.SnapshotDate).dt.to_period("M")
    per_month = mh.groupby("ym").Code.apply(set)
    assert per_month.map(len).min() > 1000, "Prime snapshot unexpectedly small"
    ym = pd.Series(dates.to_period("M"), index=dates)
    months = sorted(per_month.index)
    out = np.zeros((len(dates), len(codes)), dtype=bool)
    col = {c: j for j, c in enumerate(codes)}
    for i, p in enumerate(ym):
        k = p if p in per_month.index else max([m for m in months if m <= p], default=months[0])
        for c in per_month[k]:
            j = col.get(c)
            if j is not None:
                out[i, j] = True
    return pd.DataFrame(out, index=dates, columns=codes)


def slope_of(close):
    ma = close.rolling(MA_LEN, min_periods=MA_LEN).mean()
    return ma / ma.shift(SLOPE_LAG) - 1


def build_trades(close, opn, slope, eligible, buy=None):
    """Walk each issue: buy on a buy signal on an eligible day, sell after the next MA75 down-turn."""
    up = (((slope.shift(1) <= 0) & (slope > 0)) if buy is None else buy).to_numpy()
    down = ((slope.shift(1) > 0) & (slope <= 0)).to_numpy()
    el = eligible.to_numpy()
    C, O = close.to_numpy(), opn.to_numpy()
    n = len(close)
    last_global = n - 1
    trades = []
    for j in range(C.shape[1]):
        valid = np.flatnonzero(~np.isnan(C[:, j]))
        if not len(valid):
            continue
        last = valid[-1]
        delisted = last < last_global - 7
        i = 0
        ups = np.flatnonzero(up[:, j] & el[:, j])
        downs = np.flatnonzero(down[:, j])
        for t in ups:
            if t < i or t + 1 > last:
                continue
            e = t + 1
            if np.isnan(O[e, j]):
                continue
            nxt = downs[downs > t]
            if len(nxt) and nxt[0] + 1 <= last:
                x, px, status = nxt[0] + 1, O[nxt[0] + 1, j], "closed"
                if np.isnan(px):
                    px = C[nxt[0], j]
            elif delisted:
                x, px, status = last, C[last, j], "delisted"
            else:
                x, px, status = last, C[last, j], "open"
            trades.append((j, t, e, x, O[e, j], px, status))
            i = x + 1
    tr = pd.DataFrame(trades, columns=["j", "signal", "entry", "exit", "p_in", "p_out", "status"])
    tr["ret"] = tr.p_out / tr.p_in - 1
    tr["days"] = tr.exit - tr.entry
    tr["entry_date"] = close.index[tr.entry]
    return tr


def summarize(r, days=None):
    r = r.dropna()
    if r.empty:
        return {"n": 0}
    w, l = r[r > 0], r[r <= 0]
    out = {"n": int(len(r)), "mean_pct": round(r.mean() * 100, 2), "median_pct": round(r.median() * 100, 2),
           "win_rate": round(float((r > 0).mean()), 3),
           "avg_win_pct": round(w.mean() * 100, 2) if len(w) else None,
           "avg_loss_pct": round(l.mean() * 100, 2) if len(l) else None,
           "profit_factor": round(float(w.sum() / -l.sum()), 2) if l.sum() < 0 else None}
    if days is not None:
        out["median_days"] = float(days.loc[r.index].median())
        out["mean_days"] = round(float(days.loc[r.index].mean()), 1)
    return out


def portfolio(close, tr, universe):
    ret = close.pct_change(fill_method=None)
    held = np.zeros(close.shape, dtype=bool)
    for j, e, x in tr[["j", "entry", "exit"]].itertuples(index=False):
        held[e + 1:x + 1, j] = True  # close-to-close days after the entry day, through the exit day
    h = pd.DataFrame(held, index=close.index, columns=close.columns)
    strat = ret.where(h).mean(axis=1).fillna(0.0)
    bench = ret.where(universe.shift(1, fill_value=False)).mean(axis=1).fillna(0.0)
    start = MA_LEN + SLOPE_LAG + 1

    def stats(s):
        s = s.iloc[start:]
        eq = (1 + s).cumprod()
        yrs = len(s) / 245
        dd = (eq / eq.cummax() - 1).min()
        yearly = (1 + s).groupby(s.index.year).prod() - 1
        return {"cagr_pct": round((eq.iloc[-1] ** (1 / yrs) - 1) * 100, 2), "max_dd_pct": round(dd * 100, 1),
                "ann_vol_pct": round(s.std() * np.sqrt(245) * 100, 1),
                "by_year_pct": {str(y): round(v * 100, 1) for y, v in yearly.items()}}

    return {"strategy": stats(strat), "universe_equal_weight": stats(bench),
            "avg_positions": round(float(h.sum(axis=1).iloc[start:].mean()), 1),
            "invested_share_of_days": round(float((h.sum(axis=1).iloc[start:] > 0).mean()), 3)}


def tests():
    idx = pd.bdate_range("2020-01-01", periods=900)
    x = np.arange(900.0)
    p = 100 + 20 * np.sin(x / 40)  # slow cycle: several up/down turns
    close = pd.DataFrame({"A": p}, index=idx)
    opn = close * 1.0
    s = slope_of(close)
    i = 150
    assert abs(s.A.iloc[i] - close.A.iloc[i - 74:i + 1].mean() / close.A.iloc[i - 94:i - 19].mean() + 1) < 1e-12
    tr = build_trades(close, opn, s, pd.DataFrame(True, index=idx, columns=["A"]))
    assert len(tr) >= 2 and (tr.entry == tr.signal + 1).all()
    for r in tr[tr.status == "closed"].itertuples():
        assert s.A.iloc[r.signal - 1] <= 0 < s.A.iloc[r.signal]
        assert s.A.iloc[r.exit - 2] > 0 >= s.A.iloc[r.exit - 1]
        assert abs(r.p_out - opn.A.iloc[r.exit]) < 1e-12
    assert (tr.entry.iloc[1:].to_numpy() > tr.exit.iloc[:-1].to_numpy()).all()  # no overlapping trades
    # delisting: data stops while holding -> closed at last close
    c2 = close.copy(); c2.iloc[tr.entry.iloc[0] + 5:] = np.nan
    tr2 = build_trades(c2, c2 * 1.0, slope_of(c2), pd.DataFrame(True, index=idx, columns=["A"]))
    assert tr2.status.iloc[0] == "delisted" and abs(tr2.p_out.iloc[0] - c2.A.dropna().iloc[-1]) < 1e-12
    print("slope, turn entry/exit, no-overlap and delisting tests passed", flush=True)


def main():
    tests()
    a = load_prices()
    close = a.pivot(index="Date", columns="Code", values="P").sort_index()
    opn = a.pivot(index="Date", columns="Code", values="PO").reindex_like(close)
    to = a.pivot(index="Date", columns="Code", values="to").reindex_like(close)
    del a
    prime = prime_mask(close.index, close.columns)
    liq = to.rolling(20, min_periods=15).median() >= LIQ_MIN_YEN
    universe = prime & liq & (close >= MIN_PRICE)
    print(f"universe/day median {int(universe.sum(axis=1).median())}, {close.index[0].date()}..{close.index[-1].date()}",
          flush=True)

    slope = slope_of(close)
    tr = build_trades(close, opn, slope, universe)
    done = tr[tr.status != "open"]
    days = tr.days
    buckets = pd.cut(done.days, [0, 10, 40, 120, 10**6], labels=["1-10d", "11-40d", "41-120d", "121d+"])
    tot = done.ret.sum()
    top5 = done.ret.nlargest(max(1, len(done) // 20)).sum()

    out = {
        "settings": {"rule": "buy: slope20 of MA75 crosses above 0, next open; sell: crosses to <= 0, next open",
                     "universe": "point-in-time Prime/TSE1, common stock, close>=100, 20d median turnover>=1e8, incl. delisted",
                     "period": [str(close.index[0].date()), str(close.index[-1].date())]},
        "trades_all_closed": summarize(done.ret, days),
        "trades_with_cost": {f"{c*100:.1f}pct_roundtrip": summarize(done.ret - c) for c in COSTS},
        "delisted_exits": summarize(done[done.status == "delisted"].ret),
        "still_open_marked": summarize(tr[tr.status == "open"].ret, days),
        "by_holding_period": {str(k): {**summarize(g.ret), "share_of_trades": round(len(g) / len(done), 3)}
                              for k, g in done.groupby(buckets, observed=True)},
        "by_entry_year": {str(y): summarize(g.ret, days) for y, g in done.groupby(done.entry_date.dt.year)},
        "top5pct_trades_share_of_total_return": round(float(top5 / tot), 2) if tot > 0 else None,
        "portfolio_equal_weight_all_signals": portfolio(close, tr, universe),
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))

    # ---- variants: same exit (MA75 turns down), filtered / delayed buys ----
    tpx = pd.read_parquet(DATA / "topix/topix_daily.parquet", columns=["Date", "C"])
    tpx["Date"] = pd.to_datetime(tpx.Date)
    tc = pd.to_numeric(tpx.set_index("Date").C, errors="coerce").sort_index()
    print(f"TOPIX {tc.index[0].date()}..{tc.index[-1].date()}", flush=True)
    t_ma = tc.rolling(MA_LEN, min_periods=MA_LEN).mean()
    t_slope = (t_ma / t_ma.shift(SLOPE_LAG) - 1).reindex(close.index)
    if t_slope.index[-1] > tc.index[-1]:
        print("TOPIX ends before prices: TOPIX-filtered variants cannot buy on later days", flush=True)
    topix_up = pd.DataFrame(np.repeat((t_slope > 0).to_numpy()[:, None], close.shape[1], axis=1),
                            index=close.index, columns=close.columns)
    turn = (slope.shift(1) <= 0) & (slope > 0)
    variants = {"base_turn_up": turn, "topix_ma75_up": turn & topix_up}
    for th in THRESHOLDS:
        cross = (slope.shift(1) <= th) & (slope > th)
        variants[f"slope_cross_{th*100:g}pct"] = cross
        variants[f"slope_cross_{th*100:g}pct_and_topix_up"] = cross & topix_up
    vout = {"settings": {"exit": "MA75 slope20 crosses to <= 0, next open (same for all)",
                         "topix_filter": "TOPIX MA75 slope20 > 0 on the signal day",
                         "slope_cross": "stock slope20 crosses above the threshold (instead of above 0)",
                         "universe": out["settings"]["universe"], "period": out["settings"]["period"]},
            "variants": {}}
    for name, buy in variants.items():
        t = build_trades(close, opn, slope, universe, buy)
        d_ = t[t.status != "open"]
        tot_ = d_.ret.sum()
        pf_ = portfolio(close, t, universe)
        vout["variants"][name] = {
            "trades": summarize(d_.ret, t.days),
            "mean_after_0_5pct_cost": round((d_.ret.mean() - 0.005) * 100, 2),
            "short_trades_le40d_share": round(float((d_.days <= 40).mean()), 3),
            "top5pct_share_of_total": round(float(d_.ret.nlargest(max(1, len(d_) // 20)).sum() / tot_), 2) if tot_ > 0 else None,
            "by_entry_year": {str(y): {"n": int(len(g)), "mean_pct": round(g.ret.mean() * 100, 2),
                                       "win": round(float((g.ret > 0).mean()), 3)}
                              for y, g in d_.groupby(d_.entry_date.dt.year)},
            "portfolio": {"cagr_pct": pf_["strategy"]["cagr_pct"], "max_dd_pct": pf_["strategy"]["max_dd_pct"],
                          "invested_share_of_days": pf_["invested_share_of_days"], "avg_positions": pf_["avg_positions"],
                          "by_year_pct": pf_["strategy"]["by_year_pct"]},
        }
        v = vout["variants"][name]
        print(name, v["trades"]["n"], v["trades"]["mean_pct"], v["trades"]["win_rate"], v["trades"].get("profit_factor"),
              "cagr", v["portfolio"]["cagr_pct"], "dd", v["portfolio"]["max_dd_pct"], flush=True)
    vout["universe_equal_weight"] = {k: out["portfolio_equal_weight_all_signals"]["universe_equal_weight"][k]
                                     for k in ("cagr_pct", "max_dd_pct", "by_year_pct")}
    OUT_VARIANTS.write_text(json.dumps(vout, ensure_ascii=False, indent=1))
    print(f"wrote {OUT_VARIANTS}", flush=True)
    print(json.dumps({k: out[k] for k in ("trades_all_closed", "trades_with_cost", "by_holding_period")},
                     ensure_ascii=False, indent=1))
    pf = out["portfolio_equal_weight_all_signals"]
    print("portfolio", {k: pf["strategy"][k] for k in ("cagr_pct", "max_dd_pct")},
          "universe", {k: pf["universe_equal_weight"][k] for k in ("cagr_pct", "max_dd_pct")},
          "avg positions", pf["avg_positions"], flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
