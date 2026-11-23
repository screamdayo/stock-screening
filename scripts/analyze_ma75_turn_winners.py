"""Research: what do the big winners of the MA75 turn-up / turn-down rule have in common at the buy signal?

Trades: same as results/ma75_turn_trend.json base rule (buy after MA75 slope20 turns up, next open; sell after it
turns down, next open; point-in-time Prime incl. delisted, turnover >= 100M yen). Closed + delisted trades only.
Winner = top 10% of trade returns. Every feature uses data up to the signal day's close only.
Output: results/ma75_turn_winners.json (aggregate statistics only; no per-issue data).

For each feature: quintile bins over all trades -> trade count, winner rate, mean / median return; plus the
same winner-rate spread (top bin minus bottom bin) in 2016-2021 and 2022-2026 entries separately, so a pattern
that only exists in one half is visible.
Caution: a feature that merely lengthens the holding period (e.g. starting far above MA75) raises raw returns
and the winner rate even for random prices, because the market drifts upward over a longer hold. The primary
measure is therefore excess return per holding day (bp) versus the equal-weight universe over the same window.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_ma75_turn_trend import (DATA, LIQ_MIN_YEN, MIN_PRICE, PRIME_NAMES, build_trades, load_prices,
                                     prime_mask, read_dir, slope_of)
from price_adjust import normalize_code

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/ma75_turn_winners.json"
WIN_Q = 0.90
SPLIT_YEAR = 2022


def at(frame, tr):
    return frame.to_numpy()[tr.signal.to_numpy(), tr.j.to_numpy()]


def numeric_features(close, to, slope, tr, raw):
    f = {}
    for n in (20, 60, 120, 250):
        f[f"ret{n}_pct"] = at(close / close.shift(n) - 1, tr) * 100
    f["from_250d_high_pct"] = at(close / close.rolling(250, min_periods=200).max() - 1, tr) * 100
    f["from_250d_low_pct"] = at(close / close.rolling(250, min_periods=200).min() - 1, tr) * 100
    ma75 = close.rolling(75, min_periods=75).mean()
    ma25 = close.rolling(25, min_periods=25).mean()
    f["close_vs_ma75_pct"] = at(close / ma75 - 1, tr) * 100
    f["ma25_vs_ma75_pct"] = at(ma25 / ma75 - 1, tr) * 100
    f["slope20_min_prev60_pct"] = at(slope.rolling(60, min_periods=40).min(), tr) * 100  # depth of prior downtrend
    r = close.pct_change(fill_method=None)
    f["vol60_daily_pct"] = at(r.rolling(60, min_periods=50).std(), tr) * 100
    f["turnover20_median_oku"] = at(to.rolling(20, min_periods=15).median(), tr) / 1e8
    f["turnover_ratio_20_vs_120"] = at(to.rolling(20, min_periods=15).mean() / to.rolling(120, min_periods=90).mean(), tr)
    f["price_yen_raw"] = at(raw, tr)  # unadjusted close: adjusted prices leak later splits (big winners split)
    # how long the preceding MA75 downtrend lasted (sessions since slope was last > 0 before the turn)
    pos = (slope > 0).to_numpy()
    last_pos = np.full(pos.shape, np.nan)
    idx = np.arange(pos.shape[0])[:, None].repeat(pos.shape[1], axis=1).astype(float)
    last_pos = pd.DataFrame(np.where(pos, idx, np.nan)).ffill().to_numpy()
    s, j = tr.signal.to_numpy(), tr.j.to_numpy()
    prev = np.array([last_pos[i - 1, c] if i > 0 else np.nan for i, c in zip(s, j)])
    f["downtrend_sessions"] = s - prev
    return f


def market_features(close, slope, tr, universe):
    f = {}
    tpx = pd.read_parquet(DATA / "topix/topix_daily.parquet", columns=["Date", "C"])
    tc = pd.to_numeric(tpx.set_index(pd.to_datetime(tpx.Date)).C, errors="coerce").sort_index().reindex(close.index).ffill()
    tma = tc.rolling(75, min_periods=75).mean()
    s = tr.signal.to_numpy()
    f["topix_slope20_ma75_pct"] = ((tma / tma.shift(20) - 1) * 100).to_numpy()[s]
    f["topix_ret60_pct"] = ((tc / tc.shift(60) - 1) * 100).to_numpy()[s]
    turn = ((slope.shift(1) <= 0) & (slope > 0) & universe).sum(axis=1)
    f["same_day_turn_ups"] = turn.to_numpy()[s]
    return f


def categorical_features(close, tr):
    mh = read_dir(DATA / "master_history")
    mh["Code"] = mh.Code.astype(str).map(normalize_code)
    mh = mh[(mh.ProdCat.astype(str) == "011") & mh.MktNm.isin(PRIME_NAMES)]
    mh["ym"] = pd.to_datetime(mh.SnapshotDate).dt.to_period("M")
    mh = mh.drop_duplicates(["Code", "ym"], keep="last").set_index(["Code", "ym"])
    key = pd.MultiIndex.from_arrays([close.columns[tr.j].astype(str), close.index[tr.signal].to_period("M")])
    out = {}
    for col in ("S33Nm", "S17Nm", "ScaleCat"):
        if col in mh.columns:
            out[col] = mh[col].reindex(key).astype(str).str.strip().replace({"nan": "unknown", "None": "unknown", "-": "unknown"}).to_numpy()
    return out


def describe_numeric(x, ret, win, year, exc=None, days=None):
    exc = np.zeros(len(x)) if exc is None else exc
    days = np.ones(len(x)) if days is None else days
    df = pd.DataFrame({"x": x, "ret": ret, "win": win, "year": year, "exc": exc, "days": days}).dropna()
    df["bin"] = pd.qcut(df.x.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    edges = df.groupby("bin", observed=True).x.agg(["min", "max"])
    tab = {}
    for b, g in df.groupby("bin", observed=True):
        tab[str(b)] = {"range": [round(float(edges.loc[b, "min"]), 3), round(float(edges.loc[b, "max"]), 3)],
                       "n": int(len(g)), "winner_rate": round(float(g.win.mean()), 3),
                       "mean_ret_pct": round(float(g.ret.mean() * 100), 2),
                       "median_ret_pct": round(float(g.ret.median() * 100), 2),
                       "mean_days": round(float(g.days.mean()), 1),
                       "mean_excess_pct": round(float(g.exc.mean() * 100), 2),
                       "excess_per_day_bp": round(float(g.exc.sum() / g.days.clip(lower=1).sum() * 1e4), 2)}

    def spread(d):
        if d.empty:
            return None
        r = d.groupby("bin", observed=True).win.mean()
        return round(float(r.get(5, np.nan) - r.get(1, np.nan)), 3)

    def spread_epd(d):
        if d.empty:
            return None
        r = d.groupby("bin", observed=True).apply(lambda g: g.exc.sum() / g.days.clip(lower=1).sum() * 1e4,
                                                    include_groups=False)
        return round(float(r.get(5, np.nan) - r.get(1, np.nan)), 2)

    early, late = df[df.year < SPLIT_YEAR], df[df.year >= SPLIT_YEAR]
    e1, e2 = spread_epd(early), spread_epd(late)
    return {"bins": tab,
            "excess_per_day_bp_top_minus_bottom": spread_epd(df),
            "epd_spread_2016_2021": e1, "epd_spread_2022_2026": e2,
            "epd_consistent_sign": bool(np.sign(e1 or 0) == np.sign(e2 or 0) != 0),
            "winner_median": round(float(df[df.win].x.median()), 3),
            "others_median": round(float(df[~df.win].x.median()), 3),
            "winner_rate_top_minus_bottom": spread(df),
            "spread_2016_2021": spread(early), "spread_2022_2026": spread(late),
            "consistent_sign": bool(np.sign(spread(early) or 0) == np.sign(spread(late) or 0) != 0)}


def describe_categorical(x, ret, win, exc, days, min_n=100):
    df = pd.DataFrame({"x": x, "ret": ret, "win": win, "exc": exc, "days": days})
    g = df.groupby("x").agg(n=("win", "size"), winner_rate=("win", "mean"), mean_ret=("ret", "mean"),
                            exc_sum=("exc", "sum"), days_sum=("days", "sum"), mean_days=("days", "mean"))
    g["epd"] = g.exc_sum / g.days_sum.clip(lower=1) * 1e4
    g = g[g.n >= min_n].sort_values("epd", ascending=False)
    return {k: {"n": int(v.n), "winner_rate": round(float(v.winner_rate), 3), "mean_ret_pct": round(float(v.mean_ret * 100), 2),
                "mean_days": round(float(v.mean_days), 1), "excess_per_day_bp": round(float(v.epd), 2)}
            for k, v in g.iterrows()}


def tests():
    df = pd.DataFrame({"x": np.arange(100.0)})
    r = describe_numeric(df.x.to_numpy(), np.where(df.x > 89, 0.5, 0.0), (df.x > 89).to_numpy(),
                         np.where(df.x < 50, 2018, 2024))
    assert r["bins"]["5"]["winner_rate"] == 0.5 and r["bins"]["1"]["winner_rate"] == 0.0
    assert r["winner_rate_top_minus_bottom"] == 0.5
    close = pd.DataFrame({"A": np.arange(1.0, 11.0)})
    tr = pd.DataFrame({"signal": [5], "j": [0]})
    assert at(close / close.shift(2) - 1, tr)[0] == 6 / 4 - 1
    print("binning and point-in-time lookup tests passed", flush=True)


def main():
    tests()
    a = load_prices()
    close = a.pivot(index="Date", columns="Code", values="P").sort_index()
    opn = a.pivot(index="Date", columns="Code", values="PO").reindex_like(close)
    to = a.pivot(index="Date", columns="Code", values="to").reindex_like(close)
    raw = a.pivot(index="Date", columns="Code", values="C").reindex_like(close)
    del a
    universe = prime_mask(close.index, close.columns) & (to.rolling(20, min_periods=15).median() >= LIQ_MIN_YEN) \
        & (close >= MIN_PRICE)
    slope = slope_of(close)
    tr = build_trades(close, opn, slope, universe)
    tr = tr[tr.status != "open"].reset_index(drop=True)
    cut = tr.ret.quantile(WIN_Q)
    win = (tr.ret >= cut).to_numpy()
    year = tr.entry_date.dt.year.to_numpy()
    ret = tr.ret.to_numpy()
    print(f"trades {len(tr):,}; winner cutoff +{cut*100:.1f}%", flush=True)
    # equal-weight universe index (close-to-close) to remove market drift over each trade's holding window
    bench = close.pct_change(fill_method=None).where(universe.shift(1, fill_value=False)).mean(axis=1).fillna(0.0)
    idx = (1 + bench).cumprod().to_numpy()
    exc = (1 + ret) / (idx[tr.exit.to_numpy()] / idx[tr.entry.to_numpy() - 1]) - 1
    days = tr.days.to_numpy().astype(float)

    feats = {**numeric_features(close, to, slope, tr, raw), **market_features(close, slope, tr, universe)}
    out = {"settings": {"trades": "MA75 turn-up buy / turn-down sell, closed + delisted",
                        "winner": f"top {int((1-WIN_Q)*100)}% of trade returns (>= {cut*100:.1f}%)",
                        "features": "known at the buy signal close; quintiles over all trades",
                        "split": f"entries before {SPLIT_YEAR} vs from {SPLIT_YEAR}"},
           "n_trades": int(len(tr)), "winner_cutoff_pct": round(float(cut * 100), 1),
           "winner_holding_days_median": float(tr.days[win].median()),
           "others_holding_days_median": float(tr.days[~win].median()),
           "winner_share_by_entry_year": {str(y): round(float(win[year == y].mean()), 3) for y in sorted(set(year))},
           "numeric": {}, "categorical": {}}
    for k, x in feats.items():
        out["numeric"][k] = describe_numeric(np.asarray(x, dtype=float), ret, win, year, exc, days)
    for k, x in categorical_features(close, tr).items():
        out["categorical"][k] = describe_categorical(x, ret, win, exc, days)

    out["all_trades_excess_per_day_bp"] = round(float(exc.sum() / days.clip(min=1).sum() * 1e4), 2)
    rank = sorted(out["numeric"].items(), key=lambda kv: -abs(kv[1]["excess_per_day_bp_top_minus_bottom"] or 0))
    out["ranking_by_spread"] = [{"feature": k, "epd_spread_bp": v["excess_per_day_bp_top_minus_bottom"],
                                 "epd_early": v["epd_spread_2016_2021"], "epd_late": v["epd_spread_2022_2026"],
                                 "epd_consistent": v["epd_consistent_sign"],
                                 "winner_spread": v["winner_rate_top_minus_bottom"]} for k, v in rank]
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for r in out["ranking_by_spread"]:
        print(r, flush=True)
    for k, v in out["categorical"].items():
        top = list(v.items())
        print(k, "top", top[:5], "bottom", top[-3:], flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
