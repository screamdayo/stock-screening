"""Research: what do Standard / Growth stocks that double within 12 months look like beforehand? (technical only)

Inputs (private archive mounted under data/, see .github/workflows/smallcap_bigwinners.yml):
  data/batches, data/daily_updates, data/delisted, data/master_history
Output: results/smallcap_bigwinners.json (aggregate statistics only; no per-issue data).

Method (same universe conventions as results/long_term_value_10y.json)
- Month-end formation 2016-10 .. the last month with a full 12-month window. Forward 12-month price return
  (adjusted close, no dividends); an issue delisted inside the window is closed at its last traded price.
- Market from the master snapshot of the formation month: TSE 2nd / JASDAQ Standard / Standard -> Standard,
  Mothers / JASDAQ Growth / Growth -> Growth. Common stock only (ProdCat 011, 4-char codes).
- Universe: raw close >= 100 yen, 20-day median turnover (raw C*Vo) >= 10M yen, >= 130 sessions of history.
- Big winner = forward 12-month return >= +100%. Also reported: >= +50% and <= -50%.
- Features use data up to the formation close only. Price level uses the unadjusted close (adjusted prices leak
  later splits, which big winners often do). Each feature is ranked within its formation month, then split into
  quintiles (5 = highest), so market-wide swings do not decide the bin.
- Per quintile: doubling rate, mean / median 12-month return, mean excess over the month's universe average,
  share falling 50%+. Spreads (Q5 - Q1) are shown for formations before 2021-07 and from 2021-07 separately.
Caution: features that pick volatile stocks raise the doubling rate partly because they raise dispersion
(more big losers too), so the excess return and the -50% rate are shown next to the hit rate.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_ma75_turn_trend import load_prices, read_dir
from price_adjust import normalize_code

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results/smallcap_bigwinners.json"
H = 252
SPLIT = pd.Period("2021-07", "M")
MKT = {"東証二部": "Standard", "JASDAQ スタンダード": "Standard", "スタンダード": "Standard",
       "マザーズ": "Growth", "JASDAQ グロース": "Growth", "グロース": "Growth"}
MIN_PRICE, MIN_TURNOVER, MIN_HIST = 100, 10_000_000, 130


def market_panel(codes):
    mh = read_dir(DATA / "master_history", ["Code", "MktNm", "ProdCat", "SnapshotDate"])
    mh["Code"] = mh.Code.astype(str).map(normalize_code)
    mh["mkt"] = mh.MktNm.map(MKT)
    mh = mh[mh.mkt.notna() & (mh.ProdCat.astype(str) == "011") & mh.Code.isin(codes)]
    mh["ym"] = pd.to_datetime(mh.SnapshotDate).dt.to_period("M")
    return mh.drop_duplicates(["Code", "ym"], keep="last").set_index(["ym", "Code"]).mkt


def features(P, raw, to):
    r = P.pct_change(fill_method=None)
    f = {}
    for name, n in (("ret1m", 21), ("ret3m", 63), ("ret6m", 126), ("ret12m", 252)):
        f[name] = P / P.shift(n) - 1
    f["ret12m_skip1m"] = P.shift(21) / P.shift(252) - 1
    hi = P.rolling(252, min_periods=130).max()
    lo = P.rolling(252, min_periods=130).min()
    f["from_52w_high"] = P / hi - 1
    f["from_52w_low"] = P / lo - 1
    f["new_high_last20"] = (P >= hi * 0.999).rolling(20, min_periods=1).max()
    f["vol60"] = r.rolling(60, min_periods=50).std()
    t20 = to.rolling(20, min_periods=15).median()
    f["turnover20_median"] = t20
    f["turnover_surge_20_vs_120"] = to.rolling(20, min_periods=15).mean() / to.rolling(120, min_periods=90).mean()
    f["turnover_surge_60_vs_250"] = to.rolling(60, min_periods=45).mean() / to.rolling(250, min_periods=180).mean()
    f["price_raw"] = raw
    ma200 = P.rolling(200, min_periods=150).mean()
    f["close_vs_ma200"] = P / ma200 - 1
    f["ma200_slope60"] = ma200 / ma200.shift(60) - 1
    f["up_days_share60"] = (r > 0).astype(float).where(r.notna()).rolling(60, min_periods=50).mean()
    return f, t20


def forward(P):
    n = len(P)
    last_valid = P.apply(lambda s: s.last_valid_index())
    pos = {d: i for i, d in enumerate(P.index)}
    last_i = last_valid.map(lambda d: pos.get(d, -1) if d is not None else -1)
    fw = P.shift(-H) / P - 1
    final = P.ffill().iloc[-1]
    delisted = last_i < n - 1 - 7
    # delisted inside the window: close at last traded price
    term = (final / P - 1)
    idx = np.arange(n)[:, None]
    inside = delisted.to_numpy()[None, :] & (last_i.to_numpy()[None, :] < idx + H) & (last_i.to_numpy()[None, :] >= idx)
    out = fw.where(fw.notna(), term.where(inside))
    return out, pd.DataFrame(inside & fw.isna().to_numpy(), index=P.index, columns=P.columns)


def stats(d):
    v = d.fwd
    if v.empty:
        return {"n": 0}
    return {"n": int(len(v)), "doubled_pct": round(float((v >= 1).mean() * 100), 2),
            "up50_pct": round(float((v >= 0.5).mean() * 100), 1), "down50_pct": round(float((v <= -0.5).mean() * 100), 1),
            "mean_pct": round(float(v.mean() * 100), 1), "median_pct": round(float(v.median() * 100), 1),
            "mean_excess_pct": round(float(d.exc.mean() * 100), 1)}


def quintile_report(panel, feat):
    d = panel.dropna(subset=[feat]).copy()
    d["q"] = d.groupby("ym")[feat].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1
                                             if s.notna().sum() >= 25 else np.nan)
    d = d.dropna(subset=["q"])
    out = {"bins": {}}
    for q, g in d.groupby("q"):
        out["bins"][str(int(q))] = {"range_median": round(float(g[feat].median()), 4), **stats(g)}
    for name, part in (("before_2021_07", d[d.ym < SPLIT]), ("from_2021_07", d[d.ym >= SPLIT])):
        s5, s1 = part[part.q == 5], part[part.q == 1]
        out[name] = {"doubled_q5_minus_q1": round(float(((s5.fwd >= 1).mean() - (s1.fwd >= 1).mean()) * 100), 2),
                     "excess_q5_minus_q1": round(float((s5.exc.mean() - s1.exc.mean()) * 100), 1),
                     "doubled_q5": round(float((s5.fwd >= 1).mean() * 100), 2),
                     "doubled_q1": round(float((s1.fwd >= 1).mean() * 100), 2)}
    a, b = out["before_2021_07"], out["from_2021_07"]
    out["consistent"] = bool(np.sign(a["doubled_q5_minus_q1"]) == np.sign(b["doubled_q5_minus_q1"]) != 0
                             and np.sign(a["excess_q5_minus_q1"]) == np.sign(b["excess_q5_minus_q1"]))
    return out


def tests():
    idx = pd.bdate_range("2018-01-01", periods=600)
    P = pd.DataFrame({"A": np.linspace(100, 400, 600), "B": np.r_[np.full(300, 100.0), np.full(300, np.nan)]}, index=idx)
    fw, inside = forward(P)
    assert abs(fw.A.iloc[10] - (P.A.iloc[10 + H] / P.A.iloc[10] - 1)) < 1e-12
    assert fw.B.iloc[100] == 0.0 and inside.B.iloc[100]            # delisted in window -> last price
    assert np.isnan(fw.A.iloc[-5])                                  # no future data, not delisted
    d = pd.DataFrame({"ym": [pd.Period("2020-01", "M")] * 50, "x": np.arange(50.0),
                      "fwd": np.r_[np.zeros(40), np.full(10, 1.5)], "exc": 0.0})
    r = quintile_report(d, "x")
    assert r["bins"]["5"]["doubled_pct"] == 100.0 and r["bins"]["1"]["doubled_pct"] == 0.0
    print("forward-return, delisting and month-rank quintile tests passed", flush=True)


def main():
    tests()
    a = load_prices()
    P = a.pivot(index="Date", columns="Code", values="P").sort_index()
    raw = a.pivot(index="Date", columns="Code", values="C").reindex_like(P)
    to = a.pivot(index="Date", columns="Code", values="to").reindex_like(P)
    del a
    feats, t20 = features(P, raw, to)
    fwd, inside = forward(P)
    hist = P.notna().cumsum()
    mkt = market_panel(set(P.columns))

    ym = pd.Series(P.index.to_period("M"), index=P.index)
    month_end = P.index[ym.ne(ym.shift(-1)).to_numpy()]
    rows = []
    for d in month_end:
        i = P.index.get_loc(d)
        if i < 252:
            continue
        p = pd.Period(d, "M")
        try:
            m = mkt.loc[p]
        except KeyError:
            continue
        codes = m.index.intersection(P.columns)
        ok = (raw.loc[d, codes] >= MIN_PRICE) & (t20.loc[d, codes] >= MIN_TURNOVER) & (hist.loc[d, codes] >= MIN_HIST)
        codes = codes[ok.to_numpy()]
        f_row = fwd.loc[d, codes]
        if f_row.notna().mean() < 0.9:   # window not complete yet (end of data)
            continue
        frame = pd.DataFrame({"ym": p, "code": codes, "mkt": m.loc[codes].to_numpy(), "fwd": f_row.to_numpy(),
                              "delisted_in_window": inside.loc[d, codes].to_numpy()})
        for k, f in feats.items():
            frame[k] = f.loc[d, codes].to_numpy()
        rows.append(frame.dropna(subset=["fwd"]))
    panel = pd.concat(rows, ignore_index=True)
    panel["exc"] = panel.fwd - panel.groupby("ym").fwd.transform("mean")
    print(f"formations {panel.ym.min()}..{panel.ym.max()}, rows {len(panel):,}, per month {len(panel)/panel.ym.nunique():.0f}, "
          f"doubled {100*(panel.fwd>=1).mean():.2f}%", flush=True)

    out = {"settings": {"universe": "point-in-time Standard + Growth, common stock, raw close>=100, 20d median turnover>=10M, incl. delisted",
                        "target": "forward 12-month return >= +100%", "features": "technical only, ranked within each month",
                        "formations": [str(panel.ym.min()), str(panel.ym.max())], "split": str(SPLIT)},
           "base": {"all": stats(panel), "Standard": stats(panel[panel.mkt == "Standard"]),
                    "Growth": stats(panel[panel.mkt == "Growth"]),
                    "by_formation_year": {str(y): stats(g) for y, g in panel.groupby(panel.ym.dt.year)},
                    "delisted_in_window_pct": round(float(panel.delisted_in_window.mean() * 100), 2)},
           "features": {}}
    for k in feats:
        out["features"][k] = {"all": quintile_report(panel, k),
                              "Standard": quintile_report(panel[panel.mkt == "Standard"], k),
                              "Growth": quintile_report(panel[panel.mkt == "Growth"], k)}
    rank = sorted(out["features"].items(),
                  key=lambda kv: -abs(kv[1]["all"]["bins"]["5"]["doubled_pct"] - kv[1]["all"]["bins"]["1"]["doubled_pct"]))
    out["ranking"] = []
    for k, v in rank:
        b = v["all"]["bins"]
        out["ranking"].append({"feature": k, "doubled_q1": b["1"]["doubled_pct"], "doubled_q5": b["5"]["doubled_pct"],
                               "excess_q1": b["1"]["mean_excess_pct"], "excess_q5": b["5"]["mean_excess_pct"],
                               "down50_q1": b["1"]["down50_pct"], "down50_q5": b["5"]["down50_pct"],
                               "consistent_halves": v["all"]["consistent"]})
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("base", out["base"]["all"], "Std", out["base"]["Standard"], "Gro", out["base"]["Growth"], flush=True)
    for r in out["ranking"]:
        print(r, flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
