"""Research: earnings data as the entry filter for Standard / Growth big winners (12-month doubling).

Same universe, formation and forward-return rules as scripts/analyze_smallcap_bigwinners.py (point-in-time
Standard + Growth, delisted issues kept, month-end formation, 12-month forward return, >= +100% = big winner).
Inputs add data/fins_summary (TDnet summaries). Output: results/smallcap_fundamental_entry.json (aggregate only).

Point-in-time rules: a disclosure at/after 15:00 counts from the next day; a formation date uses disclosures
strictly before it; values older than 150 days are treated as missing.
Earnings features
  q_sales_yoy      stand-alone quarter sales vs the same quarter a year earlier
  q_op_yoy         stand-alone quarter operating profit yoy (both positive)
  q_sales_accel    q_sales_yoy minus the previous quarter's q_sales_yoy
  op_turnaround    stand-alone quarter OP > 0 while the same quarter a year earlier was <= 0 (flag)
  fc_op_growth     company OP forecast for the current FY (1Q-3Q: FOP; FY report: NxFOP for the next FY)
                   vs the last actual FY OP (positive base)
  q_sales_size     stand-alone quarter sales (yen; size proxy)
  mcap             raw close x shares outstanding, only if a ShOut* column exists in the archive
Then the technical timing check inside the strongest earnings group (top quintile of q_sales_yoy with
q_op_yoy > 0): near the 52-week high vs not, 3-month return high vs low, turnover high vs low.
Caution as before: hit rates rise with dispersion, so excess return and the -50% rate are reported too.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_ma75_turn_trend import load_prices, read_dir
from analyze_smallcap_bigwinners import (H, MIN_HIST, MIN_PRICE, MIN_TURNOVER, SPLIT, features, forward, market_panel,
                                         quintile_report, stats)
from price_adjust import normalize_code

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results/smallcap_fundamental_entry.json"
STALE_DAYS = 150


def num(s):
    return pd.to_numeric(s, errors="coerce")


def load_fins():
    fs = read_dir(DATA / "fins_summary")
    fs["Code"] = fs.Code.astype(str).map(normalize_code)
    for c in ("Sales", "OP", "FOP", "NxFOP"):
        fs[c] = num(fs[c]) if c in fs else np.nan
    late = num(fs.DiscTime.astype(str).str[:2]) >= 15
    fs["avail"] = pd.to_datetime(fs.DiscDate, errors="coerce") + pd.to_timedelta(np.where(late, 1, 0), unit="D")
    fs = fs.dropna(subset=["avail"])
    shout = [c for c in fs.columns if c.startswith("ShOut")]
    print("fins columns:", sorted(fs.columns.tolist()), flush=True)
    return fs, (shout[0] if shout else None)


def quarterly(fs):
    doc = fs.DocType.astype(str)
    st = fs[doc.str.contains("FinancialStatements") & fs.CurPerType.isin(["1Q", "2Q", "3Q", "FY"])].copy()
    st["fye"] = pd.to_datetime(st.CurFYEn, errors="coerce") + pd.offsets.MonthEnd(0)
    st = st.dropna(subset=["fye"]).sort_values(["Code", "avail"]).drop_duplicates(["Code", "fye", "CurPerType"], keep="first")
    st["qn"] = st.CurPerType.map({"1Q": 1, "2Q": 2, "3Q": 3, "FY": 4})
    for v in ("Sales", "OP"):
        prev_q = st[["Code", "fye", "qn", v]].assign(qn=lambda d: d.qn + 1).rename(columns={v: v + "_cum_prev"})
        st = st.merge(prev_q, on=["Code", "fye", "qn"], how="left")
        st["q_" + v] = np.where(st.qn == 1, st[v], st[v] - st[v + "_cum_prev"])
        py = st[["Code", "fye", "qn", "q_" + v]].assign(fye=lambda d: d.fye + pd.DateOffset(years=1) + pd.offsets.MonthEnd(0))
        st = st.merge(py.rename(columns={"q_" + v: "q_" + v + "_py"}), on=["Code", "fye", "qn"], how="left")
    with np.errstate(divide="ignore", invalid="ignore"):
        st["q_sales_yoy"] = (st.q_Sales / st.q_Sales_py - 1).where(st.q_Sales_py > 0)
        st["q_op_yoy"] = (st.q_OP / st.q_OP_py - 1).where((st.q_OP_py > 0) & (st.q_OP > 0))
    st["op_turnaround"] = np.where(st.q_OP.notna() & st.q_OP_py.notna(), ((st.q_OP > 0) & (st.q_OP_py <= 0)).astype(float), np.nan)
    st = st.sort_values(["Code", "avail"])
    st["q_sales_accel"] = st.q_sales_yoy - st.groupby("Code").q_sales_yoy.shift(1)
    st["q_sales_size"] = st.q_Sales
    # company forecast growth vs last actual FY OP
    fy_op = st[st.qn == 4][["Code", "avail", "OP"]].rename(columns={"OP": "fy_op"})
    st = pd.merge_asof(st.sort_values("avail"), fy_op.sort_values("avail"), on="avail", by="Code", direction="backward")
    fc = np.where(st.qn == 4, st.NxFOP, st.FOP)
    with np.errstate(divide="ignore", invalid="ignore"):
        st["fc_op_growth"] = pd.Series(fc / st.fy_op - 1, index=st.index).where(st.fy_op > 0)
    cols = ["Code", "avail", "q_sales_yoy", "q_op_yoy", "q_sales_accel", "op_turnaround", "fc_op_growth", "q_sales_size"]
    return st[cols].replace([np.inf, -np.inf], np.nan)


def shares(fs, col):
    s = fs[["Code", "avail", col]].copy()
    s[col] = num(s[col])
    return s.dropna(subset=[col]).rename(columns={col: "shares"}).sort_values("avail")


def build_panel():
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
    rows = []
    for d in P.index[ym.ne(ym.shift(-1)).to_numpy()]:
        if P.index.get_loc(d) < 252:
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
        if f_row.notna().mean() < 0.9:
            continue
        fr = pd.DataFrame({"Date": d, "ym": p, "Code": codes, "mkt": m.loc[codes].to_numpy(), "fwd": f_row.to_numpy(),
                           "rawC": raw.loc[d, codes].to_numpy()})
        for k in ("from_52w_high", "ret3m", "turnover20_median", "vol60"):
            fr[k] = feats[k].loc[d, codes].to_numpy()
        rows.append(fr.dropna(subset=["fwd"]))
    panel = pd.concat(rows, ignore_index=True)
    panel["exc"] = panel.fwd - panel.groupby("ym").fwd.transform("mean")
    return panel


def attach_fundamentals(panel, q, sh):
    p = pd.merge_asof(panel.sort_values("Date"), q.sort_values("avail"), left_on="Date", right_on="avail", by="Code",
                      direction="backward", allow_exact_matches=False)
    stale = (p.Date - p.avail).dt.days > STALE_DAYS
    for c in ("q_sales_yoy", "q_op_yoy", "q_sales_accel", "op_turnaround", "fc_op_growth", "q_sales_size"):
        p.loc[stale | p.avail.isna(), c] = np.nan
    if sh is not None:
        p = pd.merge_asof(p.sort_values("Date"), sh.rename(columns={"avail": "sh_avail"}), left_on="Date",
                          right_on="sh_avail", by="Code", direction="backward", allow_exact_matches=False)
        p["mcap"] = p.rawC * p.shares
    return p


def flag_report(d, col):
    out = {}
    for v, g in d.dropna(subset=[col]).groupby(col):
        out[str(bool(v))] = {"all": stats(g), "before_2021_07": stats(g[g.ym < SPLIT]), "from_2021_07": stats(g[g.ym >= SPLIT])}
    return out


def split_report(d, col, label_hi, label_lo, cond_hi):
    out = {}
    for name, g in ((label_hi, d[cond_hi]), (label_lo, d[~cond_hi & d[col].notna()])):
        out[name] = {"all": stats(g), "before_2021_07": stats(g[g.ym < SPLIT]), "from_2021_07": stats(g[g.ym >= SPLIT])}
    return out


def tests():
    fs = pd.DataFrame({
        "Code": ["1111"] * 8, "DocType": ["1QFinancialStatements_Consolidated_JP"] * 8,
        "CurPerType": ["1Q", "2Q", "3Q", "FY", "1Q", "2Q", "3Q", "FY"],
        "CurFYEn": ["2020-03-31"] * 4 + ["2021-03-31"] * 4,
        "Sales": [100, 200, 300, 400, 150, 300, 450, 600], "OP": [10, 20, 30, 40, -5, 10, 30, 50],
        "FOP": [40, 40, 40, np.nan, 60, 60, 60, np.nan], "NxFOP": [np.nan] * 3 + [60] + [np.nan] * 3 + [70],
        "avail": pd.to_datetime(["2019-08-01", "2019-11-01", "2020-02-01", "2020-05-01",
                                 "2020-08-01", "2020-11-01", "2021-02-01", "2021-05-01"])})
    q = quarterly(fs).set_index("avail")
    r = q.loc["2020-11-01"]
    assert abs(r.q_sales_yoy - 0.5) < 1e-12                  # (300-150)/(200-100) - 1
    assert abs(r.q_op_yoy - (15 / 10 - 1)) < 1e-12           # q OP 10-(-5)=15 vs 10
    assert abs(r.q_sales_accel - 0.0) < 1e-12                # 1Q yoy also +50%
    assert np.isnan(q.loc["2020-08-01"].q_op_yoy) and q.loc["2020-08-01"].op_turnaround == 0.0
    assert abs(r.fc_op_growth - (60 / 40 - 1)) < 1e-12       # FY2021 forecast vs FY2020 actual
    assert abs(q.loc["2020-05-01"].fc_op_growth - (60 / 40 - 1)) < 1e-12  # FY report uses NxFOP
    print("stand-alone quarter, yoy, acceleration and forecast-growth tests passed", flush=True)


def main():
    tests()
    panel = build_panel()
    fs, shout_col = load_fins()
    q = quarterly(fs)
    sh = shares(fs, shout_col) if shout_col else None
    panel = attach_fundamentals(panel, q, sh)
    print(f"rows {len(panel):,}; coverage q_sales_yoy {panel.q_sales_yoy.notna().mean():.1%}, "
          f"fc_op_growth {panel.fc_op_growth.notna().mean():.1%}, mcap {'yes' if sh is not None else 'no'}", flush=True)

    numeric = ["q_sales_yoy", "q_op_yoy", "q_sales_accel", "fc_op_growth", "q_sales_size"] + (["mcap"] if sh is not None else [])
    out = {"settings": {"universe": "point-in-time Standard + Growth, raw close>=100, turnover>=10M, incl. delisted",
                        "target": "12-month forward return >= +100%", "fins": f"disclosed before formation, <= {STALE_DAYS} days old",
                        "split": str(SPLIT)},
           "base": {"all": stats(panel), "with_q_sales_yoy": stats(panel[panel.q_sales_yoy.notna()]),
                    "missing_q_sales_yoy": stats(panel[panel.q_sales_yoy.isna()])},
           "features": {}, "flags": {}, "timing_in_strong_group": {}}
    for k in numeric:
        out["features"][k] = {m: quintile_report(panel[panel.mkt == m] if m != "all" else panel, k)
                              for m in ("all", "Standard", "Growth")}
    out["flags"]["op_turnaround"] = {m: flag_report(panel[panel.mkt == m] if m != "all" else panel, "op_turnaround")
                                     for m in ("all", "Standard", "Growth")}

    # strongest earnings group: top quintile sales growth (within month) and positive OP growth
    panel["sales_q"] = panel.groupby("ym").q_sales_yoy.transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1 if s.notna().sum() >= 25 else np.nan)
    strong = panel[(panel.sales_q == 5) & (panel.q_op_yoy > 0)].copy()
    out["strong_group"] = {"definition": "top-quintile q_sales_yoy within the month and q_op_yoy > 0",
                           "all": stats(strong), "Standard": stats(strong[strong.mkt == "Standard"]),
                           "Growth": stats(strong[strong.mkt == "Growth"]),
                           "before_2021_07": stats(strong[strong.ym < SPLIT]), "from_2021_07": stats(strong[strong.ym >= SPLIT]),
                           "by_formation_year": {str(y): stats(g) for y, g in strong.groupby(strong.ym.dt.year)}}
    med_ret3 = strong.groupby("ym").ret3m.transform("median")
    med_to = strong.groupby("ym").turnover20_median.transform("median")
    out["timing_in_strong_group"] = {
        "near_52w_high(<=10% below)": split_report(strong, "from_52w_high", "near_high", "not_near_high", strong.from_52w_high >= -0.10),
        "ret3m_above_group_median": split_report(strong, "ret3m", "ret3m_high", "ret3m_low", strong.ret3m > med_ret3),
        "turnover_above_group_median": split_report(strong, "turnover20_median", "turnover_high", "turnover_low",
                                                    strong.turnover20_median > med_to)}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))

    print("base", out["base"], flush=True)
    for k in numeric:
        b = out["features"][k]["all"]
        print(k, {q_: (v["doubled_pct"], v["mean_excess_pct"], v["down50_pct"]) for q_, v in b["bins"].items()},
              "halves", b["before_2021_07"]["doubled_q5_minus_q1"], b["from_2021_07"]["doubled_q5_minus_q1"],
              "consistent", b["consistent"], flush=True)
    print("turnaround", {k: (v["all"]["n"], v["all"]["doubled_pct"], v["all"]["mean_excess_pct"]) for k, v in out["flags"]["op_turnaround"]["all"].items()})
    print("strong", out["strong_group"]["all"], "Std", out["strong_group"]["Standard"], "Gro", out["strong_group"]["Growth"], flush=True)
    for k, v in out["timing_in_strong_group"].items():
        print(k, {g: (x["all"]["n"], x["all"]["doubled_pct"], x["all"]["mean_excess_pct"], x["all"]["down50_pct"]) for g, x in v.items()}, flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
