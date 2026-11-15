"""Research: does "cheap + improving quarterly operating profit" hold for 6-12 month holds over 10 years?

Inputs (private J-Quants archive mounted under data/, see the workflow):
  data/batches, data/daily_updates   daily bars of issues listed at archive time (+ updates)
  data/delisted                      daily bars of issues delisted during the 10 years
  data/master_history                monthly listed-issue snapshots (point-in-time market)
  data/fins_summary                  financial summaries, 2016-11 onwards
Output: results/long_term_value_10y.json (aggregate statistics only; no per-issue data).

Method
- Month-end formation, 12-month forward price return (no dividends). An issue delisted inside the
  window is closed at its last traded price, so delisted issues are not dropped (no survivor bias).
- Market at formation from the snapshot of that month: TSE 1st -> Prime, TSE 2nd / JASDAQ Standard ->
  Standard, Mothers / JASDAQ Growth -> Growth. Only common stock (product category 011, 4-char codes).
- Universe: price >= 100 yen; 20-day median turnover >= 10M yen (Standard + Growth) or 100M (Prime).
- B/P: last fiscal-year BPS (split-adjusted to the formation date) / raw close; the latest disclosure
  must be <= 200 days old. Quarterly OP yoy: stand-alone quarter vs the same quarter a year earlier
  (both positive base). Disclosures at/after 15:00 count from the next day.
- Rule: B/P rank >= 60th percentile and quarterly OP yoy rank >= 70th percentile within the month.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from price_adjust import normalize_code, stale_basis_multiplier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results/long_term_value_10y.json"
CUT = pd.Period("2021-11", "M")  # the earlier 5-year study covered formations after this month
PRICE_COLS = ["Date", "Code", "C", "Vo", "AdjC", "AdjFactor"]
MARKET_MAP = {"東証一部": "Prime", "プライム": "Prime", "東証二部": "Standard", "JASDAQ スタンダード": "Standard",
              "スタンダード": "Standard", "マザーズ": "Growth", "JASDAQ グロース": "Growth", "グロース": "Growth"}


def read_dir(path, columns=None):
    files = sorted(Path(path).glob("*.parquet"))
    return pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)


def load_prices():
    dm = pd.read_parquet(DATA / "delisted/delisted_master.parquet")
    dm = dm[(dm.Code.astype(str).str.len() == 4) & (dm.ProdCat.astype(str) == "011")]
    dl = pd.concat([pd.read_parquet(f, columns=PRICE_COLS) for f in sorted((DATA / "delisted").glob("batch_*.parquet"))], ignore_index=True)
    dl["Code"] = dl.Code.astype(str)
    dl = dl[dl.Code.isin(set(dm.Code))].assign(src=2)
    base = read_dir(DATA / "batches", PRICE_COLS).assign(src=0)
    upd = read_dir(DATA / "daily_updates", PRICE_COLS).assign(src=1)
    a = pd.concat([base, upd], ignore_index=True)
    a["Code"] = a.Code.astype(str).map(normalize_code)
    a = a[~a.Code.isin(set(dl.Code))]  # issues delisted after the archive: use their full fresh series
    a = pd.concat([a, dl], ignore_index=True)
    a["Date"] = pd.to_datetime(a.Date)
    a = a.sort_values(["Code", "Date", "src"]).drop_duplicates(["Code", "Date"], keep="last").reset_index(drop=True)
    a["P"] = a.AdjC * stale_basis_multiplier(a)
    a = a[a.P > 0].dropna(subset=["P"]).reset_index(drop=True)
    print(f"daily rows {len(a):,}, issues {a.Code.nunique():,}, delisted issues {dl.Code.nunique():,}", flush=True)
    return a


def month_end_panel(a):
    g = a.groupby("Code", sort=False)
    a["to20"] = (a.C * a.Vo).groupby(a.Code, sort=False).transform(lambda s: s.rolling(20, min_periods=10).median())
    gmax = a.Date.max()
    last = g.Date.transform("max")
    delisted = last < gmax - pd.Timedelta(days=10)
    fw = g.P.shift(-252) / a.P - 1
    term = (g.P.transform("last") / a.P - 1).where(delisted & (last < a.Date + pd.Timedelta(days=365)))
    a["f12"] = fw.fillna(term)
    a["delisted_in_window"] = fw.isna() & term.notna()
    a["cf"] = a.AdjFactor.fillna(1.0).groupby(a.Code, sort=False).cumprod()  # cumulative split factor up to each date
    a["ym"] = a.Date.dt.to_period("M")
    me = a[a.Date == a.groupby("ym").Date.transform("max")].copy()
    mh = read_dir(DATA / "master_history", ["Code", "MktNm", "ProdCat", "SnapshotDate"])
    mh["Code"] = mh.Code.astype(str).map(normalize_code)
    mh["ym"] = pd.to_datetime(mh.SnapshotDate).dt.to_period("M")
    mh["mkt"] = mh.MktNm.map(MARKET_MAP)
    mh = mh[mh.mkt.notna() & (mh.ProdCat.astype(str) == "011")].drop_duplicates(["Code", "ym"])
    me = me.merge(mh[["Code", "ym", "mkt"]], on=["Code", "ym"], how="left")
    me = me[me.mkt.notna() & (me.Date + pd.Timedelta(days=365) <= gmax)]
    return me.rename(columns={"C": "rawC"}), a[["Code", "Date", "cf"]]


def fundamentals(me, cf):
    cols = ["DiscDate", "DiscTime", "Code", "DocType", "CurPerType", "CurFYEn", "Sales", "OP", "BPS"]
    fs = read_dir(DATA / "fins_summary", cols)
    fs["Code"] = fs.Code.astype(str).map(normalize_code)
    for c in ("Sales", "OP", "BPS"):
        fs[c] = pd.to_numeric(fs[c], errors="coerce")
    late = pd.to_numeric(fs.DiscTime.astype(str).str[:2], errors="coerce") >= 15
    fs["avail"] = pd.to_datetime(fs.DiscDate, errors="coerce") + pd.to_timedelta(np.where(late, 1, 0), unit="D")
    fs = fs.dropna(subset=["avail"])
    doc = fs.DocType.astype(str)

    # --- B/P from the last fiscal-year BPS
    f = fs[doc.str.contains("FinancialStatements|EarnForecastRevision")].sort_values(["Code", "avail"]).copy()
    isfy = f.DocType.astype(str).str.startswith("FY")
    f["BPS_fy"] = f.BPS.where(isfy)
    f["fy_disc"] = f.avail.where(isfy)
    f[["BPS_fy", "fy_disc"]] = f.groupby("Code", sort=False)[["BPS_fy", "fy_disc"]].ffill()
    me = pd.merge_asof(me.sort_values("Date"), f[["Code", "avail", "BPS_fy", "fy_disc"]].sort_values("avail"),
                       left_on="Date", right_on="avail", by="Code", direction="backward", allow_exact_matches=False)
    fyd = me[["Code", "fy_disc"]].dropna().drop_duplicates().sort_values("fy_disc")
    cff = pd.merge_asof(fyd, cf.sort_values("Date"), left_on="fy_disc", right_on="Date", by="Code", direction="backward")
    me = me.merge(cff[["Code", "fy_disc", "cf"]].rename(columns={"cf": "cf_fy"}), on=["Code", "fy_disc"], how="left")
    me["BP"] = me.BPS_fy * (me.cf / me.cf_fy).fillna(1.0) / me.rawC
    me = me[(me.Date - me.avail).dt.days <= 200]

    # --- stand-alone quarterly operating profit, year on year
    st = fs[doc.str.contains("FinancialStatements") & fs.CurPerType.isin(["1Q", "2Q", "3Q", "FY"])].copy()
    st["fye"] = pd.to_datetime(st.CurFYEn, errors="coerce") + pd.offsets.MonthEnd(0)
    st = st.dropna(subset=["fye"]).sort_values(["Code", "avail"]).drop_duplicates(["Code", "fye", "CurPerType"], keep="first")
    st["qn"] = st.CurPerType.map({"1Q": 1, "2Q": 2, "3Q": 3, "FY": 4})
    prev_year = st[["Code", "fye", "qn", "OP"]].assign(fye=lambda d: d.fye + pd.DateOffset(years=1) + pd.offsets.MonthEnd(0))
    st = st.merge(prev_year.rename(columns={"OP": "OP_py"}), on=["Code", "fye", "qn"], how="left")
    prev_q = st[["Code", "fye", "qn", "OP", "OP_py"]].assign(qn=lambda d: d.qn + 1)
    st = st.merge(prev_q.rename(columns={"OP": "O_prevq", "OP_py": "Opy_prevq"}), on=["Code", "fye", "qn"], how="left")
    first = st.qn == 1
    q_op = np.where(first, st.OP, st.OP - st.O_prevq)
    q_op_py = np.where(first, st.OP_py, st.OP_py - st.Opy_prevq)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = q_op / q_op_py - 1
    st["q_op_yoy"] = pd.Series(ratio, index=st.index).where(pd.Series(q_op_py, index=st.index) > 0)
    me = pd.merge_asof(me.sort_values("Date"), st[["Code", "avail", "q_op_yoy"]].rename(columns={"avail": "q_avail"}).sort_values("q_avail"),
                       left_on="Date", right_on="q_avail", by="Code", direction="backward", allow_exact_matches=False)
    me.loc[(me.Date - me.q_avail).dt.days > 150, "q_op_yoy"] = np.nan
    return me


def r1(v):
    return None if v is None or not np.isfinite(v) else round(float(v), 1)


def stats(d, u=None):
    s = {"n": int(len(d)), "per_month": int(round(len(d) / max(1, d.ym.nunique()))),
         "doubled_pct": r1((d.f12 >= 1).mean() * 100), "fell_30_pct": r1((d.f12 <= -0.3).mean() * 100),
         "mean_pct": r1(d.f12.mean() * 100), "median_pct": r1(d.f12.median() * 100),
         "delisted_in_window_pct": r1(d.delisted_in_window.mean() * 100)}
    if u is not None and len(d):
        ex = (d.groupby("ym").f12.mean() - u.groupby("ym").f12.mean()).dropna() * 100
        s["excess_vs_universe_pct"] = r1(ex.mean())
        s["t_overlap_adjusted"] = r1(ex.mean() / (ex.std() / np.sqrt(max(1, len(ex) / 12))))
    return s


def rule_report(u):
    u = u.copy()
    for c in ("BP", "q_op_yoy"):
        u[c + "_r"] = u.groupby("ym")[c].rank(pct=True)
    rule = (u.BP_r >= 0.6) & (u.q_op_yoy_r >= 0.7)
    periods = {"all": slice(None), "2016-12_to_2021-11": u.ym <= CUT, "2021-12_onwards": u.ym > CUT}
    out = {"periods": {}, "yearly": {}, "sensitivity": {}}
    for name, m in periods.items():
        uu = u if isinstance(m, slice) else u[m]
        rr = u[rule] if isinstance(m, slice) else u[rule & m]
        out["periods"][name] = {"universe": stats(uu), "rule": stats(rr, uu)}
    for y in sorted(u.ym.dt.year.unique()):
        uu, rr = u[u.ym.dt.year == y], u[rule & (u.ym.dt.year == y)]
        if len(rr) >= 20:
            out["yearly"][str(y)] = {"universe": stats(uu), "rule": stats(rr, uu)}
    old = u.ym <= CUT
    for bp in (0.5, 0.6, 0.7, 0.8):
        for q in (0.6, 0.7, 0.8):
            m = (u.BP_r >= bp) & (u.q_op_yoy_r >= q)
            out["sensitivity"][f"bp_ge{bp}_qop_ge{q}"] = {"all": stats(u[m], u), "2016-12_to_2021-11": stats(u[m & old], u[old])}
    return out


def value_quintiles(u):
    x = u.dropna(subset=["BP"]).copy()
    x = x[x.groupby("ym").BP.transform("size") >= 100]
    x["q"] = x.groupby("ym").BP.transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
    mean = x.groupby(["ym", "q"]).f12.mean().unstack() * 100
    loss = x.groupby(["ym", "q"]).f12.apply(lambda v: (v <= -0.3).mean() * 100).unstack()
    spread = mean[4] - mean[0]
    old, new = spread[spread.index <= CUT], spread[spread.index > CUT]
    return {"quintile_mean_pct_q5_cheapest": {f"Q{int(k) + 1}": r1(v) for k, v in mean.mean().items()},
            "quintile_fell_30_pct": {f"Q{int(k) + 1}": r1(v) for k, v in loss.mean().items()},
            "cheap_minus_expensive_by_formation_year": {str(k): r1(v) for k, v in spread.groupby(spread.index.year).mean().items()},
            "cheap_minus_expensive_2016-12_to_2021-11": {"mean": r1(old.mean()), "months_positive_pct": r1((old > 0).mean() * 100)},
            "cheap_minus_expensive_2021-12_onwards": {"mean": r1(new.mean()), "months_positive_pct": r1((new > 0).mean() * 100)}}


def main():
    a = load_prices()
    me, cf = month_end_panel(a)
    del a
    p = fundamentals(me, cf)
    p = p[p.f12.notna() & (p.rawC >= 100)]
    sg = p[p.mkt.isin(["Standard", "Growth"]) & (p.to20 >= 1e7)]
    pr = p[(p.mkt == "Prime") & (p.to20 >= 1e8)]
    print(f"formations {p.ym.min()}..{p.ym.max()}; S+G rows {len(sg):,}; Prime rows {len(pr):,}", flush=True)
    out = {
        "verdict": ("NOT ADOPTED (2026-11). Over 10 years including delisted issues, 'cheap + improving quarterly OP' does not "
                    "raise the share of stocks that double and its excess return is ~0 for 2016-12..2021-11; the strong result of "
                    "the earlier 5-year study came from the 2021-12+ value rally. What holds in every year is fewer large losses: "
                    "cheap stocks fall 30%+ far less often. Cheap minus expensive was negative in 2017-2019 (Prime clearly)."),
        "method": __doc__.strip().split("Method", 1)[1].strip(),
        "formation_months": [str(p.ym.min()), str(p.ym.max())],
        "note": "q_op_yoy needs the same quarter a year earlier, so the rule has picks from about 2017-11 only.",
        "standard_growth": {"rule": rule_report(sg), "value_quintiles": value_quintiles(sg),
                            "by_market": {m: {"universe": stats(sg[sg.mkt == m])} for m in ("Standard", "Growth")}},
        "prime": {"rule": rule_report(pr), "value_quintiles": value_quintiles(pr)},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    s = out["standard_growth"]["rule"]["periods"]
    print(json.dumps({k: v["rule"] for k, v in s.items()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
