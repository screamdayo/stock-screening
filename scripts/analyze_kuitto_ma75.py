"""Research: does the MA75 state on the signal day separate good and bad Kuitto signals?

Inputs (private repo, mounted by .github/workflows/kuitto_ma75.yml):
  private_results/kuitto_runner_features_trades.csv   all 10Y Kuitto signals (n=5,846), fixed 10-session return
  private_results/kuitto_ma25_gap_fourway_10y.csv     production-like set: ma5_vs_ma25>=-5 and gap pass (n~1,294),
                                                      stop / gakutto / max15 exit pnl, 'winner' = 本命 flag
  data/batches, data/daily_updates                    daily bars (same current-listing archive the signals used)
Output: results/kuitto_ma75.json (aggregate statistics only).

MA75 features on the signal day's close (adjusted, stale split basis fixed):
  slope20 = MA75/MA75[-20]-1;  state: down (<= -1%), flat (-1%, +1%], up (> +1%)  -- fixed in advance
  above   = close > MA75
  fresh_turn = slope20 crossed above 0 within the last 20 sessions (incl. today)
Each group is reported for the whole period and for signals before / from 2021-09-20 (the split the Kuitto
research has used), so a pattern present in only one half is visible. No thresholds are tuned here.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from price_adjust import normalize_code, stale_basis_multiplier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PRIV = ROOT / "private_results"
OUT = ROOT / "results/kuitto_ma75.json"
SPLIT = pd.Timestamp("2021-09-20")
COLS = ["Date", "Code", "C", "AdjC", "AdjFactor"]


def load_close(codes):
    files = sorted((DATA / "batches").glob("batch_*.parquet")) + sorted((DATA / "daily_updates").glob("*.parquet"))
    parts = []
    for f in files:
        d = pd.read_parquet(f, columns=COLS)
        d["Code"] = d.Code.astype(str).map(normalize_code)
        parts.append(d[d.Code.isin(codes)])
    a = pd.concat(parts, ignore_index=True)
    a["Date"] = pd.to_datetime(a.Date)
    for c in ("C", "AdjC", "AdjFactor"):
        a[c] = pd.to_numeric(a[c], errors="coerce")
    a = a.sort_values(["Code", "Date"]).drop_duplicates(["Code", "Date"], keep="last").reset_index(drop=True)
    a["P"] = a.AdjC * stale_basis_multiplier(a)
    a = a[a.P > 0]
    return a.pivot(index="Date", columns="Code", values="P").sort_index()


def ma75_features(close):
    ma = close.rolling(75, min_periods=75).mean()
    slope = ma / ma.shift(20) - 1
    turn = (slope.shift(1) <= 0) & (slope > 0)
    fresh = turn.rolling(20, min_periods=1).max().astype(bool)
    return {"slope20_pct": slope * 100, "close_vs_ma75_pct": (close / ma - 1) * 100, "fresh_turn": fresh}


def attach(df, feats, close):
    r = close.index.get_indexer(df.signal_date)
    c = close.columns.get_indexer(df.code)
    ok = (r >= 0) & (c >= 0)
    for k, f in feats.items():
        v = np.full(len(df), np.nan)
        v[ok] = f.to_numpy()[r[ok], c[ok]].astype(float)
        df[k] = v
    df["state"] = pd.cut(df.slope20_pct, [-np.inf, -1, 1, np.inf], labels=["down", "flat", "up"])
    df["above"] = np.where(df.close_vs_ma75_pct.isna(), np.nan, df.close_vs_ma75_pct > 0)
    return df


def metrics(v):
    v = pd.Series(v).dropna()
    if v.empty:
        return {"n": 0}
    return {"n": int(len(v)), "avg_pct": round(float(v.mean()), 3), "median_pct": round(float(v.median()), 3),
            "win_pct": round(float((v > 0).mean() * 100), 1),
            "pf": round(float(v[v > 0].sum() / -v[v < 0].sum()), 3) if (v < 0).any() else None,
            "loss5_pct": round(float((v <= -5).mean() * 100), 1)}


def by_group(df, col, field):
    out = {}
    for g, d in df.groupby(col, observed=True):
        key = str(bool(g)) if isinstance(g, (bool, np.bool_, float)) and g in (0.0, 1.0) else str(g)
        out[key] = {"all": metrics(d[field]), "older": metrics(d[d.signal_date < SPLIT][field]),
                    "newer": metrics(d[d.signal_date >= SPLIT][field]),
                    "by_year_avg": {str(y): round(float(x[field].mean()), 2) for y, x in d.groupby(d.signal_date.dt.year)}}
    return out


def quintiles(df, col, field):
    d = df.dropna(subset=[col, field]).copy()
    d["bin"] = pd.qcut(d[col].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    out = {}
    for b, x in d.groupby("bin", observed=True):
        out[str(b)] = {"range": [round(float(x[col].min()), 2), round(float(x[col].max()), 2)], **metrics(x[field]),
                       "older_avg": round(float(x[x.signal_date < SPLIT][field].mean()), 3),
                       "newer_avg": round(float(x[x.signal_date >= SPLIT][field].mean()), 3)}
    return out


def tests():
    idx = pd.bdate_range("2020-01-01", periods=200)
    x = np.r_[np.linspace(100, 80, 100), np.linspace(80, 120, 100)]
    close = pd.DataFrame({"A": x}, index=idx)
    f = ma75_features(close)
    s = f["slope20_pct"].A
    i = int(np.flatnonzero(((s.shift(1) <= 0) & (s > 0)).to_numpy())[0])
    assert f["fresh_turn"].A.iloc[i] and f["fresh_turn"].A.iloc[i + 19] and not f["fresh_turn"].A.iloc[i + 20]
    assert not f["fresh_turn"].A.iloc[i - 1]
    df = pd.DataFrame({"code": ["A", "B"], "signal_date": [idx[150], idx[150]]})
    df = attach(df, f, close)
    assert abs(df.slope20_pct.iloc[0] - s.iloc[150]) < 1e-12 and np.isnan(df.slope20_pct.iloc[1])
    print("MA75 feature, fresh-turn window and lookup tests passed", flush=True)


def main():
    tests()
    allk = pd.read_csv(PRIV / "kuitto_runner_features_trades.csv", dtype={"code": str}, parse_dates=["signal_date"])
    prod = pd.read_csv(PRIV / "kuitto_ma25_gap_fourway_10y.csv", dtype={"code": str}, parse_dates=["signal_date"])
    prod = prod[(prod.ma5_vs_ma25_pct >= -5) & prod.gap_pass].copy()
    for d in (allk, prod):
        d["code"] = d.code.astype(str).map(normalize_code)
    print(f"all kuitto {len(allk):,}; production-like {len(prod):,}", flush=True)
    close = load_close(set(allk.code) | set(prod.code))
    feats = ma75_features(close)
    allk, prod = attach(allk, feats, close), attach(prod, feats, close)
    honmei = prod[prod.winner.astype(str).str.lower() == "true"]

    out = {"settings": {"sets": {"all_kuitto": "10Y Kuitto signals, fixed 10-session return (ret10, %)",
                                 "production_like": "ma5_vs_ma25>=-5 & gap pass; pnl with stop/gakutto/max15 exits (%)",
                                 "honmei": "production_like with winner flag (本命)"},
                        "ma75_state": "slope20 of MA75: down<=-1%, flat (-1,+1], up>+1% (fixed, not tuned)",
                        "split": str(SPLIT.date()),
                        "caveat": "current-listing archive (survivors) like the source Kuitto research; overlapping trades"},
           "coverage": {"all_kuitto_missing_ma75": int(allk.slope20_pct.isna().sum()),
                        "production_missing_ma75": int(prod.slope20_pct.isna().sum())},
           "baseline": {"all_kuitto": metrics(allk.ret10), "production_like": metrics(prod.pnl), "honmei": metrics(honmei.pnl)}}
    for name, df, field in (("all_kuitto", allk, "ret10"), ("production_like", prod, "pnl"), ("honmei", honmei, "pnl")):
        out[name] = {"by_state": by_group(df, "state", field),
                     "by_above_ma75": by_group(df.dropna(subset=["above"]), "above", field),
                     "by_fresh_turn": by_group(df.dropna(subset=["slope20_pct"]), "fresh_turn", field),
                     "slope20_quintiles": quintiles(df, "slope20_pct", field),
                     "close_vs_ma75_quintiles": quintiles(df, "close_vs_ma75_pct", field)}
        for k in ("by_state", "by_above_ma75", "by_fresh_turn"):
            print(name, k, {g: (v["all"]["n"], v["all"].get("avg_pct"), v["older"].get("avg_pct"), v["newer"].get("avg_pct"),
                                v["all"].get("pf")) for g, v in out[name][k].items()}, flush=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
