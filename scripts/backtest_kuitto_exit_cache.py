import json
from pathlib import Path
import numpy as np
import pandas as pd

RESULT_DIR = Path("results")
CACHE = RESULT_DIR / "kuitto_composite_signal_cache.parquet"


def metrics(df):
    if df.empty:
        return {"n": 0}
    r = df["return_pct"].dropna()
    if r.empty:
        return {"n": 0}
    wins = r[r > 0]
    losses = r[r < 0]
    gp = wins.sum()
    gl = -losses.sum()
    return {
        "n": int(len(r)),
        "win_rate_pct": round(float((r > 0).mean() * 100), 2),
        "avg_return_pct": round(float(r.mean()), 3),
        "median_return_pct": round(float(r.median()), 3),
        "profit_factor": round(float(gp / gl), 3) if gl > 0 else None,
    }


def strict_gakutto(path, idx):
    if idx - 3 < 0:
        return False
    vals = path.iloc[idx-3:idx+1]
    if vals["MA5"].isna().any():
        return False
    m = vals["MA5"].to_numpy()
    prior1 = m[1] >= m[0]
    prior2 = m[2] >= m[1]
    turn_down = m[3] < m[2]
    bearish = float(path.iloc[idx]["C"]) < float(path.iloc[idx]["O"])
    return prior1 and prior2 and turn_down and bearish


def simulate(path, stop_pct=None, max_hold=16, use_gakutto=False):
    entry_row = path[path["offset"] == 0]
    if entry_row.empty:
        return None
    entry = float(entry_row.iloc[0]["O"])
    if not np.isfinite(entry) or entry <= 0:
        return None

    usable = path[(path["offset"] >= 0) & (path["offset"] <= max_hold)].sort_values("offset").reset_index(drop=True)
    if usable.empty:
        return None

    stop_price = entry * (1 + stop_pct / 100.0) if stop_pct is not None else None

    for idx, row in usable.iterrows():
        off = int(row["offset"])
        if off >= max_hold:
            return {
                "return_pct": (float(row["O"]) / entry - 1) * 100,
                "exit_reason": f"fixed_{max_hold}",
                "hold_sessions_to_exit": max_hold,
            }
        if stop_price is not None:
            o = float(row["O"])
            l = float(row["L"])
            if o <= stop_price:
                return {"return_pct": (o / entry - 1) * 100, "exit_reason": "stop_gap", "hold_sessions_to_exit": off}
            if l <= stop_price:
                return {"return_pct": stop_pct, "exit_reason": f"stop_{stop_pct}", "hold_sessions_to_exit": off}

        if use_gakutto:
            full_idx = path.index[path["offset"] == off]
            if len(full_idx):
                pos = path.index.get_loc(full_idx[0])
                if strict_gakutto(path.reset_index(drop=True), pos):
                    nxt = path[path["offset"] == off + 1]
                    if not nxt.empty:
                        px = float(nxt.iloc[0]["O"])
                        return {"return_pct": (px / entry - 1) * 100, "exit_reason": "strict_gakutto_next_open", "hold_sessions_to_exit": off + 1}
    return None


def main():
    if not CACHE.exists():
        raise RuntimeError(f"Missing cache: {CACHE}")
    df = pd.read_parquet(CACHE).sort_values(["code", "signal_date", "offset"]).reset_index(drop=True)

    groups = [g.reset_index(drop=True) for _, g in df.groupby(["code", "signal_date"], sort=False)]

    fixed_holds = {}
    for hold in [10, 12, 14, 15, 16, 18, 20, 25, 30]:
        rows = [simulate(g, stop_pct=None, max_hold=hold, use_gakutto=False) for g in groups]
        part = pd.DataFrame([r for r in rows if r is not None])
        fixed_holds[str(hold)] = metrics(part)

    exit_grid = []
    for stop in [-3.0, -4.0, -5.0, -6.0, -7.0]:
        for hold in [10, 15, 16]:
            for use_gakutto in [True, False]:
                rows = [simulate(g, stop_pct=stop, max_hold=hold, use_gakutto=use_gakutto) for g in groups]
                part = pd.DataFrame([r for r in rows if r is not None])
                exit_grid.append({
                    "stop_pct": stop,
                    "max_hold_sessions": hold,
                    "use_gakutto": use_gakutto,
                    "metrics": metrics(part),
                    "exit_reasons": part["exit_reason"].value_counts().to_dict() if not part.empty else {},
                })

    # Risk stats for the chosen 16-session fixed exit.
    risk_rows = []
    for g in groups:
        sim = simulate(g, stop_pct=None, max_hold=16, use_gakutto=False)
        if not sim:
            continue
        entry_row = g[g["offset"] == 0]
        if entry_row.empty:
            continue
        risk_rows.append({
            "entry_date": str(entry_row.iloc[0]["entry_date"]),
            "return_pct": float(sim["return_pct"]),
        })

    risk_df = pd.DataFrame(risk_rows)
    risk_df["entry_date_dt"] = pd.to_datetime(risk_df["entry_date"])

    yearly_risk = {}
    for year, part in risk_df.groupby(risk_df["entry_date_dt"].dt.year):
        part = part.sort_values("entry_date_dt").reset_index(drop=True)
        equity = (1 + part["return_pct"] / 100.0).cumprod()
        running_peak = equity.cummax()
        dd = equity / running_peak - 1

        max_losing_streak = 0
        cur_losing_streak = 0
        for r in part["return_pct"]:
            if r < 0:
                cur_losing_streak += 1
                max_losing_streak = max(max_losing_streak, cur_losing_streak)
            else:
                cur_losing_streak = 0

        yearly_risk[str(int(year))] = {
            "n": int(len(part)),
            "max_drawdown_pct": round(float(dd.min() * 100), 2) if len(dd) else 0.0,
            "max_losing_streak": int(max_losing_streak),
            "year_compounded_return_pct": round(float((equity.iloc[-1] - 1) * 100), 2) if len(equity) else 0.0,
            "worst_trade_pct": round(float(part["return_pct"].min()), 3) if len(part) else None,
        }

    all_part = risk_df.sort_values("entry_date_dt").reset_index(drop=True)
    all_equity = (1 + all_part["return_pct"] / 100.0).cumprod()
    all_peak = all_equity.cummax()
    all_dd = all_equity / all_peak - 1
    max_losing_streak_all = 0
    cur_losing_streak = 0
    for r in all_part["return_pct"]:
        if r < 0:
            cur_losing_streak += 1
            max_losing_streak_all = max(max_losing_streak_all, cur_losing_streak)
        else:
            cur_losing_streak = 0

    overall_risk_16 = {
        "n": int(len(all_part)),
        "max_drawdown_pct": round(float(all_dd.min() * 100), 2) if len(all_dd) else 0.0,
        "max_losing_streak": int(max_losing_streak_all),
        "compounded_return_pct": round(float((all_equity.iloc[-1] - 1) * 100), 2) if len(all_equity) else 0.0,
        "worst_trade_pct": round(float(all_part["return_pct"].min()), 3) if len(all_part) else None,
    }

    emergency_stop_grid = []
    for stop in [-10.0, -12.0, -15.0, -18.0]:
        rows = []
        dated = []
        for g in groups:
            sim = simulate(g, stop_pct=stop, max_hold=16, use_gakutto=False)
            if not sim:
                continue
            entry_row = g[g["offset"] == 0]
            if entry_row.empty:
                continue
            rows.append(sim)
            dated.append({
                "entry_date": str(entry_row.iloc[0]["entry_date"]),
                "return_pct": float(sim["return_pct"]),
            })
        part = pd.DataFrame(rows)
        ddf = pd.DataFrame(dated)
        if not ddf.empty:
            ddf["entry_date_dt"] = pd.to_datetime(ddf["entry_date"])
            ddf = ddf.sort_values("entry_date_dt").reset_index(drop=True)
            eq = (1 + ddf["return_pct"] / 100.0).cumprod()
            peak = eq.cummax()
            dd = eq / peak - 1
            max_streak = 0
            cur = 0
            for r in ddf["return_pct"]:
                if r < 0:
                    cur += 1
                    max_streak = max(max_streak, cur)
                else:
                    cur = 0
            max_dd = round(float(dd.min() * 100), 2)
            worst = round(float(ddf["return_pct"].min()), 3)
        else:
            max_dd = None
            max_streak = 0
            worst = None

        emergency_stop_grid.append({
            "stop_pct": stop,
            "metrics": metrics(part) if not part.empty else {"n":0},
            "max_drawdown_pct": max_dd,
            "max_losing_streak": int(max_streak),
            "worst_trade_pct": worst,
            "exit_reasons": part["exit_reason"].value_counts().to_dict() if not part.empty else {},
        })

    yearly_compare_stop18 = {}
    for stop_label, stop_value in [("none", None), ("-18", -18.0)]:
        dated = []
        for g in groups:
            sim = simulate(g, stop_pct=stop_value, max_hold=16, use_gakutto=False)
            if not sim:
                continue
            entry_row = g[g["offset"] == 0]
            if entry_row.empty:
                continue
            dated.append({
                "entry_date": str(entry_row.iloc[0]["entry_date"]),
                "return_pct": float(sim["return_pct"]),
            })
        ddf = pd.DataFrame(dated)
        ddf["entry_date_dt"] = pd.to_datetime(ddf["entry_date"])
        per_year = {}
        for year, part in ddf.groupby(ddf["entry_date_dt"].dt.year):
            part = part.sort_values("entry_date_dt").reset_index(drop=True)
            eq = (1 + part["return_pct"] / 100.0).cumprod()
            peak = eq.cummax()
            dd = eq / peak - 1
            max_streak = 0
            cur = 0
            for r in part["return_pct"]:
                if r < 0:
                    cur += 1
                    max_streak = max(max_streak, cur)
                else:
                    cur = 0
            per_year[str(int(year))] = {
                "n": int(len(part)),
                "avg_return_pct": round(float(part["return_pct"].mean()), 3),
                "median_return_pct": round(float(part["return_pct"].median()), 3),
                "max_drawdown_pct": round(float(dd.min() * 100), 2),
                "max_losing_streak": int(max_streak),
                "worst_trade_pct": round(float(part["return_pct"].min()), 3),
                "compounded_return_pct": round(float((eq.iloc[-1] - 1) * 100), 2),
            }
        yearly_compare_stop18[stop_label] = per_year

    # Exact 16-session feature diagnostics using cached returns joined to entry features.
    feature_path = RESULT_DIR / "kuitto_runner_features_trades.csv"
    exact16_feature_diag = None
    if feature_path.exists():
        feat = pd.read_csv(feature_path)
        feat["code"] = feat["code"].astype(str)
        feat["signal_date"] = feat["signal_date"].astype(str)

        exact_rows = []
        for g in groups:
            sim = simulate(g, stop_pct=None, max_hold=16, use_gakutto=False)
            if not sim:
                continue
            first = g.iloc[0]
            exact_rows.append({
                "code": str(first["code"]),
                "signal_date": str(first["signal_date"]),
                "ret16": float(sim["return_pct"]),
            })
        exact = pd.DataFrame(exact_rows)
        merged = exact.merge(feat, on=["code","signal_date"], how="left")
        merged["year"] = pd.to_datetime(merged["signal_date"]).dt.year

        cols = [
            "bull_pct","ma5_decline5_pct","ma5_vs_ma25_pct","close_vs_ma5_pct",
            "volume_ratio","ma25_slope5_pct","dd20_pct","dd60_pct","ret5_pct",
            "ret20_pct","atr14_pct","avg_turnover20"
        ]
        weak_years = [2018, 2022, 2023]

        def fs(part):
            out = {"n": int(len(part))}
            for c in cols:
                if c not in part.columns:
                    continue
                x = pd.to_numeric(part[c], errors="coerce").dropna()
                if len(x):
                    out[c] = {
                        "mean": round(float(x.mean()), 4),
                        "median": round(float(x.median()), 4),
                        "q25": round(float(x.quantile(0.25)), 4),
                        "q75": round(float(x.quantile(0.75)), 4),
                    }
            return out

        weak = merged[merged["year"].isin(weak_years)]
        other = merged[~merged["year"].isin(weak_years)]
        exact16_feature_diag = {
            "weak_years": weak_years,
            "weak_losses": fs(weak[weak["ret16"] < 0]),
            "weak_wins": fs(weak[weak["ret16"] >= 0]),
            "other_losses": fs(other[other["ret16"] < 0]),
            "other_wins": fs(other[other["ret16"] >= 0]),
        }

    entry_strength_grid = []
    feature_path = RESULT_DIR / "kuitto_runner_features_trades.csv"
    if feature_path.exists():
        feat = pd.read_csv(feature_path)
        feat["code"] = feat["code"].astype(str)
        feat["signal_date"] = feat["signal_date"].astype(str)

        ret16_rows = []
        for g in groups:
            sim = simulate(g, stop_pct=None, max_hold=16, use_gakutto=False)
            if not sim:
                continue
            first = g.iloc[0]
            ret16_rows.append({
                "code": str(first["code"]),
                "signal_date": str(first["signal_date"]),
                "entry_date": str(first["entry_date"]),
                "return_pct": float(sim["return_pct"]),
            })

        ret16 = pd.DataFrame(ret16_rows)
        merged16 = ret16.merge(feat, on=["code","signal_date"], how="left")
        merged16["year"] = pd.to_datetime(merged16["signal_date"]).dt.year

        for decline_max in [-3.0, -3.5, -4.0, -4.5]:
            for close_min in [1.5, 2.0, 2.5]:
                p = merged16[
                    (merged16["ma5_decline5_pct"] <= decline_max)
                    & (merged16["close_vs_ma5_pct"] >= close_min)
                ].copy()

                weak = p[p["year"].isin([2018, 2022, 2023])].sort_values("entry_date")
                if not weak.empty:
                    eq = (1 + weak["return_pct"] / 100.0).cumprod()
                    peak = eq.cummax()
                    dd = eq / peak - 1
                    max_dd_weak = round(float(dd.min() * 100), 2)
                    streak = 0
                    cur = 0
                    for r in weak["return_pct"]:
                        if r < 0:
                            cur += 1
                            streak = max(streak, cur)
                        else:
                            cur = 0
                else:
                    max_dd_weak = None
                    streak = 0

                entry_strength_grid.append({
                    "ma5_decline5_max": decline_max,
                    "close_vs_ma5_min": close_min,
                    "overall": metrics(p[["return_pct"]].assign(hold_sessions_to_exit=16)) if not p.empty else {"n":0},
                    "weak_years": metrics(weak[["return_pct"]].assign(hold_sessions_to_exit=16)) if not weak.empty else {"n":0},
                    "weak_years_max_drawdown_pct": max_dd_weak,
                    "weak_years_max_losing_streak": int(streak),
                    "yearly_counts": {str(int(y)): int(len(gp)) for y, gp in p.groupby("year")},
                })

    out = {
        "source_cache": str(CACHE),
        "signal_count": len(groups),
        "fixed_hold_grid": fixed_holds,
        "exit_grid": exit_grid,
        "risk_16d": {"overall": overall_risk_16, "yearly": yearly_risk},
        "emergency_stop_grid": emergency_stop_grid,
        "yearly_compare_stop18": yearly_compare_stop18,
        "exact16_feature_diag": exact16_feature_diag,
        "entry_strength_grid": entry_strength_grid,
    }
    out_path = RESULT_DIR / "kuitto_exit_cache_summary.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
