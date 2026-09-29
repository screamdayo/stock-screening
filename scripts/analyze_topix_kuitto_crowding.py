import json
from pathlib import Path

import numpy as np
import pandas as pd

TRADES = Path("results/kuitto_runner_features_trades.csv")
TOPIX = Path("data/topix/topix_daily.parquet")
OUT = Path("results/topix_kuitto_crowding.json")

def trade_metrics(x):
    r = pd.to_numeric(x["ret10"], errors="coerce").dropna()
    if r.empty:
        return {"n": 0}
    gp = r[r > 0].sum()
    gl = -r[r < 0].sum()
    return {
        "n": int(len(r)),
        "win_rate_pct": round(float((r > 0).mean() * 100), 2),
        "avg_ret10_pct": round(float(r.mean()), 3),
        "median_ret10_pct": round(float(r.median()), 3),
        "p5_plus_pct": round(float((r >= 5).mean() * 100), 2),
        "p10_plus_pct": round(float((r >= 10).mean() * 100), 2),
        "profit_factor": round(float(gp / gl), 3) if gl > 0 else None,
    }

def main():
    trades = pd.read_csv(TRADES, dtype={"code": str})
    trades["signal_date"] = pd.to_datetime(trades["signal_date"]).dt.normalize()

    tx = pd.read_parquet(TOPIX).copy()
    tx["date"] = pd.to_datetime(tx["Date"]).dt.normalize()
    for c in ["O", "H", "L", "C"]:
        tx[c] = pd.to_numeric(tx[c], errors="coerce")
    tx = tx.dropna(subset=["date", "O", "C"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    tx["MA5"] = tx["C"].rolling(5).mean()
    tx["MA25"] = tx["C"].rolling(25).mean()
    tx["bull_pct"] = (tx["C"] / tx["O"] - 1) * 100
    tx["ma5_vs_ma25_pct"] = (tx["MA5"] / tx["MA25"] - 1) * 100
    tx["close_vs_ma5_pct"] = (tx["C"] / tx["MA5"] - 1) * 100
    tx["ma5_decline5_pct"] = (tx["MA5"].shift(1) / tx["MA5"].shift(6) - 1) * 100

    s1 = tx["MA5"] <= tx["MA5"].shift(1)
    s2 = tx["MA5"].shift(1) <= tx["MA5"].shift(2)
    turn = tx["MA5"] > tx["MA5"].shift(1)
    bullish = tx["C"] > tx["O"]
    core = s1 & s2 & turn & bullish

    tx["topix_core"] = core
    tx["topix_core_below25"] = core & (tx["ma5_vs_ma25_pct"] <= 0)
    tx["topix_pullback_shape"] = (
        core
        & tx["bull_pct"].between(1.5, 3.5, inclusive="both")
        & tx["ma5_decline5_pct"].between(-5.5, -2.0, inclusive="both")
        & tx["ma5_vs_ma25_pct"].between(-5.0, 0.0, inclusive="both")
        & (tx["close_vs_ma5_pct"] <= 4.0)
    )

    start = max(trades["signal_date"].min(), tx["date"].min())
    end = min(trades["signal_date"].max(), tx["date"].max())
    cal = tx[(tx["date"] >= start) & (tx["date"] <= end)].copy().reset_index(drop=True)

    daily_counts = trades.groupby("signal_date").size().rename("signal_count")
    cal = cal.merge(daily_counts, left_on="date", right_index=True, how="left")
    cal["signal_count"] = cal["signal_count"].fillna(0).astype(int)

    defs = {
        "core": "MA5 prior two slopes down/flat -> first upturn today, and TOPIX bullish candle",
        "core_below25": "core + MA5 <= MA25",
        "pullback_shape": "core + individual Kuitto price-shape thresholds; volume omitted because TOPIX index has no volume",
    }
    colmap = {
        "core": "topix_core",
        "core_below25": "topix_core_below25",
        "pullback_shape": "topix_pullback_shape",
    }

    overall_day_mean = float(cal["signal_count"].mean())
    overall_any = float((cal["signal_count"] > 0).mean() * 100)
    out = {
        "period": {"start": str(start.date()), "end": str(end.date()), "trading_days": int(len(cal))},
        "individual_kuitto": {
            "definition": "frozen kuitto_pullback_auto signals used by kuitto_runner_features_trades.csv",
            "signals": int(len(trades)),
            "overall_daily_mean": round(overall_day_mean, 3),
            "overall_any_signal_day_pct": round(overall_any, 2),
            "overall_trade_metrics": trade_metrics(trades),
        },
        "topix_definitions": defs,
        "tests": {},
    }

    for name, col in colmap.items():
        event_idx = cal.index[cal[col].fillna(False)].tolist()
        event_dates = set(cal.loc[event_idx, "date"])
        non = cal[~cal[col].fillna(False)]
        event = cal[cal[col].fillna(False)]

        event_mean = float(event["signal_count"].mean()) if len(event) else np.nan
        non_mean = float(non["signal_count"].mean()) if len(non) else np.nan
        ratio = event_mean / non_mean if np.isfinite(event_mean) and non_mean > 0 else np.nan

        same_trades = trades[trades["signal_date"].isin(event_dates)]
        other_trades = trades[~trades["signal_date"].isin(event_dates)]

        lead_lag = []
        for off in range(-3, 4):
            vals = []
            for i in event_idx:
                j = i + off
                if 0 <= j < len(cal):
                    vals.append(int(cal.at[j, "signal_count"]))
            lead_lag.append({
                "offset_sessions": off,
                "n_event_windows": int(len(vals)),
                "mean_signal_count": round(float(np.mean(vals)), 3) if vals else None,
                "median_signal_count": round(float(np.median(vals)), 3) if vals else None,
                "relative_to_overall_mean": round(float(np.mean(vals) / overall_day_mean), 3) if vals and overall_day_mean > 0 else None,
            })

        yearly = []
        for y, gy in cal.groupby(cal["date"].dt.year):
            ey = gy[gy[col].fillna(False)]
            ny = gy[~gy[col].fillna(False)]
            if ey.empty:
                continue
            em = float(ey["signal_count"].mean())
            nm = float(ny["signal_count"].mean()) if len(ny) else np.nan
            yearly.append({
                "year": int(y),
                "topix_kuitto_days": int(len(ey)),
                "event_mean_signal_count": round(em, 3),
                "non_event_mean_signal_count": round(nm, 3) if np.isfinite(nm) else None,
                "ratio": round(em / nm, 3) if np.isfinite(nm) and nm > 0 else None,
            })

        out["tests"][name] = {
            "topix_kuitto_days": int(len(event)),
            "day_frequency_pct": round(float(len(event) / len(cal) * 100), 2) if len(cal) else None,
            "event_day": {
                "mean_signal_count": round(event_mean, 3) if np.isfinite(event_mean) else None,
                "median_signal_count": round(float(event["signal_count"].median()), 3) if len(event) else None,
                "any_signal_day_pct": round(float((event["signal_count"] > 0).mean() * 100), 2) if len(event) else None,
            },
            "non_event_day": {
                "mean_signal_count": round(non_mean, 3) if np.isfinite(non_mean) else None,
                "median_signal_count": round(float(non["signal_count"].median()), 3) if len(non) else None,
                "any_signal_day_pct": round(float((non["signal_count"] > 0).mean() * 100), 2) if len(non) else None,
            },
            "event_vs_non_event_mean_ratio": round(float(ratio), 3) if np.isfinite(ratio) else None,
            "lead_lag": lead_lag,
            "trade_metrics_on_event_day": trade_metrics(same_trades),
            "trade_metrics_other_days": trade_metrics(other_trades),
            "yearly": yearly,
            "event_dates_top20_by_signal_count": [
                {"date": str(r["date"].date()), "signal_count": int(r["signal_count"])}
                for _, r in event.nlargest(20, "signal_count").iterrows()
            ],
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
