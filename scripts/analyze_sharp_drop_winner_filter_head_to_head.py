import json
from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path("results/reversal_signal_baseline_trades.csv")
OUT = Path("results/sharp_drop_winner_filter_head_to_head.json")

def metrics(df):
    s = pd.to_numeric(df["ret_15d_pct"], errors="coerce").dropna()
    if s.empty:
        return {"n": 0}
    pos = s[s > 0]
    neg = s[s < 0]
    gross_loss = -neg.sum()
    eq = (1 + s / 100.0).cumprod()
    peak = eq.cummax()
    dd = (eq / peak - 1.0) * 100.0
    return {
        "n": int(len(s)),
        "win_rate_pct": round((s > 0).mean() * 100, 2),
        "hit_5pct_rate": round((s >= 5).mean() * 100, 2),
        "hit_10pct_rate": round((s >= 10).mean() * 100, 2),
        "avg_return_pct": round(s.mean(), 3),
        "median_return_pct": round(s.median(), 3),
        "profit_factor": round(pos.sum() / gross_loss, 3) if gross_loss > 0 else None,
        "max_compound_dd_pct": round(dd.min(), 2),
    }

def yearly(df):
    z = df.copy()
    z["year"] = z["signal_date"].dt.year
    return {str(int(y)): metrics(g) for y, g in z.groupby("year")}

def main():
    t = pd.read_csv(SRC, dtype={"code": str})
    t = t[t["type"] == "sharp_drop_reversal"].copy()
    t["signal_date"] = pd.to_datetime(t["signal_date"])
    for c in ["atr14_pct", "dd20_pct", "ret_15d_pct"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t = t.dropna(subset=["signal_date", "atr14_pct", "dd20_pct", "ret_15d_pct"]).sort_values("signal_date")

    # Fixed candidate rules discovered in the preceding exploratory pass.
    rules = {
        "stable_atr2.8_dd20m5.5": (t["atr14_pct"] >= 2.8) & (t["dd20_pct"] <= -5.5),
        "aggressive_atr3.0_dd20m5.5": (t["atr14_pct"] >= 3.0) & (t["dd20_pct"] <= -5.5),
    }

    # Date split: older vs recent half of the available 10y-ish archive.
    split = pd.Timestamp("2021-09-20")

    out = {
        "source_n": int(len(t)),
        "split_date": str(split.date()),
        "rules": {},
        "incremental_band": {},
        "direct_comparison": {},
        "note": (
            "Pseudo head-to-head with thresholds fixed before evaluation. "
            "Compare full period, older/recent split, yearly stability, and the incremental ATR 2.8-3.0 band."
        ),
    }

    for name, mask in rules.items():
        p = t[mask].copy()
        out["rules"][name] = {
            "overall": metrics(p),
            "older": metrics(p[p["signal_date"] < split]),
            "recent": metrics(p[p["signal_date"] >= split]),
            "yearly": yearly(p),
        }

    stable = rules["stable_atr2.8_dd20m5.5"]
    aggressive = rules["aggressive_atr3.0_dd20m5.5"]
    band = t[stable & ~aggressive].copy()
    out["incremental_band"] = {
        "definition": "stable-only: ATR 2.8 <= ATR < 3.0, DD20 <= -5.5",
        "overall": metrics(band),
        "older": metrics(band[band["signal_date"] < split]),
        "recent": metrics(band[band["signal_date"] >= split]),
        "yearly": yearly(band),
    }

    a = out["rules"]["stable_atr2.8_dd20m5.5"]
    b = out["rules"]["aggressive_atr3.0_dd20m5.5"]
    out["direct_comparison"] = {
        "overall_avg_diff_stable_minus_aggressive": round(a["overall"]["avg_return_pct"] - b["overall"]["avg_return_pct"], 3),
        "overall_pf_diff_stable_minus_aggressive": round(a["overall"]["profit_factor"] - b["overall"]["profit_factor"], 3),
        "overall_hit5_diff_stable_minus_aggressive": round(a["overall"]["hit_5pct_rate"] - b["overall"]["hit_5pct_rate"], 2),
        "recent_avg_diff_stable_minus_aggressive": round(a["recent"]["avg_return_pct"] - b["recent"]["avg_return_pct"], 3),
        "recent_pf_diff_stable_minus_aggressive": round(a["recent"]["profit_factor"] - b["recent"]["profit_factor"], 3),
        "recent_hit5_diff_stable_minus_aggressive": round(a["recent"]["hit_5pct_rate"] - b["recent"]["hit_5pct_rate"], 2),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
