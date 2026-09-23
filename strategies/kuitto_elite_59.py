"""Forward-only elite Kuitto strategy frozen on 2026-09-23.

Separate strategy from both production Kuitto and Kuitto 204.
Adds stronger pullback/rebound requirements to the Kuitto 204 rule.
"""

from strategies import kuitto_refined_204 as refined
from strategies import kuitto_pullback_auto as base

STRATEGY_NAME = "kuitto_elite_59"
SIGNAL_TYPE = STRATEGY_NAME
SIGNAL_LABEL = "くいっと59"

def find_latest_signals(price_df, target_codes):
    out = []
    df = price_df[price_df["Code"].isin(target_codes)].copy()
    if df.empty:
        return out
    for code, group in df.groupby("Code"):
        g = base._prepare(group)
        if len(g) <= base.MA_LONG:
            continue
        idx = len(g) - 1
        f = refined._base_features_without_liquidity(g, idx)
        if not f or not refined._passes_refined(g, idx, f):
            continue
        if float(f["ma5_prior5d_decline_pct"]) > -4.0:
            continue
        if float(f["close_vs_ma5_pct"]) < 2.0:
            continue
        r = g.iloc[idx]
        ma25_slope5 = (g["MA_LONG"].iloc[idx] / g["MA_LONG"].iloc[idx-5] - 1) * 100
        out.append({
            "code": str(code),
            "close": round(float(r["C"]), 1),
            "signal_type": SIGNAL_TYPE,
            "signal_label": SIGNAL_LABEL,
            "ma25_slope5_pct": round(float(ma25_slope5), 3),
            **{k: (round(float(v), 3) if v is not None else None) for k, v in f.items()},
        })
    return out

def find_signals(price_df, target_codes):
    results = []
    df = price_df[price_df["Code"].isin(target_codes)].copy()
    for code, group in df.groupby("Code"):
        g = base._prepare(group)
        for idx in range(base.MA_LONG, len(g)):
            f = refined._base_features_without_liquidity(g, idx)
            if not f or not refined._passes_refined(g, idx, f):
                continue
            if float(f["ma5_prior5d_decline_pct"]) <= -4.0 and float(f["close_vs_ma5_pct"]) >= 2.0:
                results.append({"code": str(code), "signal_date": g["Date"].iloc[idx], "signal_idx": idx, **f})
    return results, {}
