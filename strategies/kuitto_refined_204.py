"""Forward-only refined Kuitto strategy frozen on 2026-09-23.

This is intentionally separate from production kuitto_pullback_auto.
Backtest identity: the frozen base Kuitto shape WITHOUT the later liquidity filter,
plus the composite ATR/DD20/MA25-slope filter that produced 204 evaluable 16-day exits.
"""

import pandas as pd
from strategies import kuitto_pullback_auto as base

STRATEGY_NAME = "kuitto_refined_204"
SIGNAL_TYPE = STRATEGY_NAME
SIGNAL_LABEL = "くいっと204"

def _passes_refined(g, idx, f):
    r = g.iloc[idx]
    ma25 = g["MA_LONG"]
    if idx < 5 or pd.isna(r.get("ATR14_PCT")) or pd.isna(r.get("DD20_PCT")):
        return False
    prev = ma25.iloc[idx - 5]
    cur = ma25.iloc[idx]
    if pd.isna(prev) or pd.isna(cur) or not prev > 0:
        return False
    slope5 = (cur / prev - 1) * 100
    atr = float(r["ATR14_PCT"])
    dd20 = float(r["DD20_PCT"])
    if atr < 3.4:
        return False
    if -7.0 < dd20 <= -5.5:
        return True
    return -9.0 < dd20 <= -7.0 and slope5 >= -1.5

def _base_features_without_liquidity(g, idx):
    # Recreate the frozen base shape by calling the production feature function
    # on a pre-liquidity effective date, while retaining current price-derived features.
    original_date = g.loc[idx, "Date"]
    try:
        g.loc[idx, "Date"] = pd.Timestamp("2026-09-15")
        return base._features(g, idx)
    finally:
        g.loc[idx, "Date"] = original_date

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
        f = _base_features_without_liquidity(g, idx)
        if not f or not _passes_refined(g, idx, f):
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
    # Forward use is the primary purpose; historical backtests use dedicated scripts.
    results = []
    df = price_df[price_df["Code"].isin(target_codes)].copy()
    for code, group in df.groupby("Code"):
        g = base._prepare(group)
        for idx in range(base.MA_LONG, len(g)):
            f = _base_features_without_liquidity(g, idx)
            if f and _passes_refined(g, idx, f):
                results.append({"code": str(code), "signal_date": g["Date"].iloc[idx], "signal_idx": idx, **f})
    return results, {}
