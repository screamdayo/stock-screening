"""
目視判定なしの自動くいっと戦略。

2025-11-06〜2026-09-08 の保存済み株価で検証した「広め押し目版」を
そのまま実装する。

条件:
- 直近2日までMA5が下向き/横ばい
- 当日MA5が初めて上向き
- 直前5営業日のMA5下落率: -5.5%〜-2.0%
- MA5とMA25の乖離: -5.0%〜0%
- 終値とMA5の乖離: +4.0%以下
- 当日陽線: +1.5%〜+3.5%

売買の良し悪しを人のA/B/skipで選別せず、条件通過銘柄をそのまま扱う。
"""

import pandas as pd

from logger import get_logger

logger = get_logger(__name__)

STRATEGY_NAME = "kuitto_pullback_auto"
SIGNAL_TYPE = "kuitto_pullback_auto"
SIGNAL_LABEL = "くいっと押し目版"

MA_SHORT = 5
MA_LONG = 25
TURN_LOOKBACK = 2
DECLINE_LOOKBACK = 5
DECLINE_MIN_PCT = -5.5
DECLINE_MAX_PCT = -2.0
MA5_VS_MA25_MIN_PCT = -5.0
MA5_VS_MA25_MAX_PCT = 0.0
CLOSE_VS_MA5_MAX_PCT = 4.0
BULL_MIN_PCT = 1.5
BULL_MAX_PCT = 3.5


def _prepare(group):
    g = group.dropna(subset=["C", "O"]).sort_values("Date").reset_index(drop=True).copy()
    for c in ["O", "C"]:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    g["MA_SHORT"] = g["C"].rolling(MA_SHORT).mean()
    g["MA_LONG"] = g["C"].rolling(MA_LONG).mean()
    return g


def _features(g, idx):
    if idx < MA_LONG or idx >= len(g):
        return None

    r = g.iloc[idx]
    if pd.isna(r["MA_SHORT"]) or pd.isna(r["MA_LONG"]) or not r["O"] > 0:
        return None

    bull = (r["C"] / r["O"] - 1) * 100
    if not (BULL_MIN_PCT <= bull <= BULL_MAX_PCT):
        return None

    # 検証条件どおり、MA5はMA25以下かつ乖離-5〜0%。
    if not r["MA_LONG"] > 0:
        return None
    ma5_gap = (r["MA_SHORT"] / r["MA_LONG"] - 1) * 100
    if not (MA5_VS_MA25_MIN_PCT <= ma5_gap <= MA5_VS_MA25_MAX_PCT):
        return None

    base = idx - 1
    past = base - DECLINE_LOOKBACK
    if past < 0 or pd.isna(g["MA_SHORT"].iloc[past]) or not g["MA_SHORT"].iloc[past] > 0:
        return None
    decline = (g["MA_SHORT"].iloc[base] / g["MA_SHORT"].iloc[past] - 1) * 100
    if not (DECLINE_MIN_PCT <= decline <= DECLINE_MAX_PCT):
        return None

    # 直近2日までは上向きになっていない。
    for j in range(idx - TURN_LOOKBACK, idx):
        if g["MA_SHORT"].iloc[j] > g["MA_SHORT"].iloc[j - 1]:
            return None

    # 当日初めて上向き。
    if not g["MA_SHORT"].iloc[idx] > g["MA_SHORT"].iloc[idx - 1]:
        return None

    if not r["MA_SHORT"] > 0:
        return None
    close_ma5 = (r["C"] / r["MA_SHORT"] - 1) * 100
    if close_ma5 > CLOSE_VS_MA5_MAX_PCT:
        return None

    return {
        "bull_candle_pct": bull,
        "ma5_prior5d_decline_pct": decline,
        "ma5_vs_ma25_pct": ma5_gap,
        "close_vs_ma5_pct": close_ma5,
    }


def find_signals(price_df, target_codes):
    """バックテスト用: 全日付の自動押し目版シグナルを返す。"""
    signals = []
    price_data_by_code = {}
    df = price_df[price_df["Code"].isin(target_codes)].copy()
    if df.empty:
        return signals, price_data_by_code

    for code, group in df.groupby("Code"):
        g = _prepare(group)
        if len(g) <= MA_LONG:
            continue
        price_data_by_code[code] = g
        for idx in range(MA_LONG, len(g)):
            f = _features(g, idx)
            if not f:
                continue
            signals.append({
                "code": code,
                "signal_date": g["Date"].iloc[idx],
                "signal_idx": idx,
                "signal_type": SIGNAL_TYPE,
                "signal_label": SIGNAL_LABEL,
                **f,
            })

    logger.info(f"自動くいっと検出: {len(signals)}件")
    return signals, price_data_by_code


def find_latest_signals(price_df, target_codes):
    """日次スクリーニング用: 最新日の条件通過銘柄だけ返す。"""
    results = []
    df = price_df[price_df["Code"].isin(target_codes)].copy()
    if df.empty:
        return results

    latest_date = df["Date"].max()
    logger.info(f"データ最新日付: {latest_date.date()}")

    for code, group in df.groupby("Code"):
        g = _prepare(group)
        if len(g) <= MA_LONG:
            continue
        idx = len(g) - 1
        f = _features(g, idx)
        if not f:
            continue
        r = g.iloc[idx]
        results.append({
            "code": code,
            "close": round(float(r["C"]), 1),
            "open": round(float(r["O"]), 1),
            "ma_short_today": round(float(r["MA_SHORT"]), 1),
            "ma_short_prev": round(float(g["MA_SHORT"].iloc[idx - 1]), 1),
            "ma_long_today": round(float(r["MA_LONG"]), 1),
            "signal_type": SIGNAL_TYPE,
            "signal_label": SIGNAL_LABEL,
            "auto_filtered": True,
            "screening_bucket": "auto",
            **{k: round(float(v), 3) for k, v in f.items()},
        })

    logger.info(f"目視なし・くいっと押し目版: {len(results)}件")
    return results
