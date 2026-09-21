"""
目視判定なしの自動くいっと戦略。

2025-11-06〜2026-09-08 の保存済み株価で検証した「広め押し目版」に、
追加検証で有効だった出来高条件（当日出来高が前日の1.25倍以上）を加えた本番版。
2026-09-16以降は、5年バックテストで成績と年別安定性が改善した
20日平均売買代金5億円以上の流動性フィルターも適用する。
2026-09-21以降は候補を除外せず、疑似OOSで有効だった特徴を「伸びスコア」として付与する。

条件:
- 直近2日までMA5が下向き/横ばい
- 当日MA5が初めて上向き
- 直前5営業日のMA5下落率: -5.5%〜-2.0%
- MA5とMA25の乖離: -5.0%〜0%
- 終値とMA5の乖離: +4.0%以下
- 当日陽線: +1.5%〜+3.5%
- 当日出来高: 前日比1.25倍以上
- 2026-09-16以降: 20日平均売買代金（終値×出来高）5億円以上
- 伸びスコア: ATR14>=2.92% を2点、DD20<=-5.86%を1点、20日平均売買代金>=8.28億円を1点

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
VOLUME_RATIO_MIN = 1.25
AVG_TURNOVER_20_MIN = 500_000_000
LIQUIDITY_EFFECTIVE_DATE = pd.Timestamp("2026-09-16")
RUNNER_ATR14_MIN = 2.917505
RUNNER_DD20_MAX = -5.855856
RUNNER_TURNOVER20_MIN = 828_434_090


def _prepare(group):
    g = group.dropna(subset=["C", "O"]).sort_values("Date").reset_index(drop=True).copy()
    for c in ["O", "H", "L", "C"]:
        if c in g.columns:
            g[c] = pd.to_numeric(g[c], errors="coerce")
    if "Vo" in g.columns:
        g["Vo"] = pd.to_numeric(g["Vo"], errors="coerce")
    else:
        g["Vo"] = pd.NA
    g["MA_SHORT"] = g["C"].rolling(MA_SHORT).mean()
    g["MA_LONG"] = g["C"].rolling(MA_LONG).mean()
    g["TURNOVER"] = g["C"] * g["Vo"]
    g["AVG_TURNOVER_20"] = g["TURNOVER"].rolling(20).mean()
    g["HIGH20"] = g["C"].rolling(20).max()
    g["DD20_PCT"] = (g["C"] / g["HIGH20"] - 1) * 100
    if "H" in g.columns and "L" in g.columns:
        tr = pd.concat([
            g["H"] - g["L"],
            (g["H"] - g["C"].shift(1)).abs(),
            (g["L"] - g["C"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        g["ATR14_PCT"] = tr.rolling(14).mean() / g["C"] * 100
    else:
        g["ATR14_PCT"] = pd.NA
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

    for j in range(idx - TURN_LOOKBACK, idx):
        if g["MA_SHORT"].iloc[j] > g["MA_SHORT"].iloc[j - 1]:
            return None

    if not g["MA_SHORT"].iloc[idx] > g["MA_SHORT"].iloc[idx - 1]:
        return None

    if not r["MA_SHORT"] > 0:
        return None
    close_ma5 = (r["C"] / r["MA_SHORT"] - 1) * 100
    if close_ma5 > CLOSE_VS_MA5_MAX_PCT:
        return None

    if idx < 1:
        return None
    prev_volume = g["Vo"].iloc[idx - 1]
    today_volume = r["Vo"]
    if pd.isna(prev_volume) or pd.isna(today_volume) or not float(prev_volume) > 0:
        return None
    volume_ratio = float(today_volume) / float(prev_volume)
    if volume_ratio < VOLUME_RATIO_MIN:
        return None

    avg_turnover_20 = r["AVG_TURNOVER_20"]
    signal_date = pd.Timestamp(g["Date"].iloc[idx])
    if signal_date >= LIQUIDITY_EFFECTIVE_DATE:
        if pd.isna(avg_turnover_20) or float(avg_turnover_20) < AVG_TURNOVER_20_MIN:
            return None

    atr14_pct = r["ATR14_PCT"]
    dd20_pct = r["DD20_PCT"]
    runner_score = 0
    if pd.notna(atr14_pct) and float(atr14_pct) >= RUNNER_ATR14_MIN:
        runner_score += 2
    if pd.notna(dd20_pct) and float(dd20_pct) <= RUNNER_DD20_MAX:
        runner_score += 1
    if pd.notna(avg_turnover_20) and float(avg_turnover_20) >= RUNNER_TURNOVER20_MIN:
        runner_score += 1

    return {
        "bull_candle_pct": bull,
        "ma5_prior5d_decline_pct": decline,
        "ma5_vs_ma25_pct": ma5_gap,
        "close_vs_ma5_pct": close_ma5,
        "volume_ratio": volume_ratio,
        "avg_turnover_20": float(avg_turnover_20) if pd.notna(avg_turnover_20) else None,
        "atr14_pct": float(atr14_pct) if pd.notna(atr14_pct) else None,
        "dd20_pct": float(dd20_pct) if pd.notna(dd20_pct) else None,
        "runner_score": runner_score,
    }


def find_signals(price_df, target_codes):
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

    logger.info(
        f"自動くいっと検出: {len(signals)}件（出来高前日比{VOLUME_RATIO_MIN:.2f}倍以上、"
        f"{LIQUIDITY_EFFECTIVE_DATE.date()}以降は20日平均売買代金{AVG_TURNOVER_20_MIN/100_000_000:.0f}億円以上）"
    )
    return signals, price_data_by_code


def find_latest_signals(price_df, target_codes):
    results = []
    df = price_df[price_df["Code"].isin(target_codes)].copy()
    if df.empty:
        return results

    latest_date = df["Date"].max()
    logger.info(f"データ最新日付: {latest_date}")

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
            **{k: (round(float(v), 3) if v is not None else None) for k, v in f.items()},
        })

    logger.info(
        f"目視なし・くいっと押し目版: {len(results)}件 "
        f"（出来高前日比{VOLUME_RATIO_MIN:.2f}倍以上、"
        f"{LIQUIDITY_EFFECTIVE_DATE.date()}以降は20日平均売買代金{AVG_TURNOVER_20_MIN/100_000_000:.0f}億円以上）"
    )
    return results
