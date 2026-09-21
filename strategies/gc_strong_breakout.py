"""
GC強ブレイク（日次監視用）。

10年検証で見つかった「深い下落後のGC2回目」に、
入口近傍耐性を確認した丸め条件を重ねた希少シグナル。

ベース条件:
- MA5がMA25をゴールデンクロス
- 60日高値からのDDが -20%〜-15%
- 深い下落エピソード内のGCが2回目
- MA25の5営業日傾き > +0.25%

強ブレイク条件（本番は過学習を避けるため丸め値）:
- 当日出来高 / 20日平均出来高 > 0.90
- MA5 / MA25乖離 > +0.10%
- 当日ローソク実体（終値/始値-1） > +2.00%

過去検証では年数件程度の希少シグナルなので、くいっととは別枠で通知する。
"""

import pandas as pd

from logger import get_logger

logger = get_logger(__name__)

STRATEGY_NAME = "gc_strong_breakout"
SIGNAL_TYPE = "gc_strong_breakout"
SIGNAL_LABEL = "GC強ブレイク"

MA_SHORT = 5
MA_LONG = 25
SLOPE_LOOKBACK = 5
MA25_SLOPE_MIN_PCT = 0.25
HIGH_LOOKBACK = 60
EPISODE_START_DD_PCT = -15.0
EPISODE_RESET_DD_PCT = -5.0
TARGET_DD_MIN_PCT = -20.0
TARGET_DD_MAX_PCT = -15.0

VOLUME20_RATIO_MIN = 0.90
GC_GAP_MIN_PCT = 0.10
BULL_CANDLE_MIN_PCT = 2.00


def _prepare(group):
    g = group.dropna(subset=["O", "C"]).sort_values("Date").reset_index(drop=True).copy()

    for c in ["O", "H", "L", "C", "Vo"]:
        if c in g.columns:
            g[c] = pd.to_numeric(g[c], errors="coerce")
        else:
            g[c] = pd.NA

    g["MA5"] = g["C"].rolling(MA_SHORT).mean()
    g["MA25"] = g["C"].rolling(MA_LONG).mean()
    g["MA25_SLOPE5_PCT"] = (g["MA25"] / g["MA25"].shift(SLOPE_LOOKBACK) - 1) * 100
    g["HIGH60"] = g["C"].rolling(HIGH_LOOKBACK, min_periods=HIGH_LOOKBACK).max()
    g["DD60_PCT"] = (g["C"] / g["HIGH60"] - 1) * 100
    g["GC_GAP_PCT"] = (g["MA5"] / g["MA25"] - 1) * 100
    g["VOL_RATIO20"] = g["Vo"] / g["Vo"].rolling(20).mean()
    g["BULL_CANDLE_PCT"] = (g["C"] / g["O"] - 1) * 100
    g["GC"] = (g["MA5"].shift(1) <= g["MA25"].shift(1)) & (g["MA5"] > g["MA25"])

    active = False
    seq = 0
    seqs = []
    for i in range(len(g)):
        dd = g["DD60_PCT"].iloc[i]
        if pd.isna(dd):
            seqs.append(0)
            continue

        if not active and dd <= EPISODE_START_DD_PCT:
            active = True
            seq = 0
        elif active and dd > EPISODE_RESET_DD_PCT:
            active = False
            seq = 0

        if active and bool(g["GC"].iloc[i]):
            seq += 1

        seqs.append(seq if active else 0)

    g["GC_SEQ"] = seqs
    return g


def _features(g, idx):
    if idx < HIGH_LOOKBACK or idx >= len(g):
        return None

    r = g.iloc[idx]
    required = [
        "MA5", "MA25", "MA25_SLOPE5_PCT", "DD60_PCT",
        "GC_GAP_PCT", "VOL_RATIO20", "BULL_CANDLE_PCT",
    ]
    if any(pd.isna(r[c]) for c in required):
        return None

    if not bool(r["GC"]):
        return None
    if int(r["GC_SEQ"]) != 2:
        return None
    if not float(r["MA25_SLOPE5_PCT"]) > MA25_SLOPE_MIN_PCT:
        return None

    dd = float(r["DD60_PCT"])
    if not (TARGET_DD_MIN_PCT <= dd < TARGET_DD_MAX_PCT):
        return None

    gap = float(r["GC_GAP_PCT"])
    volume_ratio20 = float(r["VOL_RATIO20"])
    bull = float(r["BULL_CANDLE_PCT"])

    if not gap > GC_GAP_MIN_PCT:
        return None
    if not volume_ratio20 > VOLUME20_RATIO_MIN:
        return None
    if not bull > BULL_CANDLE_MIN_PCT:
        return None

    return {
        "gc_sequence": 2,
        "dd60_pct": dd,
        "ma25_slope5_pct": float(r["MA25_SLOPE5_PCT"]),
        "gc_gap_pct": gap,
        "volume_ratio20": volume_ratio20,
        "bull_candle_pct": bull,
    }


def find_signals(price_df, target_codes):
    signals = []
    price_data_by_code = {}
    df = price_df[price_df["Code"].isin(target_codes)].copy()
    if df.empty:
        return signals, price_data_by_code

    for code, group in df.groupby("Code"):
        g = _prepare(group)
        if len(g) <= HIGH_LOOKBACK:
            continue
        price_data_by_code[code] = g

        for idx in range(HIGH_LOOKBACK, len(g)):
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

    logger.info(f"GC強ブレイク検出: {len(signals)}件")
    return signals, price_data_by_code


def find_latest_signals(price_df, target_codes):
    results = []
    df = price_df[price_df["Code"].isin(target_codes)].copy()
    if df.empty:
        return results

    latest_date = df["Date"].max()
    logger.info(f"GC強ブレイク判定日: {latest_date}")

    for code, group in df.groupby("Code"):
        g = _prepare(group)
        if len(g) <= HIGH_LOOKBACK:
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
            "ma5_today": round(float(r["MA5"]), 1),
            "ma25_today": round(float(r["MA25"]), 1),
            "signal_type": SIGNAL_TYPE,
            "signal_label": SIGNAL_LABEL,
            "screening_bucket": "gc_strong_breakout",
            **{k: round(float(v), 3) for k, v in f.items()},
        })

    logger.info(
        f"GC強ブレイク本日該当: {len(results)}件 "
        f"（出来高20日比>{VOLUME20_RATIO_MIN:.2f} / GC乖離>{GC_GAP_MIN_PCT:.2f}% / "
        f"陽線>{BULL_CANDLE_MIN_PCT:.2f}%）"
    )
    return results
