"""
strategies/ma5_breakout.py

5日MAの反転を2分類で検出する。

1. 大底反転（bottom_reversal）
   - 強い陽線
   - 直近5日でMA5が十分下落
   - MA5 <= MA25
   - 直近2日までMA5が下向き/横ばいで、当日初めて上向き

2. 押し目再上昇（pullback_reacceleration）
   - 強い陽線
   - MA5 > MA25
   - 直近2日までMA5が下向き/横ばいで、当日初めて上向き

後者は、任天堂のように25日線より上でいったん押してから再加速する形を拾うための分類。
"""

import config
import indicator
from logger import get_logger

logger = get_logger(__name__)

STRATEGY_NAME = "ma5_breakout"
TYPE_BOTTOM = "bottom_reversal"
TYPE_PULLBACK = "pullback_reacceleration"
LABEL_BOTTOM = "大底反転"
LABEL_PULLBACK = "押し目再上昇"


def _get_cfg():
    return config.get_strategy_config(STRATEGY_NAME)


def _classify_signal(g, idx, cfg):
    """条件通過なら (signal_type, signal_label)、非該当なら None を返す。"""
    latest = g.iloc[idx]

    if cfg["REQUIRE_BULLISH_CANDLE"] and not indicator.is_bullish_candle(latest):
        return None

    if cfg["REQUIRE_STRONG_BULLISH_CANDLE"] and not indicator.is_strong_bullish_candle(
        latest, min_gain_pct=cfg["MA5_STRONG_CANDLE_MIN_GAIN_PCT"]
    ):
        return None

    # 両分類共通: 直近2日まで下向き/横ばい → 当日初めて上向き。
    if not indicator.is_ma_breakout_at(
        g, idx, lookback_days=cfg["MA5_BREAKOUT_LOOKBACK_DAYS"]
    ):
        return None

    below_long = indicator.is_ma_short_below_long_at(g, idx)

    # 1) 大底反転: 従来条件をそのまま維持。
    if below_long:
        declined = indicator.is_ma_decline_before_turn_at(
            g, idx,
            decline_lookback_days=cfg["MA5_DECLINE_LOOKBACK_DAYS"],
            decline_max_pct=cfg["MA5_DECLINE_MAX_PCT"],
        )
        if declined:
            return TYPE_BOTTOM, LABEL_BOTTOM
        return None

    # 2) 押し目再上昇: MA5 > MA25 の上昇トレンド側。
    # 下落率-0.5%条件は要求しない。短い押し目からの再加速も拾う。
    return TYPE_PULLBACK, LABEL_PULLBACK


def _min_required_rows(cfg):
    rows = max(cfg["MA_LONG_PERIOD"], cfg["MA_SHORT_PERIOD"])
    rows = max(
        rows,
        cfg["MA_SHORT_PERIOD"] + cfg["MA5_DECLINE_LOOKBACK_DAYS"] + 1,
    )
    rows = max(
        rows,
        cfg["MA_SHORT_PERIOD"] + cfg["MA5_BREAKOUT_LOOKBACK_DAYS"] + 1,
    )
    return rows + 1


def find_signals(price_df, target_codes):
    """バックテスト用: 全銘柄・全日付のシグナルを返す。"""
    cfg = _get_cfg()
    signals = []
    price_data_by_code = {}

    df = price_df[price_df["Code"].isin(target_codes)].copy()
    if df.empty:
        logger.warning("対象データが空です。")
        return signals, price_data_by_code

    min_rows = _min_required_rows(cfg)
    code_count = 0

    for code, group in df.groupby("Code"):
        g = group.dropna(subset=["C", "O"]).sort_values("Date").reset_index(drop=True)
        if len(g) < min_rows:
            continue

        g = indicator.add_moving_averages(
            g,
            short_period=cfg["MA_SHORT_PERIOD"],
            long_period=cfg["MA_LONG_PERIOD"],
        )
        code_count += 1
        price_data_by_code[code] = g

        for idx in range(min_rows, len(g)):
            classified = _classify_signal(g, idx, cfg)
            if not classified:
                continue
            signal_type, signal_label = classified
            signals.append({
                "code": code,
                "signal_date": g["Date"].iloc[idx],
                "signal_idx": idx,
                "signal_type": signal_type,
                "signal_label": signal_label,
            })

    logger.info(f"バックテスト対象銘柄数: {code_count}件")
    logger.info(f"検出したシグナル数: {len(signals)}件")
    return signals, price_data_by_code


def find_latest_signals(price_df, target_codes):
    """日次スクリーニング用: 最新日の2分類シグナルを返す。"""
    cfg = _get_cfg()
    results = []

    cnt_no_data = 0
    cnt_bottom = 0
    cnt_pullback = 0

    df = price_df[price_df["Code"].isin(target_codes)].copy()
    if df.empty:
        logger.warning("対象データが空です。ダウンロード結果を確認してください。")
        return results

    latest_date = df["Date"].max()
    logger.info(f"データ最新日付: {latest_date.date()}")

    min_rows = _min_required_rows(cfg)

    for code, group in df.groupby("Code"):
        g = group.dropna(subset=["C", "O"]).sort_values("Date").reset_index(drop=True)
        if len(g) < min_rows:
            cnt_no_data += 1
            continue

        g = indicator.add_moving_averages(
            g,
            short_period=cfg["MA_SHORT_PERIOD"],
            long_period=cfg["MA_LONG_PERIOD"],
        )
        idx = len(g) - 1
        latest = g.iloc[idx]

        classified = _classify_signal(g, idx, cfg)
        if not classified:
            continue

        signal_type, signal_label = classified
        if signal_type == TYPE_BOTTOM:
            cnt_bottom += 1
        else:
            cnt_pullback += 1

        results.append({
            "code": code,
            "close": round(latest["C"], 1),
            "open": round(latest["O"], 1),
            "ma_short_today": round(g["MA_SHORT"].iloc[-1], 1),
            "ma_short_prev": round(g["MA_SHORT"].iloc[-2], 1),
            "ma_long_today": round(g["MA_LONG"].iloc[-1], 1),
            "signal_type": signal_type,
            "signal_label": signal_label,
        })

    logger.info(
        "\n".join([
            "===== 2分類スクリーニング =====",
            f"データ不足: {cnt_no_data}件",
            f"大底反転: {cnt_bottom}件",
            f"押し目再上昇: {cnt_pullback}件",
            f"合計: {len(results)}件",
            "=============================",
        ])
    )

    return results
