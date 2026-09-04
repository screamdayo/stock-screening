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

通常シグナルとは別に、バックテストで検証した限定救済シグナルも日次表示できる。
救済条件:
- 通常条件では非該当
- 直近2日のMA5傾きのうち正の傾きは1回だけ
- その微上昇は +0.1% 以下
- もう1回は 0% 以下
- 当日のMA5は上向き
- 当日の出来高が前日の1.5倍以上
- 強陽線など既存条件はそのまま必要
"""

import pandas as pd

import config
import indicator
from logger import get_logger

logger = get_logger(__name__)

STRATEGY_NAME = "ma5_breakout"
TYPE_BOTTOM = "bottom_reversal"
TYPE_PULLBACK = "pullback_reacceleration"
TYPE_RESCUE = "rescue"
LABEL_BOTTOM = "大底反転"
LABEL_PULLBACK = "押し目再上昇"
LABEL_RESCUE_BOTTOM = "救済候補（大底反転）"
LABEL_RESCUE_PULLBACK = "救済候補（押し目再上昇）"
RESCUE_MICRO_RISE_MAX_PCT = 0.1
RESCUE_VOLUME_RATIO_MIN = 1.5


def _get_cfg():
    return config.get_strategy_config(STRATEGY_NAME)


def _passes_common_candle_conditions(g, idx, cfg):
    latest = g.iloc[idx]

    if cfg["REQUIRE_BULLISH_CANDLE"] and not indicator.is_bullish_candle(latest):
        return False

    if cfg["REQUIRE_STRONG_BULLISH_CANDLE"] and not indicator.is_strong_bullish_candle(
        latest, min_gain_pct=cfg["MA5_STRONG_CANDLE_MIN_GAIN_PCT"]
    ):
        return False

    return True


def _classify_after_breakout(g, idx, cfg):
    below_long = indicator.is_ma_short_below_long_at(g, idx)

    if below_long:
        declined = indicator.is_ma_decline_before_turn_at(
            g, idx,
            decline_lookback_days=cfg["MA5_DECLINE_LOOKBACK_DAYS"],
            decline_max_pct=cfg["MA5_DECLINE_MAX_PCT"],
        )
        if declined:
            return TYPE_BOTTOM, LABEL_BOTTOM
        return None

    return TYPE_PULLBACK, LABEL_PULLBACK


def _classify_signal(g, idx, cfg):
    """通常条件通過なら (signal_type, signal_label)、非該当なら None を返す。"""
    if not _passes_common_candle_conditions(g, idx, cfg):
        return None

    if not indicator.is_ma_breakout_at(
        g, idx, lookback_days=cfg["MA5_BREAKOUT_LOOKBACK_DAYS"]
    ):
        return None

    return _classify_after_breakout(g, idx, cfg)


def _matches_rescue_ma_pattern(g, idx, lookback_days):
    """通常の厳密MA判定から外れたものだけ、限定救済パターンを判定する。"""
    required_start_idx = idx - lookback_days - 1
    if required_start_idx < 0 or idx >= len(g):
        return False

    ma_window = g["MA_SHORT"].iloc[required_start_idx: idx + 1]
    if ma_window.isna().any():
        return False

    values = ma_window.tolist()
    today_prev = values[-2]
    today = values[-1]
    if today_prev == 0 or today <= today_prev:
        return False

    past_values = values[:-1]
    past_changes_pct = []
    for i in range(1, len(past_values)):
        prev = past_values[i - 1]
        curr = past_values[i]
        if prev == 0:
            return False
        past_changes_pct.append((curr / prev - 1) * 100)

    recent = past_changes_pct[-lookback_days:]
    if len(recent) != lookback_days:
        return False

    positive = [change for change in recent if change > 0]
    non_positive = [change for change in recent if change <= 0]

    if len(positive) != 1 or len(non_positive) != lookback_days - 1:
        return False
    if positive[0] > RESCUE_MICRO_RISE_MAX_PCT:
        return False

    if "Vo" not in g.columns or idx - 1 < 0:
        return False
    prev_volume = g["Vo"].iloc[idx - 1]
    today_volume = g["Vo"].iloc[idx]
    if pd.isna(prev_volume) or pd.isna(today_volume) or prev_volume <= 0:
        return False
    if today_volume < prev_volume * RESCUE_VOLUME_RATIO_MIN:
        return False

    return True


def _classify_rescue_signal(g, idx, cfg):
    """通常シグナルではないものだけを、救済候補として分類する。"""
    if _classify_signal(g, idx, cfg):
        return None

    if not _passes_common_candle_conditions(g, idx, cfg):
        return None

    lookback_days = cfg["MA5_BREAKOUT_LOOKBACK_DAYS"]
    if not _matches_rescue_ma_pattern(g, idx, lookback_days):
        return None

    classified = _classify_after_breakout(g, idx, cfg)
    if not classified:
        return None

    base_type, _ = classified
    if base_type == TYPE_BOTTOM:
        return TYPE_RESCUE, LABEL_RESCUE_BOTTOM, TYPE_BOTTOM
    return TYPE_RESCUE, LABEL_RESCUE_PULLBACK, TYPE_PULLBACK


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
    """バックテスト用: 通常条件の全銘柄・全日付シグナルを返す。"""
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


def _build_latest_result(code, g, idx, signal_type, signal_label, rescue_base_type=None):
    latest = g.iloc[idx]
    result = {
        "code": code,
        "close": round(latest["C"], 1),
        "open": round(latest["O"], 1),
        "ma_short_today": round(g["MA_SHORT"].iloc[-1], 1),
        "ma_short_prev": round(g["MA_SHORT"].iloc[-2], 1),
        "ma_long_today": round(g["MA_LONG"].iloc[-1], 1),
        "signal_type": signal_type,
        "signal_label": signal_label,
    }
    if rescue_base_type:
        result["rescue_base_type"] = rescue_base_type
        result["rescue_volume_ratio_min"] = RESCUE_VOLUME_RATIO_MIN
    return result


def find_latest_signals(price_df, target_codes):
    """日次スクリーニング用: 最新日の通常2分類シグナルを返す。"""
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

        classified = _classify_signal(g, idx, cfg)
        if not classified:
            continue

        signal_type, signal_label = classified
        if signal_type == TYPE_BOTTOM:
            cnt_bottom += 1
        else:
            cnt_pullback += 1

        results.append(_build_latest_result(code, g, idx, signal_type, signal_label))

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


def find_latest_rescue_signals(price_df, target_codes):
    """日次表示用: 通常条件から漏れた限定救済候補だけを返す。"""
    cfg = _get_cfg()
    results = []

    df = price_df[price_df["Code"].isin(target_codes)].copy()
    if df.empty:
        return results

    min_rows = _min_required_rows(cfg)

    for code, group in df.groupby("Code"):
        g = group.dropna(subset=["C", "O"]).sort_values("Date").reset_index(drop=True)
        if len(g) < min_rows:
            continue

        g = indicator.add_moving_averages(
            g,
            short_period=cfg["MA_SHORT_PERIOD"],
            long_period=cfg["MA_LONG_PERIOD"],
        )
        idx = len(g) - 1

        classified = _classify_rescue_signal(g, idx, cfg)
        if not classified:
            continue

        signal_type, signal_label, rescue_base_type = classified
        results.append(
            _build_latest_result(
                code, g, idx, signal_type, signal_label,
                rescue_base_type=rescue_base_type,
            )
        )

    logger.info(
        f"限定救済候補: {len(results)}件 "
        f"（微上昇+0.1%以下 / 出来高前日比{RESCUE_VOLUME_RATIO_MIN:.1f}倍以上）"
    )
    return results
