"""
strategies/kuitto.py

「くいっと」戦略: 5日移動平均線が上向き ＆ 陽線 の銘柄を検出する。
条件の中身（何日線か、陽線を必須にするかなど）は
config.STRATEGY_CONFIG["kuitto"] で調整する。

この戦略単体で完結しており、backtest.py はこのファイルの中身を知らない。
新しい戦略を追加したいときは、このファイルをコピーして
find_signals() の中身と、config.STRATEGY_CONFIG への設定ブロック追加を行えばよい。
"""

import config
import indicator
from logger import get_logger

logger = get_logger(__name__)

STRATEGY_NAME = "kuitto"


def _get_cfg():
    """このファイル用の設定辞書を毎回取得する。
    run_backtest.py の --condition オプションなどで実行中に
    config.STRATEGY_CONFIG の中身が書き換わることがあるため、
    キャッシュせず毎回参照する。"""
    return config.get_strategy_config(STRATEGY_NAME)


def _passes_conditions(g, idx, cfg):
    """
    1銘柄の時系列DataFrame g の idx番目の行が、
    「くいっと」の条件（陽線・MA上向き・MA反転上向き・パターン全体）を満たすかどうかを判定する。
    """
    latest = g.iloc[idx]

    if cfg["REQUIRE_BULLISH_CANDLE"] and not indicator.is_bullish_candle(latest):
        return False

    if cfg["REQUIRE_MA_RISING"] and not indicator.is_ma_rising_at(g, idx):
        return False

    if cfg["REQUIRE_MA_TURNING"] and not indicator.is_ma_turning_up_at(
        g, idx, min_slope=cfg["MA_TURNING_MIN_SLOPE"]
    ):
        return False

    if cfg["REQUIRE_KUITTO_PATTERN"] and not indicator.is_kuitto_pattern_at(
        g, idx,
        lookback_days=cfg["KUITTO_PATTERN_LOOKBACK_DAYS"],
        downtrend_min_slope_pct=cfg["KUITTO_PATTERN_DOWNTREND_MIN_SLOPE_PCT"],
        turn_min_slope_pct=cfg["KUITTO_PATTERN_TURN_MIN_SLOPE_PCT"],
        require_flat=cfg["KUITTO_PATTERN_REQUIRE_FLAT"],
        flat_max_abs_slope_pct=cfg["KUITTO_PATTERN_FLAT_MAX_ABS_SLOPE_PCT"],
    ):
        return False

    return True


def find_signals(price_df, target_codes):
    """
    全銘柄・全日付について「くいっと」条件を満たす日をすべて洗い出す。
    strategies/base.py で定義されたインターフェースに従う。

    戻り値: (signals, price_data_by_code) のタプル
        signals: シグナルのリスト。各要素は以下の3キーのみを持つ軽量なdict。
            code: 銘柄コード
            signal_date: シグナル発生日（Timestamp）
            signal_idx: その銘柄のDataFrame内でのインデックス（0始まり）
        price_data_by_code: {銘柄コード: 指標計算済みDataFrame} の辞書。
            同じ銘柄のDataFrameを1つだけ保持する（シグナルの数だけ複製しない）。
            backtest.py 側はこの辞書と signals の code を突き合わせて
            該当銘柄のDataFrameを参照する。
    """
    cfg = _get_cfg()
    signals = []
    price_data_by_code = {}

    df = price_df[price_df["Code"].isin(target_codes)].copy()
    if df.empty:
        logger.warning("対象データが空です。")
        return signals, price_data_by_code

    min_required_rows = max(cfg["MA_LONG_PERIOD"], cfg["MA_SHORT_PERIOD"])
    if cfg["REQUIRE_KUITTO_PATTERN"]:
        min_required_rows = max(min_required_rows, cfg["MA_SHORT_PERIOD"] + cfg["KUITTO_PATTERN_LOOKBACK_DAYS"])
    min_required_rows += 1
    code_count = 0

    for code, group in df.groupby("Code"):
        g = group.dropna(subset=["C", "O"]).sort_values("Date").reset_index(drop=True)

        if len(g) < min_required_rows:
            continue

        g = indicator.add_moving_averages(
            g, short_period=cfg["MA_SHORT_PERIOD"], long_period=cfg["MA_LONG_PERIOD"]
        )
        code_count += 1

        # この銘柄のDataFrameは1つだけ辞書に保持する
        price_data_by_code[code] = g

        for idx in range(min_required_rows, len(g)):
            if _passes_conditions(g, idx, cfg):
                signals.append({
                    "code": code,
                    "signal_date": g["Date"].iloc[idx],
                    "signal_idx": idx,
                })

    logger.info(f"バックテスト対象銘柄数: {code_count}件")
    logger.info(f"検出したシグナル数: {len(signals)}件")

    return signals, price_data_by_code


def find_latest_signals(price_df, target_codes):
    """
    【日次スクリーニング用】
    最新日についてのみ「くいっと」条件を判定し、該当銘柄を返す。
    main.py（日次実行）から呼ばれる。

    戻り値: dictのリスト（code, close, open, ma_short_today, ma_short_prevを含む）
    """
    cfg = _get_cfg()
    results = []

    cnt_no_data = 0
    cnt_not_bullish = 0
    cnt_ma_not_rising = 0
    cnt_ma_not_turning = 0
    cnt_not_kuitto_pattern = 0
    cnt_pass = 0

    df = price_df[price_df["Code"].isin(target_codes)].copy()
    if df.empty:
        logger.warning("対象データが空です。ダウンロード結果を確認してください。")
        return results

    latest_date = df["Date"].max()
    logger.info(f"データ最新日付: {latest_date.date()}")

    # REQUIRE_KUITTO_PATTERN使用時は、傾き算出のために追加でlookback_days分の
    # データが必要になるため、必要行数に加味しておく
    min_required_rows = max(cfg["MA_LONG_PERIOD"], cfg["MA_SHORT_PERIOD"])
    if cfg["REQUIRE_KUITTO_PATTERN"]:
        min_required_rows = max(min_required_rows, cfg["MA_SHORT_PERIOD"] + cfg["KUITTO_PATTERN_LOOKBACK_DAYS"])
    min_required_rows += 1

    for code, group in df.groupby("Code"):
        g = group.dropna(subset=["C", "O"]).sort_values("Date").reset_index(drop=True)

        if len(g) < min_required_rows:
            cnt_no_data += 1
            continue

        g = indicator.add_moving_averages(
            g, short_period=cfg["MA_SHORT_PERIOD"], long_period=cfg["MA_LONG_PERIOD"]
        )
        latest = g.iloc[-1]
        idx = len(g) - 1

        bullish = indicator.is_bullish_candle(latest)
        if cfg["REQUIRE_BULLISH_CANDLE"] and not bullish:
            cnt_not_bullish += 1
            continue

        ma_rising = indicator.is_ma_rising_at(g, idx)
        if cfg["REQUIRE_MA_RISING"] and not ma_rising:
            cnt_ma_not_rising += 1
            continue

        ma_turning = indicator.is_ma_turning_up_at(g, idx, min_slope=cfg["MA_TURNING_MIN_SLOPE"])
        if cfg["REQUIRE_MA_TURNING"] and not ma_turning:
            cnt_ma_not_turning += 1
            continue

        kuitto_pattern = indicator.is_kuitto_pattern_at(
            g, idx,
            lookback_days=cfg["KUITTO_PATTERN_LOOKBACK_DAYS"],
            downtrend_min_slope_pct=cfg["KUITTO_PATTERN_DOWNTREND_MIN_SLOPE_PCT"],
            turn_min_slope_pct=cfg["KUITTO_PATTERN_TURN_MIN_SLOPE_PCT"],
            require_flat=cfg["KUITTO_PATTERN_REQUIRE_FLAT"],
            flat_max_abs_slope_pct=cfg["KUITTO_PATTERN_FLAT_MAX_ABS_SLOPE_PCT"],
        )
        if cfg["REQUIRE_KUITTO_PATTERN"] and not kuitto_pattern:
            cnt_not_kuitto_pattern += 1
            continue

        cnt_pass += 1
        results.append({
            "code": code,
            "close": round(latest["C"], 1),
            "open": round(latest["O"], 1),
            "ma_short_today": round(g["MA_SHORT"].iloc[-1], 1),
            "ma_short_prev": round(g["MA_SHORT"].iloc[-2], 1),
        })

    diagnostic_lines = [
        "===== 診断結果 =====",
        f"データ不足で除外: {cnt_no_data}件",
        f"陽線条件で除外: {cnt_not_bullish}件",
        f"MA上向き条件で除外: {cnt_ma_not_rising}件",
        f"MA反転条件で除外: {cnt_ma_not_turning}件",
        f"くいっとパターン条件で除外: {cnt_not_kuitto_pattern}件",
        f"両条件通過: {cnt_pass}件",
        "=====================",
    ]
    logger.info("\n".join(diagnostic_lines))

    return results
