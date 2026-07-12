"""
strategies/ma5_breakout.py

「5日MAブレイクアウト」戦略:
    直近 MA5_BREAKOUT_LOOKBACK_DAYS 日間、5日移動平均線がずっと下向き・水平で、
    当日初めて上向きに転じた ＆ 当日が陽線、の銘柄を検出する。

kuitto戦略との違い:
    kuitto の REQUIRE_MA_TURNING は「前日1日分だけ」下向き・水平だったかを見るが、
    こちらは「直近lookback_days日間ずっと」下向き・水平が続いていたことを要求する、
    より厳格な「初めての転換」判定。

条件の中身（日数など）は config.STRATEGY_CONFIG["ma5_breakout"] で調整する。

この戦略単体で完結しており、backtest.py はこのファイルの中身を知らない。
"""

import config
import indicator
from logger import get_logger

logger = get_logger(__name__)

STRATEGY_NAME = "ma5_breakout"


def _get_cfg():
    """このファイル用の設定辞書を毎回取得する。
    run_backtest.py の --condition オプションなどで実行中に
    config.STRATEGY_CONFIG の中身が書き換わることがあるため、
    キャッシュせず毎回参照する。"""
    return config.get_strategy_config(STRATEGY_NAME)


def _passes_conditions(g, idx, cfg):
    """
    1銘柄の時系列DataFrame g の idx番目の行が、
    「5日MAブレイクアウト」の条件（陽線・MAブレイクアウト）を満たすかどうかを判定する。
    """
    latest = g.iloc[idx]

    if cfg["REQUIRE_BULLISH_CANDLE"] and not indicator.is_bullish_candle(latest):
        return False

    if not indicator.is_ma_breakout_at(
        g, idx, lookback_days=cfg["MA5_BREAKOUT_LOOKBACK_DAYS"]
    ):
        return False

    return True


def find_signals(price_df, target_codes):
    """
    全銘柄・全日付について「5日MAブレイクアウト」条件を満たす日をすべて洗い出す。
    strategies/base.py で定義されたインターフェースに従う。

    戻り値: (signals, price_data_by_code) のタプル
        signals: シグナルのリスト。各要素は以下の3キーのみを持つ軽量なdict。
            code: 銘柄コード
            signal_date: シグナル発生日（Timestamp）
            signal_idx: その銘柄のDataFrame内でのインデックス（0始まり）
        price_data_by_code: {銘柄コード: 指標計算済みDataFrame} の辞書。
    """
    cfg = _get_cfg()
    signals = []
    price_data_by_code = {}

    df = price_df[price_df["Code"].isin(target_codes)].copy()
    if df.empty:
        logger.warning("対象データが空です。")
        return signals, price_data_by_code

    # MA5算出に short_period 分、さらにブレイクアウト判定に
    # lookback_days + 1 日分の傾き計算が必要になる
    min_required_rows = max(cfg["MA_LONG_PERIOD"], cfg["MA_SHORT_PERIOD"])
    min_required_rows = max(
        min_required_rows,
        cfg["MA_SHORT_PERIOD"] + cfg["MA5_BREAKOUT_LOOKBACK_DAYS"] + 1,
    )
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
    最新日についてのみ「5日MAブレイクアウト」条件を判定し、該当銘柄を返す。
    main.py（日次実行、毎日16時想定）から呼ばれる。

    戻り値: dictのリスト（code, close, open, ma_short_today, ma_short_prevを含む）
    """
    cfg = _get_cfg()
    results = []

    cnt_no_data = 0
    cnt_not_bullish = 0
    cnt_not_breakout = 0
    cnt_pass = 0

    df = price_df[price_df["Code"].isin(target_codes)].copy()
    if df.empty:
        logger.warning("対象データが空です。ダウンロード結果を確認してください。")
        return results

    latest_date = df["Date"].max()
    logger.info(f"データ最新日付: {latest_date.date()}")

    min_required_rows = max(cfg["MA_LONG_PERIOD"], cfg["MA_SHORT_PERIOD"])
    min_required_rows = max(
        min_required_rows,
        cfg["MA_SHORT_PERIOD"] + cfg["MA5_BREAKOUT_LOOKBACK_DAYS"] + 1,
    )
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

        breakout = indicator.is_ma_breakout_at(
            g, idx, lookback_days=cfg["MA5_BREAKOUT_LOOKBACK_DAYS"]
        )
        if not breakout:
            cnt_not_breakout += 1
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
        f"MAブレイクアウト条件で除外: {cnt_not_breakout}件",
        f"条件通過: {cnt_pass}件",
        "=====================",
    ]
    logger.info("\n".join(diagnostic_lines))

    return results
