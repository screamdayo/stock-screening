"""
strategies/ma5_breakout.py

「5日MAブレイクアウト」戦略:
    ① 過去5日間で5日移動平均線が MA5_DECLINE_MAX_PCT(%) 以上下落していた
       （下落の実体があったことを要求する）
    ② シグナル当日、5日移動平均線が25日移動平均線以下にある
       （まだ大きな上昇トレンドに転換していない、下位にいる状態に絞る）
    ③ シグナル当日、5日移動平均線が初めて上向きに転じた
    ④ シグナル当日が陽線（終値 > 始値）
の全てを満たす銘柄を検出する。

条件の中身（下落率の閾値など）は config.STRATEGY_CONFIG["ma5_breakout"] で調整する。

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
    「5日MAブレイクアウト」の条件（下落実体・位置・転換・陽線）を満たすかどうかを判定する。
    """
    latest = g.iloc[idx]

    if cfg["REQUIRE_BULLISH_CANDLE"] and not indicator.is_bullish_candle(latest):
        return False

    # ① 過去MA5_DECLINE_LOOKBACK_DAYS日間で5日MAが十分下落していたか（転換前日を基準に判定）
    if not indicator.is_ma_decline_before_turn_at(
        g, idx,
        decline_lookback_days=cfg["MA5_DECLINE_LOOKBACK_DAYS"],
        decline_max_pct=cfg["MA5_DECLINE_MAX_PCT"],
    ):
        return False

    # ② シグナル当日、5日MAが25日MA以下にあるか（位置条件）
    if not indicator.is_ma_short_below_long_at(g, idx):
        return False

    # ③ シグナル当日、5日MAが初めて上向きに転じたか
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

    # 必要行数: 25日MA算出分、下落率判定（前日基準でさらにdecline_lookback_days日分）、
    # ブレイクアウト判定（lookback_days+1日分の傾き計算）のうち最大のものを満たす必要がある
    min_required_rows = max(cfg["MA_LONG_PERIOD"], cfg["MA_SHORT_PERIOD"])
    min_required_rows = max(
        min_required_rows,
        cfg["MA_SHORT_PERIOD"] + cfg["MA5_DECLINE_LOOKBACK_DAYS"] + 1,
    )
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

    戻り値: dictのリスト（code, close, open, ma_short_today, ma_short_prev, ma_long_todayを含む）
    """
    cfg = _get_cfg()
    results = []

    cnt_no_data = 0
    cnt_not_bullish = 0
    cnt_not_decline = 0
    cnt_not_below_long = 0
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
        cfg["MA_SHORT_PERIOD"] + cfg["MA5_DECLINE_LOOKBACK_DAYS"] + 1,
    )
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

        declined = indicator.is_ma_decline_before_turn_at(
            g, idx,
            decline_lookback_days=cfg["MA5_DECLINE_LOOKBACK_DAYS"],
            decline_max_pct=cfg["MA5_DECLINE_MAX_PCT"],
        )
        if not declined:
            cnt_not_decline += 1
            continue

        below_long = indicator.is_ma_short_below_long_at(g, idx)
        if not below_long:
            cnt_not_below_long += 1
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
            "ma_long_today": round(g["MA_LONG"].iloc[-1], 1),
        })

    diagnostic_lines = [
        "===== 診断結果 =====",
        f"データ不足で除外: {cnt_no_data}件",
        f"陽線条件で除外: {cnt_not_bullish}件",
        f"下落実体条件で除外: {cnt_not_decline}件",
        f"位置条件（5日MA<=25日MA）で除外: {cnt_not_below_long}件",
        f"MAブレイクアウト条件で除外: {cnt_not_breakout}件",
        f"条件通過: {cnt_pass}件",
        "=====================",
    ]
    logger.info("\n".join(diagnostic_lines))

    return results
