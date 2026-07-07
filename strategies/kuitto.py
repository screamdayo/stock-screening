"""
strategies/kuitto.py

「くいっと」戦略: 5日移動平均線が上向き ＆ 陽線 の銘柄を検出する。
条件の中身（何日線か、陽線を必須にするかなど）はconfig.pyで調整する。

この戦略単体で完結しており、backtest.py はこのファイルの中身を知らない。
新しい戦略を追加したいときは、このファイルをコピーして
find_signals() の中身だけを書き換えればよい。
"""

import config
import indicator


def _passes_conditions(g, idx):
    """
    1銘柄の時系列DataFrame g の idx番目の行が、
    「くいっと」の条件（陽線・MA上向き・MA反転上向き）を満たすかどうかを判定する。
    """
    latest = g.iloc[idx]

    if config.REQUIRE_BULLISH_CANDLE and not indicator.is_bullish_candle(latest):
        return False

    if config.REQUIRE_MA_RISING and not indicator.is_ma_rising_at(g, idx):
        return False

    if config.REQUIRE_MA_TURNING and not indicator.is_ma_turning_up_at(g, idx):
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
    signals = []
    price_data_by_code = {}

    df = price_df[price_df["Code"].isin(target_codes)].copy()
    if df.empty:
        print("[kuitto] 対象データが空です。")
        return signals, price_data_by_code

    min_required_rows = max(config.MA_LONG_PERIOD, config.MA_SHORT_PERIOD) + 1
    code_count = 0

    for code, group in df.groupby("Code"):
        g = group.dropna(subset=["C", "O"]).sort_values("Date").reset_index(drop=True)

        if len(g) < min_required_rows:
            continue

        g = indicator.add_moving_averages(g)
        code_count += 1

        # この銘柄のDataFrameは1つだけ辞書に保持する
        price_data_by_code[code] = g

        for idx in range(min_required_rows, len(g)):
            if _passes_conditions(g, idx):
                signals.append({
                    "code": code,
                    "signal_date": g["Date"].iloc[idx],
                    "signal_idx": idx,
                })

    print(f"[kuitto] バックテスト対象銘柄数: {code_count}件")
    print(f"[kuitto] 検出したシグナル数: {len(signals)}件")

    return signals, price_data_by_code


def find_latest_signals(price_df, target_codes):
    """
    【日次スクリーニング用】
    最新日についてのみ「くいっと」条件を判定し、該当銘柄を返す。
    main.py（日次実行）から呼ばれる。

    戻り値: dictのリスト（code, close, open, ma_short_today, ma_short_prevを含む）
    """
    results = []

    cnt_no_data = 0
    cnt_not_bullish = 0
    cnt_ma_not_rising = 0
    cnt_ma_not_turning = 0
    cnt_pass = 0

    df = price_df[price_df["Code"].isin(target_codes)].copy()
    if df.empty:
        print("[kuitto] 対象データが空です。ダウンロード結果を確認してください。")
        return results

    latest_date = df["Date"].max()
    print(f"[kuitto] データ最新日付: {latest_date.date()}")

    min_required_rows = max(config.MA_LONG_PERIOD, config.MA_SHORT_PERIOD) + 1

    for code, group in df.groupby("Code"):
        g = group.dropna(subset=["C", "O"]).sort_values("Date").reset_index(drop=True)

        if len(g) < min_required_rows:
            cnt_no_data += 1
            continue

        g = indicator.add_moving_averages(g)
        latest = g.iloc[-1]
        idx = len(g) - 1

        bullish = indicator.is_bullish_candle(latest)
        if config.REQUIRE_BULLISH_CANDLE and not bullish:
            cnt_not_bullish += 1
            continue

        ma_rising = indicator.is_ma_rising_at(g, idx)
        if config.REQUIRE_MA_RISING and not ma_rising:
            cnt_ma_not_rising += 1
            continue

        ma_turning = indicator.is_ma_turning_up_at(g, idx)
        if config.REQUIRE_MA_TURNING and not ma_turning:
            cnt_ma_not_turning += 1
            continue

        cnt_pass += 1
        results.append({
            "code": code,
            "close": round(latest["C"], 1),
            "open": round(latest["O"], 1),
            "ma_short_today": round(g["MA_SHORT"].iloc[-1], 1),
            "ma_short_prev": round(g["MA_SHORT"].iloc[-2], 1),
        })

    print("[kuitto] ===== 診断結果 =====")
    print(f"[kuitto] データ不足で除外: {cnt_no_data}件")
    print(f"[kuitto] 陽線条件で除外: {cnt_not_bullish}件")
    print(f"[kuitto] MA上向き条件で除外: {cnt_ma_not_rising}件")
    print(f"[kuitto] MA反転条件で除外: {cnt_ma_not_turning}件")
    print(f"[kuitto] 両条件通過: {cnt_pass}件")
    print("[kuitto] =====================")

    return results
