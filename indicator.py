"""
indicator.py
株価データから各種テクニカル指標を計算する。
「計算すること」だけに専念し、条件判定は各戦略ファイル（strategies/*.py）側で行う。

このファイルは特定の戦略のconfigを知らない（汎用モジュールとして設計）。
期間やパラメータは呼び出し側（戦略ファイル）が config.get_strategy_config() から
取り出して引数で渡す。
"""

import pandas as pd


def add_moving_averages(df_one_code, short_period, long_period):
    """
    1銘柄分の時系列データ（日付昇順）に、短期・長期移動平均列を追加する。

    df_one_code: Code, Date, O, C などの列を持つDataFrame（1銘柄分、日付昇順）
    short_period: 短期移動平均の期間（例: 5）
    long_period: 長期移動平均の期間（例: 25）
    """
    df = df_one_code.copy()
    df["MA_SHORT"] = df["C"].rolling(window=short_period).mean()
    df["MA_LONG"] = df["C"].rolling(window=long_period).mean()
    return df


def add_rsi(df_one_code, period=14):
    """RSI（相対力指数）を計算して列を追加する（拡張用、現状は任意）。"""
    df = df_one_code.copy()
    delta = df["C"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def is_bullish_candle(row):
    """陽線判定: 終値 > 始値"""
    return row["C"] > row["O"]


def is_ma_rising_at(df_with_ma, idx):
    """
    指定インデックス idx の時点で、短期移動平均線が「上向き」かどうかを判定する。
    df_with_ma: add_moving_averages() 済みのDataFrame（日付昇順、reset_index済み推奨）
    idx: 判定したい行のインデックス（0始まり）
    """
    if idx < 1 or idx >= len(df_with_ma):
        return False

    today_ma = df_with_ma["MA_SHORT"].iloc[idx]
    prev_ma = df_with_ma["MA_SHORT"].iloc[idx - 1]

    if pd.isna(today_ma) or pd.isna(prev_ma):
        return False

    return today_ma > prev_ma


def is_ma_rising(df_with_ma):
    """直近日（最終行）について is_ma_rising_at を呼ぶショートカット。"""
    return is_ma_rising_at(df_with_ma, len(df_with_ma) - 1)


def is_ma_turning_up_at(df_with_ma, idx, min_slope=0.0):
    """
    指定インデックス idx の時点で、短期移動平均線が
    「昨日まで下向き（または横ばい） → 今日はっきり上向きに転換した」かどうかを判定する。

    is_ma_rising_at() が「今日が前日より高いか」だけを見るのに対し、
    こちらは「傾きの向きが変わった瞬間（反転）」を捉える。

    判定の流れ（3日前・2日前・昨日・今日の4点、傾きは2つ使う）:
        slope_yesterday  = MA[idx-1] - MA[idx-2]
        slope_today      = MA[idx]   - MA[idx-1]

    条件:
        1. 昨日の傾き（slope_yesterday）が 0 以下 = 下向き or 横ばいだった
        2. 今日の傾き（slope_today）が min_slope より大きい
           = はっきりと上向きに転じた（小さすぎる傾きはノイズとして除外）

    min_slope: 「はっきり上向き」とみなす傾きの最小値。
        0だと「0より大きければOK」になり、+0.01のような微小な傾きも拾ってしまう。
        横ばいのノイズを除外したい場合はもう少し大きい値
        （例: 0.05や0.1など、株価水準に応じて調整）を渡すとよい。
    """
    if idx < 2 or idx >= len(df_with_ma):
        return False

    ma = df_with_ma["MA_SHORT"]

    today_ma = ma.iloc[idx]
    yesterday_ma = ma.iloc[idx - 1]
    two_days_ago_ma = ma.iloc[idx - 2]

    if pd.isna(today_ma) or pd.isna(yesterday_ma) or pd.isna(two_days_ago_ma):
        return False

    slope_yesterday = yesterday_ma - two_days_ago_ma
    slope_today = today_ma - yesterday_ma

    was_falling_or_flat = slope_yesterday <= 0
    is_now_clearly_rising = slope_today > min_slope

    return was_falling_or_flat and is_now_clearly_rising
