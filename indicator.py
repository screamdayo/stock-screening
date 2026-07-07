"""
indicator.py
株価データから各種テクニカル指標を計算する。
「計算すること」だけに専念し、条件判定はscreen.py側で行う。
"""

import pandas as pd
import config


def add_moving_averages(df_one_code):
    """
    1銘柄分の時系列データ（日付昇順）に、短期・長期移動平均列を追加する。
    df_one_code: Code, Date, O, C などの列を持つDataFrame（1銘柄分、日付昇順）
    """
    df = df_one_code.copy()
    df["MA_SHORT"] = df["C"].rolling(window=config.MA_SHORT_PERIOD).mean()
    df["MA_LONG"] = df["C"].rolling(window=config.MA_LONG_PERIOD).mean()
    return df


def add_rsi(df_one_code, period=None):
    """RSI（相対力指数）を計算して列を追加する（拡張用、現状は任意）。"""
    if period is None:
        period = config.RSI_PERIOD

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
