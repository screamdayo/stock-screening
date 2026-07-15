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


def is_ma_breakout_at(df_with_ma, idx, lookback_days=2):
    """
    指定インデックス idx の時点で、短期移動平均線が
    「直近 lookback_days 日間ずっと下向き（または横ばい） →
      今日初めて上向きに転じた」かどうかを判定する。

    is_ma_turning_up_at() が「昨日1日分の傾きが下向き・横ばいだったか」しか
    見ないのに対し、こちらは過去 lookback_days 日分の傾きすべてが
    下向き・横ばいだったことを要求する（より厳格な「初めての転換」判定）。

    判定の流れ（lookback_days=2の場合、3日前・2日前・昨日・今日の4点、傾きは3つ使う）:
        slope[-2] = MA[idx-2] - MA[idx-3]   （観測期間1日目の傾き）
        slope[-1] = MA[idx-1] - MA[idx-2]   （観測期間2日目=昨日の傾き）
        slope[0]  = MA[idx]   - MA[idx-1]   （今日の傾き）

    条件:
        1. 観測期間（lookback_days日）の傾きが、すべて0以下（下向き or 横ばい）だった
        2. 今日の傾きが0より大きい（はっきり上向きに転じた）

    df_with_ma: add_moving_averages() 済みのDataFrame（日付昇順、reset_index済み推奨）
    idx: 判定したい行のインデックス（0始まり、「今日」に相当する）
    lookback_days: 「下向き・横ばいが続いていたか」を何日分さかのぼって確認するか
        （config.pyで調整可能にする想定。例えば2なら、前日・2日前の2日分を確認する）

    戻り値: 条件を満たせばTrue
    """
    # 判定に必要な行数: 今日(idx) + 観測期間(lookback_days) + 傾き算出用に1日分余分に必要
    required_start_idx = idx - lookback_days - 1
    if required_start_idx < 0 or idx >= len(df_with_ma):
        return False

    ma = df_with_ma["MA_SHORT"]
    ma_window = ma.iloc[required_start_idx: idx + 1]

    if ma_window.isna().any():
        return False

    ma_values = ma_window.tolist()
    # slopes[i] = ma_values[i+1] - ma_values[i]。末尾が「今日」の傾きになる。
    slopes = [ma_values[i] - ma_values[i - 1] for i in range(1, len(ma_values))]

    today_slope = slopes[-1]
    past_slopes = slopes[:-1]  # 直近lookback_days日分の傾き

    if len(past_slopes) < lookback_days:
        return False

    was_falling_or_flat_all_days = all(s <= 0 for s in past_slopes)
    is_now_rising = today_slope > 0

    return was_falling_or_flat_all_days and is_now_rising


def is_strong_bullish_candle(row, min_gain_pct=1.5):
    """
    当日の陽線の「強さ」を判定する: 終値が始値から min_gain_pct(%) 以上
    上昇しているかどうか。

    is_bullish_candle() が「終値 > 始値」という陽線の有無だけを見るのに対し、
    こちらは上昇の大きさ（勢い）を要求する、より厳格な条件。

    row: O（始値）, C（終値）を持つ1行のデータ（Series）
    min_gain_pct: 「強い陽線」とみなす上昇率の下限（%）。デフォルト1.5。

    戻り値: (C / O - 1) * 100 >= min_gain_pct であればTrue
    """
    open_price = row["O"]
    close_price = row["C"]

    if pd.isna(open_price) or pd.isna(close_price) or open_price <= 0:
        return False

    gain_pct = (close_price / open_price - 1) * 100

    return gain_pct >= min_gain_pct


def is_ma_decline_before_turn_at(df_with_ma, idx, decline_lookback_days=5, decline_max_pct=-0.5):
    """
    指定インデックス idx（シグナル当日）の「前日」を基準に、
    直近 decline_lookback_days 日間で5日MAが decline_max_pct(%) 以上
    下落していたかどうかを判定する。

    「今日初めて上向きに転じた」瞬間を検出する前段階として、
    転換前にちゃんと下落の実体があったかを見るための条件。
    （転換直前の判定に today=idx を使わないのは、当日はすでに
     上向きに転じているため、下落の有無を測るには前日を基準にする必要があるため）

    判定式:
        ma_decline_pct = (MA[idx-1] / MA[idx-1-decline_lookback_days] - 1) * 100
        ma_decline_pct <= decline_max_pct であればTrue

    df_with_ma: add_moving_averages() 済みのDataFrame（日付昇順、reset_index済み推奨）
    idx: 判定したい行のインデックス（0始まり、「今日」に相当する）
    decline_lookback_days: 下落率を測る期間（日数）。デフォルト5。
    decline_max_pct: 「下落した」とみなす変化率の上限（この値以下ならOK）。
        デフォルト-0.5なら、直近5日間で5日MAが0.5%以上下落していたことを要求する。

    戻り値: 条件を満たせばTrue
    """
    base_idx = idx - 1
    past_idx = base_idx - decline_lookback_days

    if past_idx < 0 or base_idx >= len(df_with_ma):
        return False

    ma = df_with_ma["MA_SHORT"]
    base_ma = ma.iloc[base_idx]
    past_ma = ma.iloc[past_idx]

    if pd.isna(base_ma) or pd.isna(past_ma) or past_ma == 0:
        return False

    ma_decline_pct = (base_ma / past_ma - 1) * 100

    return ma_decline_pct <= decline_max_pct


def is_ma_short_below_long_at(df_with_ma, idx):
    """
    指定インデックス idx の時点で、短期移動平均線（MA_SHORT）が
    長期移動平均線（MA_LONG）以下にあるかどうかを判定する
    （まだ大きな上昇トレンドに転換していない、下位にいる状態に絞るための位置条件）。

    df_with_ma: add_moving_averages() 済みのDataFrame（日付昇順、reset_index済み推奨）
    idx: 判定したい行のインデックス（0始まり）

    戻り値: MA_SHORT <= MA_LONG であればTrue
    """
    if idx < 0 or idx >= len(df_with_ma):
        return False

    ma_short = df_with_ma["MA_SHORT"].iloc[idx]
    ma_long = df_with_ma["MA_LONG"].iloc[idx]

    if pd.isna(ma_short) or pd.isna(ma_long):
        return False

    return ma_short <= ma_long


def is_kuitto_pattern_at(
    df_with_ma,
    idx,
    lookback_days=4,
    downtrend_min_slope_pct=-0.3,
    turn_min_slope_pct=0.05,
    require_flat=False,
    flat_max_abs_slope_pct=0.1,
):
    """
    「くいっと」のシンプルな形（下落があった後の反転）を判定する。

    is_ma_turning_up_at() が「前日→今日で傾きがプラスに転じたか」だけを見るのに対し、
    こちらは「直近のlookback_days日以内に、はっきりした下落があったかどうか」も
    合わせて確認する。これにより、ずっと上昇が続いている銘柄がたまたま1日
    傾きを緩めてまた上昇したようなケース（下落トレンドを経ていない）を除外できる。

    判定する条件（デフォルトはシンプルな2条件のみ）:
        1. 観測期間（lookback_days日）の中に、はっきりした下落（マイナスの傾き）が
           少なくとも1つあったこと
        2. 今日、はっきり上向きに転換すること（今日の傾きが閾値を超えてプラス）

        require_flat=True にした場合はオプションで以下も追加される:
        3. 今日の一つ前（反転直前）の傾きがほぼ水平であること（底固め）

    「傾きが徐々に緩やかになっていく」という単調性は判定しない
    （実際の株価はノイズが多く、完全な単調推移になることは稀で、
    条件を満たす銘柄がほぼ無くなってしまうため）。

    傾きは「MAの値そのものの差分（円）」ではなく、「前日のMA値に対する変化率（%）」で
    判定する。株価が100円の銘柄と10000円の銘柄で同じ閾値を使い回せるようにするため。

    df_with_ma: add_moving_averages() 済みのDataFrame（日付昇順、reset_index済み推奨）
    idx: 判定したい行のインデックス（0始まり、「今日」に相当する）
    lookback_days: 下落の有無を何日分の傾きで見るか（デフォルト4 = 傾き4つ分 = MAの値5点分）
    downtrend_min_slope_pct: 観測期間内の傾きが「明確な下落」とみなす上限
        （この値以下の傾きが1つでもあればOK。例えば-0.3なら前日比-0.3%以下）
    turn_min_slope_pct: 今日の傾き（%）が「はっきり反転上向き」とみなす下限
        （例えば0.05なら、前日比+0.05%を超えないと反転と認めない）
    require_flat: Trueにすると、反転直前の傾きが「ほぼ水平」であることも追加で要求する
    flat_max_abs_slope_pct: require_flat=True の場合に「ほぼ水平」とみなす絶対値の上限

    戻り値: 条件を満たせばTrue
    """
    start_idx = idx - lookback_days
    if start_idx < 0 or idx >= len(df_with_ma):
        return False

    ma = df_with_ma["MA_SHORT"]
    ma_window = ma.iloc[start_idx: idx + 1]

    if ma_window.isna().any() or (ma_window == 0).any():
        return False

    # 傾きを「前日のMA値に対する変化率（%）」として算出する
    ma_values = ma_window.tolist()
    slopes_pct = [
        (ma_values[i] - ma_values[i - 1]) / ma_values[i - 1] * 100
        for i in range(1, len(ma_values))
    ]
    # slopes_pct[-1] が「今日」の傾き、それ以外（slopes_pct[:-1]）が観測期間の過去分

    if len(slopes_pct) < 2:
        return False

    past_slopes = slopes_pct[:-1]
    today_slope = slopes_pct[-1]

    # ---- 条件1: 観測期間内に、はっきりした下落が少なくとも1つあったこと ----
    had_downtrend = any(s <= downtrend_min_slope_pct for s in past_slopes)
    if not had_downtrend:
        return False

    # ---- 条件3（オプション）: 反転直前の傾きがほぼ水平であること ----
    if require_flat:
        flat_slope = slopes_pct[-2]
        if abs(flat_slope) > flat_max_abs_slope_pct:
            return False

    # ---- 条件2: 今日、はっきり反転上向きであること ----
    if today_slope <= turn_min_slope_pct:
        return False

    return True
