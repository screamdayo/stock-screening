"""
設定ファイル
条件を変えたいときはここの数値だけ触ればOK。
"""

import os

# ===== 認証情報（GitHub Secretsから読む） =====
JQUANTS_API_KEY = os.environ.get("JQUANTS_API_KEY", "")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

# ===== APIリクエストの共通設定（タイムアウト・リトライ） =====
# 1リクエストあたりの最大待ち時間（秒）。
# これがないと、J-Quants側が応答しない場合に処理が永遠に止まってしまう。
API_REQUEST_TIMEOUT_SECONDS = 30

# タイムアウトや502/503/504などの一時的なエラーが起きた場合の自動リトライ回数
API_MAX_RETRIES = 3

# リトライ時の待機時間（秒）。リトライのたびに増えていく（1回目1秒、2回目2秒、3回目4秒 等）
API_RETRY_BACKOFF_SECONDS = 1

# リトライ対象とするHTTPステータスコード
# 502/503/504はサーバー側の一時的な障害でよく返ってくるコード。
# 429はレート制限超過（一定時間待てば再度通ることが多いため対象に含める）。
API_RETRYABLE_STATUS_CODES = (429, 502, 503, 504)

# 429（レート制限）の場合は、502/503/504よりも長めに待ってからリトライする方が
# 成功しやすいため、専用の待機時間を設ける。
API_RATE_LIMIT_BACKOFF_SECONDS = 5

# 銘柄ごとにAPIリクエストする際、連続で叩きすぎてレート制限に
# 引っかからないよう、1リクエストごとに最低限空ける間隔（秒）。
API_REQUEST_INTERVAL_SECONDS = 0.3

# ===== フォルダパス =====
DATA_DIR = "data"
OUTPUT_DIR = "output"
LOGS_DIR = "logs"

# ===== データ取得設定（日次スクリーニング用） =====
# 何営業日分のデータを取得するか（移動平均の計算に必要な日数+余裕分）
TARGET_BUSINESS_DAYS = 30

# 遡って探索する最大日数（休日・データ欠損時のセーフティ）
MAX_LOOKBACK_DAYS = 45

# 対象市場（J-Quantsのマスターデータの "MktNm" 列と一致させる）
TARGET_MARKET = "プライム"

# 除外する銘柄コードの末尾（優先株式など）
EXCLUDED_CODE_SUFFIXES = ("5", "6")


# ===== 使用する戦略 =====
# strategies/registry.py に登録されている戦略名を指定する。
# 新しい戦略を追加したら、ここを変えるだけで日次実行・バックテスト両方に反映される。
ACTIVE_STRATEGY = "ma5_breakout"


# ===== 戦略ごとの設定 =====
# 戦略名 -> その戦略専用の設定値の辞書。
# 新しい戦略を追加する場合は、ここに同じ形で1ブロック追記するだけでよい
# （例: STRATEGY_CONFIG["golden_cross"] = {...}）。
#
# 各戦略ファイル（strategies/*.py）は、自分の戦略名に対応する設定を
# config.STRATEGY_CONFIG["戦略名"] から参照する。
STRATEGY_CONFIG = {
    "kuitto": {
        # 移動平均線の期間設定
        "MA_SHORT_PERIOD": 5,     # 短期移動平均（例：5日線）
        "MA_LONG_PERIOD": 25,     # 長期移動平均（例：25日線、使う場合）

        # 陽線判定を有効にするか（終値 > 始値）
        "REQUIRE_BULLISH_CANDLE": True,

        # 短期移動平均線が「上向き」と判定する条件
        # 当日の短期MA が 前日の短期MA より大きければ上向きとみなす
        "REQUIRE_MA_RISING": True,

        # 短期移動平均線が「昨日まで下向き（または横ばい）→今日はっきり上向きに転換」した
        # 瞬間だけを検出する条件（REQUIRE_MA_RISINGとは別軸。両方Trueにもできるが、
        # 通常はどちらか一方をTrueにしてバックテストで比較する使い方を想定）
        "REQUIRE_MA_TURNING": False,

        # is_ma_turning_up_at() で「はっきり上向きに転じた」とみなす傾きの最小値。
        # 0だと「0より大きければOK」になり、+0.01のような微小な傾きも拾ってしまう。
        # 横ばいのノイズを除外したい場合はもう少し大きい値にする
        "MA_TURNING_MIN_SLOPE": 0.0,

        # 「くいっと」の完全な形（下落トレンド→傾きが徐々に緩やかに→底固め→反転上向き）
        # を判定する条件。REQUIRE_MA_RISING/REQUIRE_MA_TURNINGよりも厳密で、
        # 検出件数を絞り込みたい場合に使う（3つとも独立したON/OFFなので、
        # 通常はこれか他の2つのどれか1つだけをTrueにして比較する）。
        "REQUIRE_KUITTO_PATTERN": False,

        # is_kuitto_pattern_at() のパラメータ。
        # 傾きは「前日のMA値に対する変化率（%）」で判定する
        # （円ベースの絶対値だと値がさが違う銘柄ごとに閾値を変える必要が出るため、
        # %ベースにして株価水準に依存しないようにしている）。
        #
        # 判定する条件はシンプルに2つ:
        #   1. 観測期間（LOOKBACK_DAYS日）内に、はっきりした下落が1回でもあったこと
        #   2. 今日、はっきり反転上向きになること
        # （「傾きが徐々に緩やかになる」という単調性は判定しない。実際の株価は
        #   ノイズが多く、完全な単調推移になることは稀なため）
        #
        # KUITTO_PATTERN_LOOKBACK_DAYS: 下落の有無を何日分の傾きで見るか。
        #   4なら、直近5点のMA値から4つの傾きを算出して判定する。
        "KUITTO_PATTERN_LOOKBACK_DAYS": 4,

        # KUITTO_PATTERN_DOWNTREND_MIN_SLOPE_PCT: 観測期間内の傾き（%）が
        #   「明確な下落」とみなす上限（この値以下の傾きが1つでもあればOK）。
        #   -0.3なら、前日比-0.3%以下の下落が期間内に1回はあることを要求する。
        "KUITTO_PATTERN_DOWNTREND_MIN_SLOPE_PCT": -0.3,

        # KUITTO_PATTERN_TURN_MIN_SLOPE_PCT: 今日の傾き（%）が
        #   「はっきり反転上向き」とみなす下限。0.05なら前日比+0.05%を超えないと
        #   反転と認めない。
        "KUITTO_PATTERN_TURN_MIN_SLOPE_PCT": 0.05,

        # KUITTO_PATTERN_REQUIRE_FLAT: Trueにすると、反転直前（今日の一つ前）の
        #   傾きが「ほぼ水平（底固め）」であることも追加で要求する。
        #   デフォルトはFalse（水平は必須にしない、下落→反転の2条件のみで判定）。
        "KUITTO_PATTERN_REQUIRE_FLAT": False,

        # KUITTO_PATTERN_FLAT_MAX_ABS_SLOPE_PCT: REQUIRE_FLAT=True の場合に
        #   「ほぼ水平」とみなす絶対値の上限。0.1なら前日比±0.1%以内ならOK。
        "KUITTO_PATTERN_FLAT_MAX_ABS_SLOPE_PCT": 0.1,

        # RSIを使う場合の設定（現時点では未使用、拡張用）
        "RSI_PERIOD": 14,
        "RSI_LOWER_BOUND": None,   # 例: 30 に設定するとRSI30以上のみ対象
        "RSI_UPPER_BOUND": None,   # 例: 70 に設定するとRSI70以下のみ対象
    },

    # 「5日MAブレイクアウト」戦略:
    #   直近MA5_BREAKOUT_LOOKBACK_DAYS日間ずっと5日MAが下向き・水平で、
    #   当日初めて上向きに転じた ＆ 当日が陽線、を検出する。
    "ma5_breakout": {
        # 移動平均線の期間設定
        "MA_SHORT_PERIOD": 5,     # 短期移動平均（5日線）
        "MA_LONG_PERIOD": 25,     # 長期移動平均（現状は判定に未使用、拡張用に保持）

        # 陽線判定を有効にするか（終値 > 始値）
        "REQUIRE_BULLISH_CANDLE": True,

        # 「直近MA5_BREAKOUT_LOOKBACK_DAYS日間ずっと下向き・水平だったMA5が、
        #  当日初めて上向きに転じた」を判定する日数。
        #   2なら、前日・2日前の2日分の傾きがともに0以下（下向き・水平）で、
        #   当日の傾きが初めて0より大きくなった日を検出する。
        #   3にすると、前日・2日前・3日前の3日分を確認するようになり、
        #   より厳格な「初めての転換」を要求する（検出件数は減る）。
        "MA5_BREAKOUT_LOOKBACK_DAYS": 2,
    },

    # 新しい戦略を追加する場合はここに追記する。例:
    # "golden_cross": {
    #     "MA_SHORT_PERIOD": 5,
    #     "MA_LONG_PERIOD": 25,
    #     ...
    # },
}


def get_strategy_config(strategy_name=None):
    """
    指定した戦略名の設定辞書を取得するヘルパー。
    strategy_name を省略した場合は ACTIVE_STRATEGY の設定を返す。

    使い方（各戦略ファイル内）:
        import config
        cfg = config.get_strategy_config("kuitto")
        ma_short = cfg["MA_SHORT_PERIOD"]
    """
    name = strategy_name or ACTIVE_STRATEGY
    if name not in STRATEGY_CONFIG:
        available = ", ".join(STRATEGY_CONFIG.keys())
        raise ValueError(f"STRATEGY_CONFIGに未登録の戦略です: '{name}'（登録済み: {available}）")
    return STRATEGY_CONFIG[name]


# ===== バックテスト設定 =====
# バックテスト対象期間（年）
BACKTEST_YEARS = 5

# エントリー：発生日の翌営業日の始値で買う
# 保有：最大何営業日保有するか
BACKTEST_HOLD_DAYS = 10

# 利確ライン（+5% で決済）
BACKTEST_TAKE_PROFIT_PCT = 5.0

# 損切りライン（-3% で決済）
BACKTEST_STOP_LOSS_PCT = 3.0

BACKTEST_MAX_POSITIONS = 8  # 同時保有できる最大銘柄数

# ===== エクイティカーブ（資産推移）計算用の設定 =====
# 資産推移は「決済順に1銘柄ずつ取引した場合」を想定した簡略化モデルで計算する。
# 実際には複数銘柄を同時保有できるため、あくまで目安の指標として扱うこと。

# 初期資金（円）
BACKTEST_INITIAL_CAPITAL = 1_000_000

# 1トレードにつき資産の何%を投じるか（100なら全額を毎回投じる想定）
BACKTEST_POSITION_SIZE_PCT = 100


# ===== Discord通知設定 =====
MAX_RESULTS_TO_SHOW = 30          # 通知に表示する最大銘柄数
DISCORD_MESSAGE_CHUNK_SIZE = 1900  # Discordの1メッセージあたり文字数上限対策

KABUTAN_CHART_URL_TEMPLATE = "https://kabutan.jp/stock/chart?code={code}"
