"""
設定ファイル
条件を変えたいときはここの数値だけ触ればOK。
"""

import os

# ===== 認証情報（GitHub Secretsから読む） =====
JQUANTS_API_KEY = os.environ.get("JQUANTS_API_KEY", "")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

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
ACTIVE_STRATEGY = "kuitto"


# ===== スクリーニング条件（「くいっと」戦略の定義） =====
# 他の戦略を追加した場合、それぞれの戦略ファイル内で
# 同様に「その戦略専用の設定値」をこのファイルに追記していく想定。
# 移動平均線の期間設定
MA_SHORT_PERIOD = 5     # 短期移動平均（例：5日線）
MA_LONG_PERIOD = 25     # 長期移動平均（例：25日線、使う場合）

# 陽線判定を有効にするか（終値 > 始値）
REQUIRE_BULLISH_CANDLE = True

# 短期移動平均線が「上向き」と判定する条件
# 当日の短期MA が 前日の短期MA より大きければ上向きとみなす
REQUIRE_MA_RISING = True

# RSIを使う場合の設定（現時点では未使用、拡張用）
RSI_PERIOD = 14
RSI_LOWER_BOUND = None   # 例: 30 に設定するとRSI30以上のみ対象
RSI_UPPER_BOUND = None   # 例: 70 に設定するとRSI70以下のみ対象


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
