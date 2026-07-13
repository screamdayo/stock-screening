"""
logger.py
プロジェクト全体で共通のロギング設定。

GitHub Actionsのログ画面とローカルのターミナル両方で見やすいよう、
- コンソール（標準出力）: GitHub Actionsのログとしてそのまま表示される
- ファイル（logs/以下）: ローカルで後から見返せるように保存する
の両方に出力する。

使い方（各ファイルの先頭で）:
    from logger import get_logger
    logger = get_logger(__name__)

    logger.info("株価データ取得中...")
    logger.warning("データが空です")
    logger.error(f"APIエラー: {e}")
"""

import os
import logging
from datetime import datetime

import config

_LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure_root_logger():
    """
    ルートロガーを一度だけ設定する。
    複数ファイルから get_logger() が呼ばれても、ハンドラが重複登録されないようにする。
    """
    global _configured
    if _configured:
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # ---- コンソール出力（GitHub Actionsのログ・ローカルのターミナル両方に出る） ----
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # ---- ファイル出力（logs/以下、ローカルで後から見返す用） ----
    try:
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        today_str = datetime.now().strftime("%Y%m%d")
        log_path = os.path.join(config.LOGS_DIR, f"{today_str}.log")

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    except OSError as e:
        # ファイル出力に失敗しても（読み取り専用の環境など）致命的にはしない。
        # コンソール出力さえ生きていればGitHub Actions上は困らないため。
        console_handler.stream.write(
            f"[logger] ログファイルの作成に失敗しました（コンソール出力のみ継続）: {e}\n"
        )

    _configured = True


def get_logger(name):
    """
    モジュールごとのロガーを取得する。
    各ファイルの先頭で `logger = get_logger(__name__)` のように呼び出す。
    """
    _configure_root_logger()
    return logging.getLogger(name)
