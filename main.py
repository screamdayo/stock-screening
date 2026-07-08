"""
main.py
日次スクリーニングの全体フローをまとめるエントリーポイント。
GitHub Actionsからは `python main.py` を呼ぶだけでよい。

使用する戦略は config.py の ACTIVE_STRATEGY で指定する。
過去検証（バックテスト）をしたい場合は run_backtest.py を使う。

エラーが発生した場合はDiscordに通知してから例外を再送出する
（GitHub Actions側でも失敗として検知できるように raise は残す）。
"""

import time

import config
import download
import notifier
from strategies import registry
from logger import get_logger

logger = get_logger(__name__)


def run():
    logger.info("=== 株スクリーニング開始 ===")
    logger.info(f"使用する戦略: {config.ACTIVE_STRATEGY}")

    screener_fn = registry.get_latest_screener(config.ACTIVE_STRATEGY)

    logger.info("対象銘柄リスト取得中...")
    target_codes, code_to_name = download.get_target_codes_and_names()

    logger.info("株価データ取得中...")
    price_df = download.get_price_history()

    logger.info("スクリーニング中...")
    results = screener_fn(price_df, target_codes)

    # 会社名を各結果に付与する（見つからない場合は空文字のままにしておく）
    for r in results:
        r["name"] = code_to_name.get(r["code"], "")

    logger.info(f"該当: {len(results)}件")

    logger.info("Discord通知中...")
    notifier.notify(results)

    logger.info("=== 完了 ===")


def main():
    start = time.time()
    try:
        run()
    except Exception as e:
        logger.error(f"エラー: {e}", exc_info=True)
        notifier.notify_error(e, context="main.py（日次スクリーニング）実行中")
        raise
    finally:
        logger.info(f"実行時間: {time.time() - start:.1f}秒")


if __name__ == "__main__":
    main()
