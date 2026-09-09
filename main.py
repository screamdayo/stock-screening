"""
main.py
日次の全体フローをまとめるエントリーポイント。
GitHub Actionsからは `python main.py` を呼ぶだけでよい。

2026-09-09以降の日次運用は、目視A/B/skipを廃止し、
検証済みの自動条件 `kuitto_pullback_auto` を使用する。
旧ma5_breakout/A-B/救済ロジックは比較検証用にコードを残すが、
日次本番では起動しない。

エラーが発生した場合はDiscordに通知してから例外を再送出する。
"""

import time

import config
import download
import notifier
import export_docs_prices
from strategies import registry
from logger import get_logger

logger = get_logger(__name__)

# 目視撤廃後の本番日次戦略。
DAILY_STRATEGY = "kuitto_pullback_auto"


def run():
    logger.info("=== 株スクリーニング開始 ===")
    logger.info(f"使用する日次戦略: {DAILY_STRATEGY}（目視判定なし）")

    screener_fn = registry.get_latest_screener(DAILY_STRATEGY)

    logger.info("対象銘柄リスト取得中...")
    target_codes, code_to_name = download.get_target_codes_and_names()

    logger.info(
        f"株価データ取得中（直近{config.CHART_BUSINESS_DAYS}営業日、全銘柄）..."
    )
    price_df = download.get_price_history(
        target_business_days=config.CHART_BUSINESS_DAYS,
        max_lookback_days=config.CHART_MAX_LOOKBACK_DAYS,
    )
    price_df = price_df[price_df["Code"].isin(target_codes)]

    logger.info("自動スクリーニング中...")
    results = screener_fn(price_df, target_codes)

    for r in results:
        r["name"] = code_to_name.get(r["code"], "")

    logger.info(f"自動通過候補: {len(results)}件")

    logger.info("Discord通知中...")
    notifier.notify(results)

    logger.info("GitHub Pages用データを出力中...")
    export_docs_prices.export_docs(
        price_df,
        target_codes,
        code_to_name,
        results,
    )

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
