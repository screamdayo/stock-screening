"""
main.py
日次の全体フローをまとめるエントリーポイント。
GitHub Actionsからは `python main.py` を呼ぶだけでよい。

株価データはここで1回だけ取得し、Discord通知用のシグナル判定と
GitHub Pages用のエクスポート（docs/prices/, docs/screening.json）の
両方に使い回す。
（以前はexport_docs_prices.pyが別途、独立して全銘柄データを取得しており、
1日2回J-Quantsにリクエストしていた。取得期間は日次スクリーニング用より
チャート表示用の方が長いため、長い方（config.CHART_BUSINESS_DAYS）で
1回だけ取得すれば両方をカバーできる。）

使用する戦略は config.py の ACTIVE_STRATEGY で指定する。
過去検証（バックテスト）をしたい場合は run_backtest.py を使う。

ma5_breakoutでは通常シグナルとは別に、検証済みの限定救済候補を
参考枠として別表示する。通常シグナルの判定ロジック自体は変更しない。

エラーが発生した場合はDiscordに通知してから例外を再送出する
（GitHub Actions側でも失敗として検知できるように raise は残す）。
"""

import time

import config
import download
import notifier
import export_docs_prices
from strategies import registry
from logger import get_logger

logger = get_logger(__name__)


def run():
    logger.info("=== 株スクリーニング開始 ===")
    logger.info(f"使用する戦略: {config.ACTIVE_STRATEGY}")

    screener_fn = registry.get_latest_screener(config.ACTIVE_STRATEGY)

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

    logger.info("スクリーニング中...")
    results = screener_fn(price_df, target_codes)

    rescue_results = []
    if config.ACTIVE_STRATEGY == "ma5_breakout":
        from strategies import ma5_breakout
        rescue_results = ma5_breakout.find_latest_rescue_signals(price_df, target_codes)

    for r in results + rescue_results:
        r["name"] = code_to_name.get(r["code"], "")

    logger.info(f"通常シグナル: {len(results)}件")
    logger.info(f"救済シグナル: {len(rescue_results)}件")

    logger.info("Discord通知中...")
    notifier.notify(results, rescue_results=rescue_results)

    logger.info("GitHub Pages用データを出力中...")
    export_docs_prices.export_docs(
        price_df,
        target_codes,
        code_to_name,
        results + rescue_results,
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
