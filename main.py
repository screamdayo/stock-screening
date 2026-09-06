"""
main.py
日次の全体フローをまとめるエントリーポイント。
GitHub Actionsからは `python main.py` を呼ぶだけでよい。

株価データはここで1回だけ取得し、Discord通知用のシグナル判定と
GitHub Pages用のエクスポート（docs/prices/, docs/screening.json）の
両方に使い回す。

ma5_breakoutでは、厳格な通常シグナルからバックテストで固定した
A/B正規化混合の本命候補を作る。通常シグナルと限定救済候補も残し、
本命候補を先頭に表示する。

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
    primary_results = []
    primary_meta = None

    if config.ACTIVE_STRATEGY == "ma5_breakout":
        from strategies import ma5_breakout
        from production_screening import build_production_shortlist

        rescue_results = ma5_breakout.find_latest_rescue_signals(price_df, target_codes)
        primary_results, primary_meta = build_production_shortlist(price_df, results)

    for r in results + rescue_results + primary_results:
        r["name"] = code_to_name.get(r["code"], "")

    primary_codes = {str(r["code"]) for r in primary_results}
    normal_results = []
    for r in results:
        if str(r["code"]) in primary_codes:
            continue
        x = dict(r)
        x["screening_bucket"] = "normal"
        normal_results.append(x)

    for r in rescue_results:
        r["screening_bucket"] = "rescue"

    logger.info(f"本命候補: {len(primary_results)}件")
    if primary_meta:
        logger.info(
            "A候補 %s→%s件 / B候補 %s→%s件 / 正規化混合 %s件",
            primary_meta["a_before_cut"], primary_meta["a_after_cut"],
            primary_meta["b_before_cut"], primary_meta["b_after_cut"],
            primary_meta["primary_count"],
        )
    logger.info(f"通常シグナル（本命除く）: {len(normal_results)}件")
    logger.info(f"救済シグナル: {len(rescue_results)}件")

    logger.info("Discord通知中...")
    notifier.notify(
        normal_results,
        rescue_results=rescue_results,
        primary_results=primary_results,
    )

    logger.info("GitHub Pages用データを出力中...")
    export_docs_prices.export_docs(
        price_df,
        target_codes,
        code_to_name,
        primary_results + normal_results + rescue_results,
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
