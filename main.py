"""
main.py
日次の全体フローをまとめるエントリーポイント。
GitHub Actionsからは `python main.py` を呼ぶだけでよい。

2026-09-09以降の日次運用は、目視A/B/skipを廃止し、
検証済みの自動条件 `kuitto_pullback_auto` を使用する。
旧ma5_breakout/A-B/救済ロジックは比較検証用にコードを残すが、
日次本番では起動しない。

保有銘柄が holdings.json に登録されている場合は、同じ日足データで
厳しめがくっと/最大15営業日を監視し、売りルール成立時だけDiscord通知する。

2026-09-09以降の新規シグナルは forward_test.py で将来検証用に記録し、
翌朝ギャップ・5/10/15日後騰落・MFE/MAE・出口結果を日次で追記する。
同じJ-Quants日足からプライム市場の値上がり/値下がり比率も
market_breadth.py で日次記録する。

エラーが発生した場合はDiscordに通知してから例外を再送出する。
"""

import time

import config
import download
import notifier
import export_docs_prices
import sell_monitor
import forward_test
import market_breadth
import earnings_warning
import pinch_to_chance
from strategies import registry
from logger import get_logger

logger = get_logger(__name__)

# 目視撤廃後の本番日次戦略。
DAILY_STRATEGY = "kuitto_pullback_auto"
GC_DAILY_STRATEGY = "gc_strong_breakout"


def run():
    logger.info("=== 株スクリーニング開始 ===")
    logger.info(f"使用する日次戦略: {DAILY_STRATEGY}（目視判定なし）")

    screener_fn = registry.get_latest_screener(DAILY_STRATEGY)
    gc_screener_fn = registry.get_latest_screener(GC_DAILY_STRATEGY)

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

    logger.info("保有銘柄の売りシグナル確認中...")
    sell_alerts = sell_monitor.check_sell_signals(price_df, code_to_name)
    if sell_alerts:
        logger.info(f"売りルール成立: {len(sell_alerts)}件")
        notifier.notify_sell_signals(sell_alerts)
    else:
        logger.info("売りルール成立なし")

    logger.info("自動スクリーニング中...")
    results = screener_fn(price_df, target_codes)

    logger.info("GC強ブレイクを別枠スクリーニング中...")
    gc_results = gc_screener_fn(price_df, target_codes)

    for r in results:
        r["name"] = code_to_name.get(r["code"], "")
    for r in gc_results:
        r["name"] = code_to_name.get(r["code"], "")

    latest_signal_date = price_df["Date"].max()
    logger.info("決算予定チェック中...")
    earnings_warning.add_earnings_warnings(results, latest_signal_date)
    earnings_warning.add_earnings_warnings(gc_results, latest_signal_date)

    logger.info(f"自動通過候補: {len(results)}件")
    logger.info(f"GC強ブレイク候補: {len(gc_results)}件")

    logger.info("未来検証ログ更新中...")
    forward_test.update_forward_test(price_df, code_to_name)

    logger.info("市場地合いログ更新中...")
    market_breadth.update_market_breadth(price_df)

    logger.info("ピンチをチャンスにセンサー判定中...")
    pinch_sensor = pinch_to_chance.evaluate(price_df, target_codes)
    for r in pinch_sensor.get("candidates", []):
        r["name"] = code_to_name.get(r["code"], "")
    earnings_warning.add_earnings_warnings(
        pinch_sensor.get("candidates", []),
        latest_signal_date,
    )

    logger.info("Discord通知中...")
    notifier.notify(results)
    notifier.notify_gc_strong_breakout(gc_results)
    notifier.notify_pinch_to_chance(pinch_sensor)

    logger.info("GitHub Pages用データを出力中...")
    export_docs_prices.export_docs(
        price_df,
        target_codes,
        code_to_name,
        results,
        gc_results,
        pinch_sensor,
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
