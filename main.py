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


def run():
    print("=== 株スクリーニング開始 ===")
    print(f"[main] 使用する戦略: {config.ACTIVE_STRATEGY}")

    screener_fn = registry.get_latest_screener(config.ACTIVE_STRATEGY)

    print("[main] 対象銘柄リスト取得中...")
    target_codes = download.get_target_codes()

    print("[main] 株価データ取得中...")
    price_df = download.get_price_history()

    print("[main] スクリーニング中...")
    results = screener_fn(price_df, target_codes)
    print(f"[main] 該当: {len(results)}件")

    print("[main] Discord通知中...")
    notifier.notify(results)

    print("=== 完了 ===")


def main():
    start = time.time()
    try:
        run()
    except Exception as e:
        print(f"エラー: {e}")
        notifier.notify_error(e, context="main.py（日次スクリーニング）実行中")
        raise
    finally:
        print(f"実行時間: {time.time() - start:.1f}秒")


if __name__ == "__main__":
    main()
