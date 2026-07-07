"""
run_backtest.py
過去N年分のデータで、指定した戦略のバックテストを実行するエントリーポイント。

使い方:
    python run_backtest.py                  # デフォルト戦略（kuitto）で実行
    python run_backtest.py --strategy kuitto
    python run_backtest.py --strategy golden_cross   # 新しい戦略を追加した場合

新しい戦略を追加する手順:
    1. strategies/ 以下に新しいファイルを作り、find_signals(price_df, target_codes) を実装する
       （strategies/kuitto.py をコピーして中身を書き換えるのが早い）
    2. strategies/registry.py の STRATEGIES 辞書に1行追加する
    3. python run_backtest.py --strategy <新しい戦略名> で実行できる

    backtest.py 自体は変更不要（戦略の中身を知らない設計になっているため）。

初回実行時はJ-Quantsからデータを取得してdata/以下にCSVキャッシュを作る。
2回目以降はキャッシュを使うため高速に再実行できる。
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

import pandas as pd

import config
import download
import backtest
import notifier
from strategies import registry

DEFAULT_STRATEGY = "kuitto"


def parse_args():
    parser = argparse.ArgumentParser(description="株価バックテストツール")
    parser.add_argument(
        "--strategy",
        default=DEFAULT_STRATEGY,
        help=f"使用する戦略名（デフォルト: {DEFAULT_STRATEGY}）。"
             f"利用可能な戦略は strategies/registry.py を参照。",
    )
    return parser.parse_args()


def run(strategy_name):
    strategy_fn = registry.get_strategy(strategy_name)

    cache_filename = f"backtest_prices_{config.TARGET_MARKET}_{config.BACKTEST_YEARS}y.csv"

    print("=== バックテスト開始 ===")
    print(f"[設定] 戦略: {strategy_name}")
    print(f"[設定] 対象市場: {config.TARGET_MARKET}")
    print(f"[設定] 対象期間: 過去{config.BACKTEST_YEARS}年")
    print(f"[設定] 保有日数: {config.BACKTEST_HOLD_DAYS}日 "
          f"/ 利確: +{config.BACKTEST_TAKE_PROFIT_PCT}% "
          f"/ 損切り: -{config.BACKTEST_STOP_LOSS_PCT}%")

    print("\n[main] 対象銘柄リスト取得中...")
    target_codes = download.get_target_codes()

    print("\n[main] 株価データ取得中（キャッシュがあれば差分だけ取得）...")
    price_df = download.get_price_history_incremental(
        cache_filename=cache_filename,
        years=config.BACKTEST_YEARS,
    )

    print(f"\n[main] 戦略「{strategy_name}」でシグナル検出中...")
    signals, price_data_by_code = strategy_fn(price_df, target_codes)

    print("\n[main] バックテスト実行中...")
    trades = backtest.run_backtest(signals, price_data_by_code)

    print("\n[main] 結果集計中...")
    summary = backtest.summarize_trades(trades)
    backtest.print_summary(summary)

    _save_results(trades, summary, strategy_name)

    print("\n=== バックテスト完了 ===")


def main():
    args = parse_args()
    strategy_name = args.strategy

    # 戦略名が未登録の場合は、通知するまでもない設定ミスなので早期に終了する
    try:
        registry.get_strategy(strategy_name)
    except ValueError as e:
        print(f"[エラー] {e}")
        sys.exit(1)

    start = time.time()
    try:
        run(strategy_name)
    except Exception as e:
        print(f"エラー: {e}")
        notifier.notify_error(e, context=f"run_backtest.py（戦略: {strategy_name}）実行中")
        raise
    finally:
        print(f"実行時間: {time.time() - start:.1f}秒")


def _save_results(trades, summary, strategy_name):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- summary.json（固定ファイル名、常に最新で上書き） ----
    summary_with_meta = {
        "strategy": strategy_name,
        "generated_at": timestamp,
        **summary,
    }
    summary_path = os.path.join(config.OUTPUT_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_with_meta, f, ensure_ascii=False, indent=2, default=str)
    print(f"[main] サマリーを保存しました: {summary_path}")

    if not trades:
        print("[main] トレードが0件のため、trades.csv以降の出力はスキップします。")
        return

    # ---- trades.csv（固定ファイル名、個別トレード明細） ----
    df = pd.DataFrame(trades)
    df["result"] = df["profit_pct"].apply(lambda x: "win" if x > 0 else "loss")

    column_order = [
        "code", "signal_date", "entry_date", "entry_price",
        "exit_date", "exit_price", "exit_reason",
        "holding_days", "profit_pct", "result",
    ]
    df = df[[c for c in column_order if c in df.columns]]
    df = df.sort_values("signal_date")

    trades_path = os.path.join(config.OUTPUT_DIR, "trades.csv")
    df.to_csv(trades_path, index=False, encoding="utf-8-sig")
    print(f"[main] 個別トレード結果を保存しました: {trades_path}")

    # ---- losing_trades.csv（負けトレードのみ、敗因分析用） ----
    losses_df = df[df["result"] == "loss"].copy()
    if not losses_df.empty:
        losses_path = os.path.join(config.OUTPUT_DIR, "losing_trades.csv")
        losses_df.to_csv(losses_path, index=False, encoding="utf-8-sig")
        print(f"[main] 負けトレードのみ抽出したファイルを保存しました: {losses_path}")

        print("\n[main] 負けトレードの内訳:")
        print(losses_df["exit_reason"].value_counts().to_string())

    # ---- monthly.csv（月別成績） ----
    monthly_df = backtest.build_monthly_summary(trades)
    if not monthly_df.empty:
        monthly_path = os.path.join(config.OUTPUT_DIR, "monthly.csv")
        monthly_df.to_csv(monthly_path, index=False, encoding="utf-8-sig")
        print(f"[main] 月別成績を保存しました: {monthly_path}")

    # ---- yearly.csv（年別成績） ----
    yearly_df = backtest.build_yearly_summary(trades)
    if not yearly_df.empty:
        yearly_path = os.path.join(config.OUTPUT_DIR, "yearly.csv")
        yearly_df.to_csv(yearly_path, index=False, encoding="utf-8-sig")
        print(f"[main] 年別成績を保存しました: {yearly_path}")
        print("\n[main] 年別成績:")
        print(yearly_df.to_string(index=False))

    # ---- equity_curve.csv（資産推移） ----
    equity_df = backtest.build_equity_curve(
        trades,
        initial_capital=config.BACKTEST_INITIAL_CAPITAL,
        position_size_pct=config.BACKTEST_POSITION_SIZE_PCT,
    )
    if not equity_df.empty:
        equity_path = os.path.join(config.OUTPUT_DIR, "equity_curve.csv")
        equity_df.to_csv(equity_path, index=False, encoding="utf-8-sig")
        print(f"[main] 資産推移を保存しました: {equity_path}")
        final_capital = equity_df["capital"].iloc[-1]
        total_return_pct = (final_capital / config.BACKTEST_INITIAL_CAPITAL - 1) * 100
        print(f"[main] 初期資金 {config.BACKTEST_INITIAL_CAPITAL:,.0f}円 → "
              f"最終資金 {final_capital:,.0f}円 "
              f"（トータルリターン: {total_return_pct:+.2f}%）")


if __name__ == "__main__":
    main()
