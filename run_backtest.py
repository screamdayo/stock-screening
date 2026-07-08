"""
run_backtest.py
過去N年分のデータで、指定した戦略のバックテストを実行するエントリーポイント。

使い方:
    python run_backtest.py                  # デフォルト戦略（kuitto）で実行
    python run_backtest.py --strategy kuitto
    python run_backtest.py --strategy golden_cross   # 新しい戦略を追加した場合
    python run_backtest.py --condition rising        # MA上向き判定方式を強制指定して比較
    python run_backtest.py --condition turning

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
from logger import get_logger

logger = get_logger(__name__)

DEFAULT_STRATEGY = "kuitto"


def parse_args():
    parser = argparse.ArgumentParser(description="株価バックテストツール")
    parser.add_argument(
        "--strategy",
        default=DEFAULT_STRATEGY,
        help=f"使用する戦略名（デフォルト: {DEFAULT_STRATEGY}）。"
             f"利用可能な戦略は strategies/registry.py を参照。",
    )
    parser.add_argument(
        "--condition",
        choices=["config", "rising", "turning"],
        default="config",
        help="MAの上向き判定方式を一時的に切り替える。"
             " 'config'（デフォルト）はconfig.pyの設定をそのまま使う。"
             " 'rising'はREQUIRE_MA_RISING=True/REQUIRE_MA_TURNING=Falseに強制。"
             " 'turning'はREQUIRE_MA_RISING=False/REQUIRE_MA_TURNING=Trueに強制。"
             " rising/turningを両方実行して結果を見比べることで、"
             " どちらの判定方式が成績が良いか比較できる。",
    )
    return parser.parse_args()


def _apply_condition_override(condition, strategy_name):
    """
    --condition の指定に応じて、config.STRATEGY_CONFIG[strategy_name] の
    MA判定フラグを一時的に上書きする。
    'config' の場合は何もしない（config.py の値をそのまま使う）。
    """
    cfg = config.get_strategy_config(strategy_name)

    if condition == "rising":
        cfg["REQUIRE_MA_RISING"] = True
        cfg["REQUIRE_MA_TURNING"] = False
        logger.info(f"--condition rising: STRATEGY_CONFIG['{strategy_name}'] の "
                     f"REQUIRE_MA_RISING=True, REQUIRE_MA_TURNING=False に上書き")
    elif condition == "turning":
        cfg["REQUIRE_MA_RISING"] = False
        cfg["REQUIRE_MA_TURNING"] = True
        logger.info(f"--condition turning: STRATEGY_CONFIG['{strategy_name}'] の "
                     f"REQUIRE_MA_RISING=False, REQUIRE_MA_TURNING=True に上書き")
    else:
        logger.info(f"--condition config: STRATEGY_CONFIG['{strategy_name}']の設定をそのまま使用 "
                     f"(REQUIRE_MA_RISING={cfg['REQUIRE_MA_RISING']}, "
                     f"REQUIRE_MA_TURNING={cfg['REQUIRE_MA_TURNING']})")


def run(strategy_name, run_label=None):
    if run_label is None:
        run_label = strategy_name

    strategy_fn = registry.get_strategy(strategy_name)

    cache_filename = f"backtest_prices_{config.TARGET_MARKET}_{config.BACKTEST_YEARS}y.csv"

    logger.info("=== バックテスト開始 ===")
    logger.info(f"[設定] 戦略: {strategy_name}")
    logger.info(f"[設定] 対象市場: {config.TARGET_MARKET}")
    logger.info(f"[設定] 対象期間: 過去{config.BACKTEST_YEARS}年")
    cfg = config.get_strategy_config(strategy_name)
    logger.info(f"[設定] MA上向き判定: REQUIRE_MA_RISING={cfg['REQUIRE_MA_RISING']}, "
                f"REQUIRE_MA_TURNING={cfg['REQUIRE_MA_TURNING']}")
    logger.info(f"[設定] 保有日数: {config.BACKTEST_HOLD_DAYS}日 "
                f"/ 利確: +{config.BACKTEST_TAKE_PROFIT_PCT}% "
                f"/ 損切り: -{config.BACKTEST_STOP_LOSS_PCT}%")

    logger.info("対象銘柄リスト取得中...")
    target_codes = download.get_target_codes()

    logger.info("株価データ取得中（キャッシュがあれば差分だけ取得）...")
    price_df = download.get_price_history_incremental(
        cache_filename=cache_filename,
        years=config.BACKTEST_YEARS,
    )

    logger.info(f"戦略「{strategy_name}」でシグナル検出中...")
    signals, price_data_by_code = strategy_fn(price_df, target_codes)

    logger.info("バックテスト実行中...")
    trades = backtest.run_backtest(signals, price_data_by_code)

    logger.info("結果集計中...")
    summary = backtest.summarize_trades(trades)
    backtest.print_summary(summary)

    _save_results(trades, summary, run_label)

    logger.info("=== バックテスト完了 ===")


def main():
    args = parse_args()
    strategy_name = args.strategy
    condition = args.condition

    # 戦略名が未登録の場合は、通知するまでもない設定ミスなので早期に終了する
    try:
        registry.get_strategy(strategy_name)
    except ValueError as e:
        logger.error(f"[エラー] {e}")
        sys.exit(1)

    _apply_condition_override(condition, strategy_name)

    # rising/turningを切り替えて比較したい場合、出力ファイルが上書きされて
    # 比較できなくなると困るため、conditionを明示指定した場合はラベルとして使う
    run_label = f"{strategy_name}_{condition}" if condition != "config" else strategy_name

    start = time.time()
    try:
        run(strategy_name, run_label)
    except Exception as e:
        logger.error(f"エラー: {e}", exc_info=True)
        notifier.notify_error(e, context=f"run_backtest.py（戦略: {strategy_name}, 条件: {condition}）実行中")
        raise
    finally:
        logger.info(f"実行時間: {time.time() - start:.1f}秒")


def _save_results(trades, summary, run_label):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- summary.json（固定ファイル名、常に最新で上書き） ----
    summary_with_meta = {
        "run_label": run_label,
        "generated_at": timestamp,
        **summary,
    }
    summary_path = os.path.join(config.OUTPUT_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_with_meta, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"サマリーを保存しました: {summary_path}")

    if not trades:
        logger.info("トレードが0件のため、trades.csv以降の出力はスキップします。")
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
    logger.info(f"個別トレード結果を保存しました: {trades_path}")

    # ---- losing_trades.csv（負けトレードのみ、敗因分析用） ----
    losses_df = df[df["result"] == "loss"].copy()
    if not losses_df.empty:
        losses_path = os.path.join(config.OUTPUT_DIR, "losing_trades.csv")
        losses_df.to_csv(losses_path, index=False, encoding="utf-8-sig")
        logger.info(f"負けトレードのみ抽出したファイルを保存しました: {losses_path}")
        logger.info("負けトレードの内訳:\n" + losses_df["exit_reason"].value_counts().to_string())

    # ---- monthly.csv（月別成績） ----
    monthly_df = backtest.build_monthly_summary(trades)
    if not monthly_df.empty:
        monthly_path = os.path.join(config.OUTPUT_DIR, "monthly.csv")
        monthly_df.to_csv(monthly_path, index=False, encoding="utf-8-sig")
        logger.info(f"月別成績を保存しました: {monthly_path}")

    # ---- yearly.csv（年別成績） ----
    yearly_df = backtest.build_yearly_summary(trades)
    if not yearly_df.empty:
        yearly_path = os.path.join(config.OUTPUT_DIR, "yearly.csv")
        yearly_df.to_csv(yearly_path, index=False, encoding="utf-8-sig")
        logger.info(f"年別成績を保存しました: {yearly_path}")
        logger.info("年別成績:\n" + yearly_df.to_string(index=False))

    # ---- equity_curve.csv（資産推移） ----
    equity_df = backtest.build_equity_curve(
        trades,
        initial_capital=config.BACKTEST_INITIAL_CAPITAL,
        position_size_pct=config.BACKTEST_POSITION_SIZE_PCT,
    )
    if not equity_df.empty:
        equity_path = os.path.join(config.OUTPUT_DIR, "equity_curve.csv")
        equity_df.to_csv(equity_path, index=False, encoding="utf-8-sig")
        final_capital = equity_df["capital"].iloc[-1]
        total_return_pct = (final_capital / config.BACKTEST_INITIAL_CAPITAL - 1) * 100
        logger.info(f"資産推移を保存しました: {equity_path}")
        logger.info(f"初期資金 {config.BACKTEST_INITIAL_CAPITAL:,.0f}円 → "
                     f"最終資金 {final_capital:,.0f}円 "
                     f"（トータルリターン: {total_return_pct:+.2f}%）")


if __name__ == "__main__":
    main()
