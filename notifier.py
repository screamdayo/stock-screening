"""
notifier.py
スクリーニング結果をDiscordに通知する。
"""

import traceback
import requests
from datetime import datetime

import config
from logger import get_logger

logger = get_logger(__name__)

SCREENING_VIEW_URL = "https://screamdayo.github.io/stock-screening/screening.html"


def notify(results):
    today = datetime.now().strftime("%Y/%m/%d")
    cfg = config.get_strategy_config()

    if not results:
        _post(f"📊 **株スクリーニング結果 {today}**\n該当銘柄なし\n{SCREENING_VIEW_URL}")
        return

    preview_limit = 15
    names = [f"・{(r.get('name') or r['code'])}" for r in results[:preview_limit]]
    remaining = len(results) - preview_limit
    footer = f"\n…他{remaining}件" if remaining > 0 else ""

    header = (
        f"📊 **株スクリーニング結果 {today}**\n"
        f"✅ {cfg['MA_SHORT_PERIOD']}日MA上向き"
        + ("＆陽線" if cfg["REQUIRE_BULLISH_CANDLE"] else "")
        + f"（{len(results)}件）\n\n"
    )

    msg = header + "\n".join(names) + footer + f"\n\n📈 **チャートで確認**\n{SCREENING_VIEW_URL}"
    _post_long(msg)


def notify_error(error, context=""):
    now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    tb_text = traceback.format_exc()
    if len(tb_text) > 1500:
        tb_text = "...(省略)...\n" + tb_text[-1500:]
    context_line = f"箇所: {context}\n" if context else ""
    _post_long(
        f"🚨 **エラー発生 {now}**\n"
        f"{context_line}"
        f"エラー内容: `{error}`\n"
        f"```\n{tb_text}\n```"
    )


def notify_backtest_result(summary, run_label, equity_summary=None):
    now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    if summary["total_trades"] == 0:
        _post(f"📈 **バックテスト結果 {now}**\n戦略/条件: `{run_label}`\nトレードが0件でした。")
        return

    lines = [
        f"📈 **バックテスト結果 {now}**",
        f"戦略/条件: `{run_label}`",
        "",
        f"総トレード数　　　: {summary['total_trades']}件",
        f"勝率　　　　　　　: {summary['win_rate']}%（{summary['win_count']}勝 {summary['loss_count']}敗）",
        f"平均利益（勝ち時）: +{summary['avg_profit_pct']}%",
        f"平均損失（負け時）: {summary['avg_loss_pct']}%",
        f"プロフィットファクター: {summary['profit_factor']}",
        f"平均保有日数　　　: {summary['avg_holding_days']}日",
        "",
        f"利確決済: {summary['take_profit_count']}件 / 損切り決済: {summary['stop_loss_count']}件 / 期日決済: {summary['time_exit_count']}件",
    ]
    if equity_summary:
        lines += [
            "",
            f"初期資金: {equity_summary['initial_capital']:,.0f}円 → 最終資金: {equity_summary['final_capital']:,.0f}円 "
            f"（トータルリターン: {equity_summary['total_return_pct']:+.2f}%）",
        ]
    _post_long("\n".join(lines))


def _post(msg):
    if not config.DISCORD_WEBHOOK_URL:
        logger.warning("DISCORD_WEBHOOK_URLが未設定のため通知をスキップします。")
        return
    requests.post(config.DISCORD_WEBHOOK_URL, json={"content": msg})


def _post_long(msg):
    chunk_size = config.DISCORD_MESSAGE_CHUNK_SIZE
    for i in range(0, len(msg), chunk_size):
        _post(msg[i:i + chunk_size])
