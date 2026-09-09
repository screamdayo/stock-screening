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


def notify(results, rescue_results=None, primary_results=None):
    today = datetime.now().strftime("%Y/%m/%d")
    rescue_results = rescue_results or []
    primary_results = primary_results or []

    # 目視撤廃後の自動くいっと通知。
    if results and all(r.get("auto_filtered") for r in results) and not rescue_results and not primary_results:
        lines = []
        for r in results[:20]:
            label = r.get("name") or r["code"]
            decline = r.get("ma5_prior5d_decline_pct")
            gap = r.get("ma5_vs_ma25_pct")
            bull = r.get("bull_candle_pct")
            volume = r.get("volume_ratio")
            detail = []
            if decline is not None:
                detail.append(f"MA5直前5日 {decline:+.2f}%")
            if gap is not None:
                detail.append(f"MA5/25乖離 {gap:+.2f}%")
            if bull is not None:
                detail.append(f"当日 {bull:+.2f}%")
            if volume is not None:
                detail.append(f"出来高 {volume:.2f}倍")
            suffix = f" — {' / '.join(detail)}" if detail else ""
            lines.append(f"🟢 {label}{suffix}")

        remaining = len(results) - 20
        if remaining > 0:
            lines.append(f"…他{remaining}件")

        msg = (
            f"📊 **くいっと押し目版 {today}**\n"
            f"🤖 **目視判定なし / 出来高1.25倍以上 / 自動通過 {len(results)}件**\n"
            + "\n".join(lines)
            + f"\n\n📈 **チャート**\n{SCREENING_VIEW_URL}"
        )
        _post_long(msg)
        return

    if not results and not rescue_results and not primary_results:
        _post(
            f"📊 **くいっと押し目版 {today}**\n"
            f"🤖 目視判定なし / 出来高1.25倍以上 / 本日の該当銘柄なし\n{SCREENING_VIEW_URL}"
        )
        return

    # 旧ma5_breakoutの比較検証用通知。日次本番では通常ここを通らない。
    parts = [f"📊 **株スクリーニング結果 {today}**"]

    if primary_results:
        lines = []
        for r in primary_results[:15]:
            rank = r.get("production_rank")
            kind = r.get("production_strategy") or "?"
            label = r.get("name") or r["code"]
            lines.append(f"🥇 #{rank} **{kind}** {label}")
        remaining = len(primary_results) - 15
        if remaining > 0:
            lines.append(f"…他{remaining}件")
        parts.append(f"🏆 **旧A/B本命候補 {len(primary_results)}件**\n" + "\n".join(lines))

    if results:
        preview = []
        for r in results[:10]:
            label = r.get("name") or r["code"]
            preview.append(f"• {label}")
        remaining = len(results) - 10
        if remaining > 0:
            preview.append(f"…他{remaining}件")
        parts.append(f"📋 **旧通常候補 {len(results)}件**\n" + "\n".join(preview))

    if rescue_results:
        preview = []
        for r in rescue_results[:10]:
            label = r.get("name") or r["code"]
            preview.append(f"🟣 {label}")
        remaining = len(rescue_results) - 10
        if remaining > 0:
            preview.append(f"…他{remaining}件")
        parts.append(f"🟣 **旧救済候補 {len(rescue_results)}件**\n" + "\n".join(preview))

    parts.append(f"📈 **チャートで確認**\n{SCREENING_VIEW_URL}")
    _post_long("\n\n".join(parts))


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
