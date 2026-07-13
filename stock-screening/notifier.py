"""
notifier.py
スクリーニング結果をDiscordに通知する。
実行中にエラーが起きた場合の通知（notify_error）もここで扱う。
"""

import traceback
import requests
from datetime import datetime

import config
from logger import get_logger

logger = get_logger(__name__)


def notify(results):
    today = datetime.now().strftime("%Y/%m/%d")
    cfg = config.get_strategy_config()  # ACTIVE_STRATEGYの設定を使う

    if not results:
        msg = f"📊 **株スクリーニング結果 {today}**\n該当銘柄なし"
        _post(msg)
        return

    lines = []
    for r in results[:config.MAX_RESULTS_TO_SHOW]:
        url = config.KABUTAN_CHART_URL_TEMPLATE.format(code=r["code"])
        # URLを < > で囲むとDiscordが埋め込みカード（OGP画像等）を表示しなくなる。
        # 銘柄数が多いと埋め込みでチャットが縦に長くなりすぎるため無効化している。
        name = f"（{r['name']}）" if r.get("name") else ""
        lines.append(
            f"・**{r['code']}**{name}｜終値 {r['close']}円｜"
            f"MA{cfg['MA_SHORT_PERIOD']} {r['ma_short_prev']}→{r['ma_short_today']}↑｜"
            f"<{url}>"
        )

    header = (
        f"📊 **株スクリーニング結果 {today}**\n"
        f"✅ {cfg['MA_SHORT_PERIOD']}日MA上向き"
        + ("＆陽線" if cfg["REQUIRE_BULLISH_CANDLE"] else "")
        + f"（{len(results)}件）\n"
    )

    remaining = len(results) - config.MAX_RESULTS_TO_SHOW
    footer = f"\n...他{remaining}件" if remaining > 0 else ""

    msg = header + "\n".join(lines) + footer
    _post_long(msg)


def notify_error(error, context=""):
    """
    実行中に例外が発生した際にDiscordへ通知する。

    error: 発生した例外（Exceptionインスタンス）
    context: どの処理で起きたかを示す短い文字列（任意、例: "main.py実行中"）
    """
    now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    tb_text = traceback.format_exc()

    # Discordの1メッセージ上限に収まるようトレースバックは適度に切り詰める
    max_tb_length = 1500
    if len(tb_text) > max_tb_length:
        tb_text = tb_text[-max_tb_length:]
        tb_text = "...(省略)...\n" + tb_text

    context_line = f"箇所: {context}\n" if context else ""

    msg = (
        f"🚨 **エラー発生 {now}**\n"
        f"{context_line}"
        f"エラー内容: `{error}`\n"
        f"```\n{tb_text}\n```"
    )
    _post_long(msg)


def notify_backtest_result(summary, run_label, equity_summary=None):
    """
    バックテスト結果（勝率・PF・平均利益損失など）をDiscordに通知する。

    GitHub Actions上でバックテストを実行した場合、data/やoutput/の中身は
    実行終了と同時に消えてしまう（コンテナが使い捨てのため）ので、
    結果はこの通知でしか後から確認できない。ローカル実行時も、
    output/フォルダの中身を見るまでもなくDiscordでサマリーを確認できる。

    summary: backtest.summarize_trades() の戻り値
    run_label: 実行ラベル（例: "kuitto" や "kuitto_rising"）
    equity_summary: 任意。{"initial_capital", "final_capital", "total_return_pct"} を
                     含む辞書を渡すと資産推移の要約も併記する。
    """
    now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    if summary["total_trades"] == 0:
        msg = (
            f"📈 **バックテスト結果 {now}**\n"
            f"戦略/条件: `{run_label}`\n"
            f"トレードが0件でした。"
        )
        _post(msg)
        return

    lines = [
        f"📈 **バックテスト結果 {now}**",
        f"戦略/条件: `{run_label}`",
        "",
        f"総トレード数　　　: {summary['total_trades']}件",
        f"勝率　　　　　　　: {summary['win_rate']}%"
        f"（{summary['win_count']}勝 {summary['loss_count']}敗）",
        f"平均利益（勝ち時）: +{summary['avg_profit_pct']}%",
        f"平均損失（負け時）: {summary['avg_loss_pct']}%",
        f"プロフィットファクター: {summary['profit_factor']}",
        f"平均保有日数　　　: {summary['avg_holding_days']}日",
        "",
        f"利確決済: {summary['take_profit_count']}件 / "
        f"損切り決済: {summary['stop_loss_count']}件 / "
        f"期日決済: {summary['time_exit_count']}件",
    ]

    if equity_summary:
        lines += [
            "",
            f"初期資金: {equity_summary['initial_capital']:,.0f}円 → "
            f"最終資金: {equity_summary['final_capital']:,.0f}円 "
            f"（トータルリターン: {equity_summary['total_return_pct']:+.2f}%）",
        ]

    msg = "\n".join(lines)
    _post_long(msg)


def _post(msg):
    if not config.DISCORD_WEBHOOK_URL:
        logger.warning("DISCORD_WEBHOOK_URLが未設定のため通知をスキップします。")
        return
    requests.post(config.DISCORD_WEBHOOK_URL, json={"content": msg})


def _post_long(msg):
    chunk_size = config.DISCORD_MESSAGE_CHUNK_SIZE
    for i in range(0, len(msg), chunk_size):
        _post(msg[i:i + chunk_size])
