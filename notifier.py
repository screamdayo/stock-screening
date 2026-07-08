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
        lines.append(
            f"・**{r['code']}**｜終値 {r['close']}円｜"
            f"MA{cfg['MA_SHORT_PERIOD']} {r['ma_short_prev']}→{r['ma_short_today']}↑\n"
            f"  {url}"
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


def _post(msg):
    if not config.DISCORD_WEBHOOK_URL:
        logger.warning("DISCORD_WEBHOOK_URLが未設定のため通知をスキップします。")
        return
    requests.post(config.DISCORD_WEBHOOK_URL, json={"content": msg})


def _post_long(msg):
    chunk_size = config.DISCORD_MESSAGE_CHUNK_SIZE
    for i in range(0, len(msg), chunk_size):
        _post(msg[i:i + chunk_size])
