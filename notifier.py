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
NEXT_OPEN_GAP_MAX_PCT = 0.5
STOP_LOSS_PCT = 5.0


def _stock_label(item):
    code = str(item.get("code") or "").strip()
    name = str(item.get("name") or "").strip()
    if name and code:
        return f"{name} ({code})"
    return name or code or "?"


def notify(results, rescue_results=None, primary_results=None):
    today = datetime.now().strftime("%Y/%m/%d")
    rescue_results = rescue_results or []
    primary_results = primary_results or []

    # 目視撤廃後の自動くいっと通知。
    if results and all(r.get("auto_filtered") for r in results) and not rescue_results and not primary_results:
        # 5銘柄以上出た日は、検証で採用した優先順位に合わせて
        # MA5がMA25に近い順（乖離の絶対値が小さい順）でDiscord表示する。
        display_results = list(results)
        sorted_by_ma25 = len(display_results) >= 5
        if sorted_by_ma25:
            display_results.sort(
                key=lambda r: abs(r.get("ma5_vs_ma25_pct", float("inf")))
            )

        lines = []
        for r in display_results[:20]:
            label = _stock_label(r)
            decline = r.get("ma5_prior5d_decline_pct")
            gap = r.get("ma5_vs_ma25_pct")
            bull = r.get("bull_candle_pct")
            volume = r.get("volume_ratio")
            close = r.get("close")
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
            if close is not None:
                max_open = float(close) * (1 + NEXT_OPEN_GAP_MAX_PCT / 100)
                stop_at_max_open = max_open * (1 - STOP_LOSS_PCT / 100)
                lines.append(
                    f"   ↳ 翌朝寄値 **{max_open:,.1f}円以下なら買い** / 超えたら見送り "
                    f"（終値 {float(close):,.1f}円 × +{NEXT_OPEN_GAP_MAX_PCT:.1f}%）"
                )
                lines.append(
                    f"   🛑 損切り **実際の買値 × 0.95** "
                    f"（参考：寄値上限で買った場合 **{stop_at_max_open:,.1f}円**）"
                )

        remaining = len(display_results) - 20
        if remaining > 0:
            lines.append(f"…他{remaining}件")

        order_note = "\n📐 **MA25に近い順で表示**" if sorted_by_ma25 else ""
        msg = (
            f"📊 **くいっと押し目版 {today}**\n"
            f"🤖 **目視判定なし / 出来高1.25倍以上 / 自動通過 {len(results)}件**\n"
            f"🌅 **翌朝ルール：前日終値比 +{NEXT_OPEN_GAP_MAX_PCT:.1f}%以内で寄れば買い、超えたら見送り**\n"
            f"🛑 **損切り：実際の買値から -{STOP_LOSS_PCT:.0f}%**"
            f"{order_note}\n"
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
            label = _stock_label(r)
            lines.append(f"🥇 #{rank} **{kind}** {label}")
        remaining = len(primary_results) - 15
        if remaining > 0:
            lines.append(f"…他{remaining}件")
        parts.append(f"🏆 **旧A/B本命候補 {len(primary_results)}件**\n" + "\n".join(lines))

    if results:
        preview = []
        for r in results[:10]:
            label = _stock_label(r)
            preview.append(f"• {label}")
        remaining = len(results) - 10
        if remaining > 0:
            preview.append(f"…他{remaining}件")
        parts.append(f"📋 **旧通常候補 {len(results)}件**\n" + "\n".join(preview))

    if rescue_results:
        preview = []
        for r in rescue_results[:10]:
            label = _stock_label(r)
            preview.append(f"🟣 {label}")
        remaining = len(rescue_results) - 10
        if remaining > 0:
            preview.append(f"…他{remaining}件")
        parts.append(f"🟣 **旧救済候補 {len(rescue_results)}件**\n" + "\n".join(preview))

    parts.append(f"📈 **チャートで確認**\n{SCREENING_VIEW_URL}")
    _post_long("\n\n".join(parts))


def notify_sell_signals(alerts):
    if not alerts:
        return
    today = datetime.now().strftime("%Y/%m/%d")
    lines = []
    for a in alerts:
        label = _stock_label(a)
        reason = a.get("reason")
        if reason == "strict_gakutto":
            reason_text = "厳しめがくっと成立"
        elif reason == "max_hold":
            reason_text = "最大15営業日到達"
        else:
            reason_text = str(reason or "売りルール成立")
        hold_days = a.get("hold_days")
        close = a.get("latest_close")
        detail = [reason_text]
        if hold_days is not None:
            detail.append(f"保有{hold_days}営業日")
        if close is not None:
            detail.append(f"終値 {close:,.1f}")
        lines.append(
            f"🔴 **{label}** — {' / '.join(detail)}\n"
            f"➡️ **翌営業日始値で売却**"
        )

    msg = (
        f"🚨 **売りルール成立 {today}**\n"
        f"感情判断なし・ルール通り実行\n\n"
        + "\n\n".join(lines)
    )
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
