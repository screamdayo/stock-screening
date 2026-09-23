"""
notifier.py
スクリーニング結果をDiscordに通知する。
"""

import math
import traceback
import requests
from datetime import datetime

import config
from logger import get_logger

logger = get_logger(__name__)

SCREENING_VIEW_URL = "https://screamdayo.github.io/stock-screening/screening.html"
GC_SCREENING_VIEW_URL = "https://screamdayo.github.io/stock-screening/gc.html"
PINCH_SENSOR_VIEW_URL = "https://screamdayo.github.io/stock-screening/pinch.html"
SELLING_CLIMAX_VIEW_URL = "https://screamdayo.github.io/stock-screening/selling_climax.html"
NEXT_OPEN_GAP_MAX_PCT = 0.5
STOP_LOSS_PCT = 5.0


def _stock_label(item):
    code = str(item.get("code") or "").strip()
    name = str(item.get("name") or "").strip()
    if name and code:
        return f"{name} ({code})"
    return name or code or "?"


def _safe_tick_size(price):
    """東証の一般銘柄側の呼値を使う安全側テーブル。

    TOPIX500等はより細かい呼値を使える場合があるが、一般銘柄側の
    刻みに合わせれば有効な価格になり、+0.5%上限を超えない。
    """
    p = float(price)
    if p <= 3000:
        return 1.0
    if p <= 5000:
        return 5.0
    if p <= 30000:
        return 10.0
    if p <= 50000:
        return 50.0
    if p <= 300000:
        return 100.0
    if p <= 500000:
        return 500.0
    if p <= 3000000:
        return 1000.0
    if p <= 5000000:
        return 5000.0
    if p <= 30000000:
        return 10000.0
    if p <= 50000000:
        return 50000.0
    return 100000.0


def _floor_to_valid_tick(price):
    p = float(price)
    tick = _safe_tick_size(p)
    # 浮動小数誤差で1呼値余計に下がらないよう微小値を足す。
    return math.floor((p + 1e-9) / tick) * tick


def _fmt_yen(price):
    p = float(price)
    if p.is_integer():
        return f"{p:,.0f}円"
    return f"{p:,.1f}円"


def notify(results, rescue_results=None, primary_results=None):
    today = datetime.now().strftime("%Y/%m/%d")
    rescue_results = rescue_results or []
    primary_results = primary_results or []

    # 目視撤廃後の自動くいっと通知。
    if results and all(r.get("auto_filtered") for r in results) and not rescue_results and not primary_results:
        # 伸びスコアを最優先し、同点なら従来どおりMA5がMA25に近い順で表示する。
        display_results = list(results)
        display_results.sort(
            key=lambda r: (
                -int(r.get("runner_score") or 0),
                abs(r.get("ma5_vs_ma25_pct", float("inf"))),
                str(r.get("code") or ""),
            )
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
            score = int(r.get("runner_score") or 0)
            atr = r.get("atr14_pct")
            dd20 = r.get("dd20_pct")
            detail.append(f"伸び {score}/4")
            if atr is not None:
                detail.append(f"ATR {atr:.2f}%")
            if dd20 is not None:
                detail.append(f"DD20 {dd20:+.2f}%")
            if decline is not None:
                detail.append(f"MA5直前5日 {decline:+.2f}%")
            if gap is not None:
                detail.append(f"MA5/25乖離 {gap:+.2f}%")
            if bull is not None:
                detail.append(f"当日 {bull:+.2f}%")
            if volume is not None:
                detail.append(f"出来高 {volume:.2f}倍")
            suffix = f" — {' / '.join(detail)}" if detail else ""
            earnings_note = ""
            if r.get("earnings_within_10bd"):
                ed = r.get("earnings_date") or "日付不明"
                bd = r.get("earnings_business_days")
                when = f"（{bd}営業日後 / {ed}）" if bd is not None else f"（{ed}）"
                earnings_note = f"\n   ⚠️ **10営業日以内に決算あり** {when}"
            lines.append(f"🟢 {label}{suffix}{earnings_note}")
            if close is not None:
                theoretical_max_open = float(close) * (1 + NEXT_OPEN_GAP_MAX_PCT / 100)
                max_open = _floor_to_valid_tick(theoretical_max_open)
                theoretical_stop = max_open * (1 - STOP_LOSS_PCT / 100)
                stop_at_max_open = _floor_to_valid_tick(theoretical_stop)
                lines.append(
                    f"   ↳ 翌朝寄指 **{_fmt_yen(max_open)}以下なら買い** / 超えたら見送り "
                    f"（理論上限 {_fmt_yen(theoretical_max_open)} → 呼値切下げ）"
                )
                lines.append(
                    f"   🛑 損切り **実際の買値 × 0.95** "
                    f"（参考：寄指上限で買った場合 **{_fmt_yen(stop_at_max_open)}**）"
                )

        remaining = len(display_results) - 20
        if remaining > 0:
            lines.append(f"…他{remaining}件")

        order_note = "\n⭐ **伸びスコア順 → 同点はMA25に近い順**"
        msg = (
            f"📊 **くいっと押し目版 {today}**\n"
            f"🤖 **目視判定なし / 出来高1.25倍以上 / 自動通過 {len(results)}件**\n"
            f"🌅 **翌朝ルール：前日終値比 +{NEXT_OPEN_GAP_MAX_PCT:.1f}%以内で寄れば買い、超えたら見送り**\n"
            f"💴 **注文価格は呼値に合わせて安全側へ切り下げ表示**\n"
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




def notify_kuitto_variants(refined_results, elite_results):
    """Kuitto 204 / 59 are separate shadow strategies; notify only when either fires."""
    refined_results = refined_results or []
    elite_results = elite_results or []
    if not refined_results and not elite_results:
        logger.info("くいっと204 / 59 通知なし（本日の該当なし）")
        return

    today = datetime.now().strftime("%Y/%m/%d")

    def _line(r):
        label = _stock_label(r)
        parts = []
        decline = r.get("ma5_prior5d_decline_pct")
        close_gap = r.get("close_vs_ma5_pct")
        atr = r.get("atr14_pct")
        dd20 = r.get("dd20_pct")
        slope = r.get("ma25_slope5_pct")
        if decline is not None:
            parts.append(f"MA5直前5日 {decline:+.2f}%")
        if close_gap is not None:
            parts.append(f"終値/MA5 {close_gap:+.2f}%")
        if atr is not None:
            parts.append(f"ATR {atr:.2f}%")
        if dd20 is not None:
            parts.append(f"DD20 {dd20:+.2f}%")
        if slope is not None:
            parts.append(f"MA25傾き {slope:+.2f}%")
        earnings_note = ""
        if r.get("earnings_within_10bd"):
            ed = r.get("earnings_date") or "日付不明"
            bd = r.get("earnings_business_days")
            when = f"（{bd}営業日後 / {ed}）" if bd is not None else f"（{ed}）"
            earnings_note = f"\n   ⚠️ **10営業日以内に決算あり** {when}"
        detail = " / ".join(parts)
        return f"• **{label}**" + (f" — {detail}" if detail else "") + earnings_note

    sections = []
    if refined_results:
        sections.append(
            f"🔵 **くいっと204：{len(refined_results)}件**\n"
            f"3段階ルール / フォワード出口16営業日固定\n"
            + "\n".join(_line(r) for r in refined_results)
        )
    if elite_results:
        sections.append(
            f"🟣 **くいっと59：{len(elite_results)}件**\n"
            f"204条件 + MA5直前5日≤-4% + 終値/MA5≥+2% / 出口16営業日固定\n"
            + "\n".join(_line(r) for r in elite_results)
        )

    msg = (
        f"🧪 **別戦略シグナル {today}**\n"
        f"⚠️ 現行くいっとの本番候補とは別枠。フォワード比較用。\n\n"
        + "\n\n".join(sections)
    )
    _post_long(msg)


def notify_gc_strong_breakout(results):
    """GC強ブレイクは希少シグナルなので、該当時だけ別メッセージで通知する。"""
    if not results:
        logger.info("GC強ブレイク通知なし（本日の該当なし）")
        return

    today = datetime.now().strftime("%Y/%m/%d")
    lines = []
    for r in results:
        label = _stock_label(r)
        detail = [
            f"DD60 {r.get('dd60_pct', 0):+.2f}%",
            f"MA25傾き {r.get('ma25_slope5_pct', 0):+.2f}%",
            f"GC乖離 {r.get('gc_gap_pct', 0):+.2f}%",
            f"出来高20日比 {r.get('volume_ratio20', 0):.2f}倍",
            f"当日 {r.get('bull_candle_pct', 0):+.2f}%",
        ]
        earnings_note = ""
        if r.get("earnings_within_10bd"):
            ed = r.get("earnings_date") or "日付不明"
            bd = r.get("earnings_business_days")
            when = f"（{bd}営業日後 / {ed}）" if bd is not None else f"（{ed}）"
            earnings_note = f"\n   ⚠️ **10営業日以内に決算あり** {when}"

        lines.append(
            f"🟡 **{label}** — " + " / ".join(detail) + earnings_note
        )

    msg = (
        f"✨ **GC強ブレイク出現 {today}**\n"
        f"深い下落後のGC2回目＋強ブレイク条件\n"
        f"検証上は年数件の希少シグナル。くいっととは別枠で観察。\n"
        f"⏱️ **出口検証の山：9〜10営業日**（固定10日が平均利益最大、9日はPF高め）\n\n"
        + "\n\n".join(lines)
        + f"\n\n📈 **GC専用ページ**\n{GC_SCREENING_VIEW_URL}"
    )
    _post_long(msg)


def notify_selling_climax(sensor):
    """暴落当日の売り尽くし先行センサー。未発動時は通知しない。"""
    if not sensor or not sensor.get("active"):
        if sensor:
            logger.info(
                "売り尽くし通知なし（待機中: TOPIX%s%% / DD20%s%% / 強反転比率%s%%）",
                sensor.get("topix_return_pct"),
                sensor.get("topix_dd20_pct"),
                sensor.get("anchor_share_pct"),
            )
        return

    signal_date = sensor.get("signal_date")
    if hasattr(signal_date, "strftime"):
        day = signal_date.strftime("%Y/%m/%d")
    else:
        day = str(signal_date or datetime.now().strftime("%Y/%m/%d"))[:10].replace("-", "/")

    def _line(r, icon):
        label = _stock_label(r)
        detail = [
            f"DD20 {r.get('dd20_pct', 0):+.2f}%",
            f"5日 {r.get('ret5_pct', 0):+.2f}%",
            f"当日 {r.get('bull_candle_pct', 0):+.2f}%",
            f"出来高20日比 {r.get('volume_ratio20', 0):.2f}倍",
        ]
        earnings_note = ""
        if r.get("earnings_within_10bd"):
            ed = r.get("earnings_date") or "日付不明"
            bd = r.get("earnings_business_days")
            when = f"（{bd}営業日後 / {ed}）" if bd is not None else f"（{ed}）"
            earnings_note = f"\n   ⚠️ **10営業日以内に決算あり** {when}"
        return f"{icon} **#{r.get('dd20_rank', '?')} {label}** — " + " / ".join(detail) + earnings_note

    main = sensor.get("main_candidates", [])
    refs = sensor.get("reference_candidates", [])
    sections = []
    if main:
        sections.append("🏆 **本命：DD20深い順 TOP3**\n" + "\n\n".join(_line(r, "🔥") for r in main))
    if refs:
        sections.append("👀 **参考：4〜5位**\n" + "\n\n".join(_line(r, "▫️") for r in refs))
    candidates_text = "\n\n".join(sections) if sections else "個別候補なし"
    cycle = sensor.get("cycle_state") or {}
    elapsed = cycle.get("business_days_since_selling")
    cycle_text = (
        f"⏳ **ピンチ確認待ち：セリクラ発動から {elapsed if elapsed is not None else 0}営業日**\n"
        if cycle.get("status") == "waiting_for_pinch"
        else ""
    )

    msg = (
        f"🔥 **売り尽くしセンサー発動 {day}**\n"
        f"TOPIX **{sensor.get('topix_return_pct', 0):+.3f}%** / "
        f"DD20 **{sensor.get('topix_dd20_pct', 0):+.3f}%**\n"
        f"強反転比率 **{sensor.get('anchor_share_pct', 0):.3f}%** "
        f"（{sensor.get('anchor_count', 0)}/{sensor.get('coarse_reversal_count', 0)}）\n"
        f"✅ 固定条件：TOPIX -2%以下 + DD20 -15%以下 + 強反転比率10%以上\n"
        f"🏆 本番選別：**DD20が深い順 TOP3**（4〜5位は参考）\n"
        f"💰 資金配分：**1銘柄最大{sensor.get('single_stock_cap_pct', 80)}%** / "
        f"**100株単位**で #1 → #2 → #3 の順に配分 / 余りは現金\n"
        f"🟢 入口：**{sensor.get('entry_rule', '翌営業日寄り')}**"
        + (f"（{sensor.get('planned_entry_date')}）\n" if sensor.get("planned_entry_date") else "\n")
        + f"🔴 出口：**{sensor.get('exit_rule', '41営業日目の寄り')}**"
        + (f"（{sensor.get('planned_exit_date')}）\n\n" if sensor.get("planned_exit_date") else "\n\n")
        + cycle_text
        + "⏳ **先行シグナルです。次はピンチをチャンスにセンサーの底打ち確認待ち。**\n\n"
        + candidates_text
        + f"\n\n📈 **専用ページ**\n{SELLING_CLIMAX_VIEW_URL}"
    )
    _post_long(msg)


def notify_pinch_to_chance(sensor):
    """暴落後の底打ち確認センサー。未発動時は通知しない。"""
    if not sensor or not sensor.get("active"):
        if sensor:
            logger.info(
                "ピンチをチャンスに通知なし（待機中: 反転率%s%% / TOPIX%s%% / 安値切上げ=%s）",
                sensor.get("reversal_rate_pct"),
                sensor.get("topix_return_pct"),
                sensor.get("topix_low_up"),
            )
        return

    signal_date = sensor.get("signal_date")
    if hasattr(signal_date, "strftime"):
        day = signal_date.strftime("%Y/%m/%d")
    else:
        day = str(signal_date or datetime.now().strftime("%Y/%m/%d"))[:10].replace("-", "/")

    def _candidate_line(r, icon):
        label = _stock_label(r)
        detail = [
            f"DD20 {r.get('dd20_pct', 0):+.2f}%",
            f"5日 {r.get('ret5_pct', 0):+.2f}%",
            f"当日 {r.get('bull_candle_pct', 0):+.2f}%",
            f"出来高20日比 {r.get('volume_ratio20', 0):.2f}倍",
        ]
        earnings_note = ""
        if r.get("earnings_within_10bd"):
            ed = r.get("earnings_date") or "日付不明"
            bd = r.get("earnings_business_days")
            when = f"（{bd}営業日後 / {ed}）" if bd is not None else f"（{ed}）"
            earnings_note = f"\n   ⚠️ **10営業日以内に決算あり** {when}"
        return f"{icon} **#{r.get('dd20_rank', '?')} {label}** — " + " / ".join(detail) + earnings_note

    main = sensor.get("main_candidates", [])
    refs = sensor.get("reference_candidates", [])
    sections = []
    if main:
        sections.append(
            "🏆 **本命：DD20深い順 TOP3**\n"
            + "\n\n".join(_candidate_line(r, "🔥") for r in main)
        )
    if refs:
        sections.append(
            "👀 **参考：4〜5位**\n"
            + "\n\n".join(_candidate_line(r, "▫️") for r in refs)
        )
    remaining = max(0, int(sensor.get("candidate_count", 0)) - len(main) - len(refs))
    if remaining:
        sections.append(f"ほか候補 **{remaining}件**（専用ページで確認）")
    candidates_text = "\n\n".join(sections) if sections else "個別候補なし"
    cycle = sensor.get("cycle_state") or {}
    if cycle.get("status") == "pinch_confirmed":
        n = cycle.get("confirmed_after_business_days")
        cycle_text = f"✅ **セリクラ先行から {n}営業日でピンチ確認**\n"
    elif cycle.get("status") == "pinch_without_selling":
        cycle_text = "ℹ️ **今回は先行セリクラなしのピンチ単独発動**\n"
    else:
        cycle_text = ""
    msg = (
        f"🚨 **ピンチをチャンスにセンサー発動 {day}**\n"
        f"市場反転率 **{sensor.get('reversal_rate_pct', 0):.3f}%** "
        f"（{sensor.get('coarse_reversal_count', 0)}/{sensor.get('universe_n', 0)}）\n"
        f"TOPIX **{sensor.get('topix_return_pct', 0):+.3f}%** / "
        f"安値切り上げ **YES**\n"
        f"✅ 固定条件：反転率1%以上 + TOPIX+3%以上 + 安値切り上げ\n"
        f"🏆 本番選別：**DD20が深い順 TOP3**（4〜5位は参考）\n"
        f"💰 資金配分：**1銘柄最大{sensor.get('single_stock_cap_pct', 65)}%** / "
        f"**100株単位**で #1 → #2 → #3 の順に配分 / 余りは現金\n"
        f"🟢 入口：**{sensor.get('entry_rule', '翌営業日寄り')}**"
        + (f"（{sensor.get('planned_entry_date')}）\n" if sensor.get("planned_entry_date") else "\n")
        + f"🔴 出口：**{sensor.get('exit_rule', '41営業日目の寄り')}**"
        + (f"（{sensor.get('planned_exit_date')}）\n\n" if sensor.get("planned_exit_date") else "\n\n")
        + cycle_text
        + f"{candidates_text}\n\n"
        f"📈 **専用ページ**\n{PINCH_SENSOR_VIEW_URL}"
    )
    _post_long(msg)

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
