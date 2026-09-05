"""有力2戦略を最大8同時保有で現実運用に近づけて比較する実験。

本番スクリーニング条件・通常バックテストは変更しない。
既存 ranking モードを実験枠として使う。

戦略A
- 入口: MA25乖離 -10〜-5% × 出来高前日比 <1.0
- 出口: がっくりB + -3%損切り + 最大30営業日

戦略B
- 入口: MA25乖離 -10〜-5% × RSI14 40〜50
- 出口: 純がっくりB（損切りなし・保有上限なし）

最大8ポジション。1銘柄は同時に1ポジションまで。
同日候補が空き枠を超える場合はコード昇順で決定する。
これは将来のexit_dateや損益を使わないため、既存のmax-position処理にあった
未来情報による並び替えを避けた決定的な比較用ルール。

資産推移は初期資金100万円を8スロットに等分（各12.5万円）し、
各スロットが決済ごとに複利で増減する簡易モデル。
"""

import json
import os

import pandas as pd

import backtest
import config
import download
from logger import get_logger
from strategies import registry

logger = get_logger(__name__)

RSI_PERIOD = 14
MAX_POSITIONS = 8
INITIAL_CAPITAL = 1_000_000
GAKKURI_MAX_HOLD_DAYS = 30


def _rsi_series(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(100).where(avg_gain.notna())


def _entry_features(g, idx):
    row = g.iloc[idx]
    prev = g.iloc[idx - 1] if idx > 0 else None

    prev_volume = prev.get("Vo") if prev is not None else None
    today_volume = row.get("Vo")
    volume_ratio = None
    if (
        prev_volume is not None and pd.notna(prev_volume) and prev_volume > 0
        and pd.notna(today_volume)
    ):
        volume_ratio = today_volume / prev_volume

    ma25 = g["MA_LONG"].iloc[idx]
    ma25_dev_pct = None
    if pd.notna(ma25) and ma25 != 0:
        ma25_dev_pct = (row["C"] / ma25 - 1) * 100

    if "RSI14" not in g.columns:
        g["RSI14"] = _rsi_series(g["C"], RSI_PERIOD)

    return {
        "volume_ratio": volume_ratio,
        "ma25_dev_pct": ma25_dev_pct,
        "rsi14": g["RSI14"].iloc[idx],
    }


def _ma5_slope(g, idx):
    if idx <= 0:
        return None
    prev = g["MA_SHORT"].iloc[idx - 1]
    today = g["MA_SHORT"].iloc[idx]
    if pd.isna(prev) or pd.isna(today):
        return None
    return today - prev


def _is_gakkuri_b(g, idx):
    if idx < 2:
        return False

    row = g.iloc[idx]
    if pd.isna(row["O"]) or pd.isna(row["C"]) or row["C"] >= row["O"]:
        return False

    s0 = _ma5_slope(g, idx)
    s1 = _ma5_slope(g, idx - 1)
    s2 = _ma5_slope(g, idx - 2)
    if s0 is None or s1 is None or s2 is None:
        return False

    ma5 = g["MA_SHORT"].iloc[idx]
    return s2 > s1 >= 0 and s0 < 0 and pd.notna(ma5) and row["C"] < ma5


def _simulate_b_trade(g, signal_idx, use_stop_loss, max_hold_days):
    entry_idx = signal_idx + 1
    if entry_idx >= len(g):
        return None

    entry_row = g.iloc[entry_idx]
    entry_price = entry_row["O"]
    if pd.isna(entry_price) or entry_price <= 0:
        return None

    stop_price = entry_price * (1 - config.BACKTEST_STOP_LOSS_PCT / 100)
    hold_end_idx = (
        len(g) - 1
        if max_hold_days is None
        else min(entry_idx + max_hold_days - 1, len(g) - 1)
    )

    for idx in range(entry_idx, hold_end_idx + 1):
        row = g.iloc[idx]
        holding_days = idx - entry_idx + 1

        if use_stop_loss:
            low = row.get("L")
            if pd.notna(low) and low <= stop_price:
                return {
                    "entry_date": entry_row["Date"],
                    "entry_price": round(entry_price, 2),
                    "exit_date": row["Date"],
                    "exit_price": round(stop_price, 2),
                    "exit_reason": "stop_loss",
                    "holding_days": holding_days,
                    "profit_pct": round((stop_price - entry_price) / entry_price * 100, 3),
                }

        if _is_gakkuri_b(g, idx):
            exit_idx = idx + 1
            if exit_idx < len(g):
                exit_row = g.iloc[exit_idx]
                exit_price = exit_row["O"]
                if pd.notna(exit_price) and exit_price > 0:
                    return {
                        "entry_date": entry_row["Date"],
                        "entry_price": round(entry_price, 2),
                        "exit_date": exit_row["Date"],
                        "exit_price": round(exit_price, 2),
                        "exit_reason": "gakkuri_b",
                        "holding_days": exit_idx - entry_idx + 1,
                        "profit_pct": round((exit_price - entry_price) / entry_price * 100, 3),
                    }

    final_row = g.iloc[hold_end_idx]
    exit_price = final_row["C"]
    if pd.isna(exit_price) or exit_price <= 0:
        return None

    return {
        "entry_date": entry_row["Date"],
        "entry_price": round(entry_price, 2),
        "exit_date": final_row["Date"],
        "exit_price": round(exit_price, 2),
        "exit_reason": "data_end_exit" if max_hold_days is None else "time_exit",
        "holding_days": hold_end_idx - entry_idx + 1,
        "profit_pct": round((exit_price - entry_price) / entry_price * 100, 3),
    }


def _build_strategy_trades(signals, price_data_by_code, kind):
    trades = []
    eligible_signals = 0

    for sig in signals:
        g = price_data_by_code.get(sig["code"])
        if g is None:
            continue

        f = _entry_features(g, sig["signal_idx"])
        in_band = pd.notna(f["ma25_dev_pct"]) and -10 <= f["ma25_dev_pct"] < -5
        if not in_band:
            continue

        if kind == "A":
            eligible = pd.notna(f["volume_ratio"]) and f["volume_ratio"] < 1.0
            use_stop_loss = True
            max_hold_days = GAKKURI_MAX_HOLD_DAYS
        elif kind == "B":
            eligible = pd.notna(f["rsi14"]) and 40 <= f["rsi14"] < 50
            use_stop_loss = False
            max_hold_days = None
        else:
            raise ValueError(kind)

        if not eligible:
            continue
        eligible_signals += 1

        result = _simulate_b_trade(
            g,
            sig["signal_idx"],
            use_stop_loss=use_stop_loss,
            max_hold_days=max_hold_days,
        )
        if result is None:
            continue

        result["code"] = str(sig["code"])
        result["signal_date"] = sig["signal_date"]
        trades.append(result)

    logger.info(f"戦略{kind}: 対象シグナル {eligible_signals}件 / トレード候補 {len(trades)}件")
    return trades


def _release_is_before_entry(trade, entry_date):
    exit_date = pd.Timestamp(trade["exit_date"])
    entry_date = pd.Timestamp(entry_date)
    if exit_date < entry_date:
        return True
    # がっくりBは翌日寄り決済なので、同日の新規寄り買いに枠を再利用できる。
    return exit_date == entry_date and trade.get("exit_reason") == "gakkuri_b"


def _portfolio_max8(candidate_trades):
    """未来の損益・exit_dateで候補順位を付けず、最大8同時保有を再現する。"""
    candidates = sorted(
        candidate_trades,
        key=lambda t: (pd.Timestamp(t["entry_date"]), str(t["code"])),
    )

    slots = [
        {"capital": INITIAL_CAPITAL / MAX_POSITIONS, "trade": None}
        for _ in range(MAX_POSITIONS)
    ]
    selected = []
    skipped_capacity = 0
    skipped_duplicate_code = 0
    max_concurrent = 0

    # 決済時にスロット資金を確定させる。
    def release_slots(entry_date):
        for slot in slots:
            tr = slot["trade"]
            if tr is None:
                continue
            if _release_is_before_entry(tr, entry_date):
                slot["capital"] *= 1 + tr["profit_pct"] / 100
                slot["trade"] = None

    for tr in candidates:
        entry_date = pd.Timestamp(tr["entry_date"])
        release_slots(entry_date)

        active_codes = {
            str(slot["trade"]["code"])
            for slot in slots
            if slot["trade"] is not None
        }
        if str(tr["code"]) in active_codes:
            skipped_duplicate_code += 1
            continue

        free_slot = next((slot for slot in slots if slot["trade"] is None), None)
        if free_slot is None:
            skipped_capacity += 1
            continue

        tr_copy = dict(tr)
        tr_copy["slot_entry_capital"] = round(free_slot["capital"], 2)
        free_slot["trade"] = tr_copy
        selected.append(tr_copy)
        concurrent = sum(slot["trade"] is not None for slot in slots)
        max_concurrent = max(max_concurrent, concurrent)

    # 最後に残っているポジションも各トレードの既定exitで決済して資金確定。
    for slot in slots:
        tr = slot["trade"]
        if tr is not None:
            slot["capital"] *= 1 + tr["profit_pct"] / 100
            slot["trade"] = None

    ending_capital = sum(slot["capital"] for slot in slots)
    summary = backtest.summarize_trades(selected)
    summary.update({
        "candidate_trades": len(candidate_trades),
        "actual_entries": len(selected),
        "skipped_capacity": skipped_capacity,
        "skipped_duplicate_code": skipped_duplicate_code,
        "max_concurrent_positions": max_concurrent,
        "initial_capital": INITIAL_CAPITAL,
        "ending_capital": round(ending_capital, 0),
        "portfolio_return_pct": round((ending_capital / INITIAL_CAPITAL - 1) * 100, 2),
    })
    return selected, summary


def _yearly_records(trades):
    yearly = backtest.build_yearly_summary(trades)
    return yearly.to_dict("records") if not yearly.empty else []


def _log_result(name, summary, yearly):
    logger.info("\n" + "=" * 108)
    logger.info(name)
    logger.info("=" * 108)
    logger.info(
        f"候補 {summary['candidate_trades']}件 → 実エントリー {summary['actual_entries']}件 / "
        f"容量見送り {summary['skipped_capacity']}件 / 同一銘柄重複見送り {summary['skipped_duplicate_code']}件"
    )
    logger.info(
        f"勝率 {summary.get('win_rate')}% / PF {summary.get('profit_factor')} / "
        f"平均勝ち {summary.get('avg_profit_pct')}% / 平均負け {summary.get('avg_loss_pct')}% / "
        f"平均保有 {summary.get('avg_holding_days')}日"
    )
    logger.info(
        f"最大同時保有 {summary['max_concurrent_positions']} / "
        f"資産 {summary['initial_capital']:,.0f}円 → {summary['ending_capital']:,.0f}円 "
        f"({summary['portfolio_return_pct']:+.2f}%)"
    )
    if yearly:
        logger.info("年別: " + " / ".join(
            f"{int(r['year'])}: {int(r['total_trades'])}件 PF{r['profit_factor']} 勝率{r['win_rate']}%"
            for r in yearly
        ))


def main():
    logger.info("=== 有力2戦略 最大8ポジション現実運用バックテスト開始 ===")
    logger.info("未来情報で候補順位を付けず、同日候補はコード昇順で処理します。")

    strategy = registry.get_strategy("ma5_breakout")
    target_codes = download.get_target_codes()
    cache_filename = f"backtest_prices_{config.TARGET_MARKET}_{config.BACKTEST_YEARS}y.csv"
    price_df = download.get_price_history_incremental(
        cache_filename=cache_filename,
        years=config.BACKTEST_YEARS,
    )
    signals, price_data_by_code = strategy(price_df, target_codes)

    candidate_a = _build_strategy_trades(signals, price_data_by_code, "A")
    candidate_b = _build_strategy_trades(signals, price_data_by_code, "B")

    selected_a, summary_a = _portfolio_max8(candidate_a)
    selected_b, summary_b = _portfolio_max8(candidate_b)
    yearly_a = _yearly_records(selected_a)
    yearly_b = _yearly_records(selected_b)

    _log_result(
        "戦略A: MA25 -10〜-5% × 出来高<1.0x → がっくりB + -3%SL + 30日",
        summary_a,
        yearly_a,
    )
    _log_result(
        "戦略B: MA25 -10〜-5% × RSI40-50 → 純がっくりB",
        summary_b,
        yearly_b,
    )

    os.makedirs("output", exist_ok=True)
    output = {
        "settings": {
            "max_positions": MAX_POSITIONS,
            "initial_capital": INITIAL_CAPITAL,
            "position_model": "8 independent equal-capital slots, compounded per slot",
            "same_day_candidate_order": "code ascending; no future outcome/exit-date ranking",
            "same_code_rule": "one simultaneous position per code",
            "same_day_reuse": "gakkuri next-open exits free a slot for same-day next-open entries",
        },
        "strategy_A": {"summary": summary_a, "yearly": yearly_a},
        "strategy_B": {"summary": summary_b, "yearly": yearly_b},
    }
    with open("output/max8_two_strategy_comparison.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    pd.DataFrame(selected_a).to_csv(
        "output/max8_strategy_A_trades.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(selected_b).to_csv(
        "output/max8_strategy_B_trades.csv", index=False, encoding="utf-8-sig"
    )

    logger.info("\n結果を output/max8_two_strategy_comparison.json に保存しました。")
    logger.info("=== 有力2戦略 最大8ポジション現実運用バックテスト完了 ===")


if __name__ == "__main__":
    main()
