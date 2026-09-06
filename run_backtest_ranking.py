"""戦略A/Bを同じ最大8枠で運用する、本番寄りの共有ポートフォリオ実験。

本番スクリーニング条件・通常バックテストは変更しない。
候補順位・切り捨て判定はシグナル当日の情報だけを使い、未来の決済日や損益は使わない。

戦略A
- 入口: MA25乖離 -10〜-5% × 出来高前日比 <1.0
- 出口: がっくりB + -3%損切り + 最大30営業日
- 同日順位: MA25乖離が浅い（-5%側）ほど上
- 下位20%を除外

戦略B
- 入口: MA25乖離 -10〜-5% × RSI14 40〜50
- 出口: 純がっくりB（損切りなし・保有上限なし）
- 同日順位: MA25乖離が深い（-10%側）ほど上
- 下位50%を除外

A/Bは別々に8枠を持たず、合計8枠を共有する。
同じ銘柄が同日にA/B両方へ入った場合は、優先ルールに従って片方だけ採用する。
比較する共有順位:
- A優先
- B優先
- A/Bそれぞれの同日順位を0〜1へ正規化して混ぜるバランス順位
"""

import json
import math
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
RECENT_START_YEAR = 2025
A_CUT_PCT = 20
B_CUT_PCT = 50

MERGE_MODES = [
    ("a_first", "A優先 → B"),
    ("b_first", "B優先 → A"),
    ("balanced_rank", "A/B同日順位を正規化して混合"),
]


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

    volume_ratio = None
    if prev is not None:
        prev_volume = prev.get("Vo")
        today_volume = row.get("Vo")
        if (
            prev_volume is not None
            and pd.notna(prev_volume)
            and prev_volume > 0
            and pd.notna(today_volume)
        ):
            volume_ratio = today_volume / prev_volume

    ma25 = g["MA_LONG"].iloc[idx]
    ma25_dev_pct = None
    if pd.notna(ma25) and ma25 != 0:
        ma25_dev_pct = (row["C"] / ma25 - 1) * 100

    if "RSI14" not in g.columns:
        g["RSI14"] = _rsi_series(g["C"], RSI_PERIOD)
    rsi14 = g["RSI14"].iloc[idx]

    gain_pct = None
    if pd.notna(row["O"]) and row["O"] > 0 and pd.notna(row["C"]):
        gain_pct = (row["C"] / row["O"] - 1) * 100

    ma5_rise_pct = None
    if idx > 0:
        ma5_prev = g["MA_SHORT"].iloc[idx - 1]
        ma5_today = g["MA_SHORT"].iloc[idx]
        if pd.notna(ma5_prev) and ma5_prev != 0 and pd.notna(ma5_today):
            ma5_rise_pct = (ma5_today / ma5_prev - 1) * 100

    close_position_pct = None
    if (
        pd.notna(row["H"])
        and pd.notna(row["L"])
        and pd.notna(row["C"])
        and row["H"] > row["L"]
    ):
        close_position_pct = (row["C"] - row["L"]) / (row["H"] - row["L"]) * 100

    return {
        "volume_ratio": volume_ratio,
        "ma25_dev_pct": ma25_dev_pct,
        "rsi14": rsi14,
        "gain_pct": gain_pct,
        "ma5_rise_pct": ma5_rise_pct,
        "close_position_pct": close_position_pct,
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


def _simulate_trade(g, signal_idx, kind):
    entry_idx = signal_idx + 1
    if entry_idx >= len(g):
        return None

    entry_row = g.iloc[entry_idx]
    entry_price = entry_row["O"]
    if pd.isna(entry_price) or entry_price <= 0:
        return None

    use_stop_loss = kind == "A"
    max_hold_days = GAKKURI_MAX_HOLD_DAYS if kind == "A" else None
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
        if not (pd.notna(f["ma25_dev_pct"]) and -10 <= f["ma25_dev_pct"] < -5):
            continue

        if kind == "A":
            eligible = pd.notna(f["volume_ratio"]) and f["volume_ratio"] < 1.0
        elif kind == "B":
            eligible = pd.notna(f["rsi14"]) and 40 <= f["rsi14"] < 50
        else:
            raise ValueError(kind)

        if not eligible:
            continue
        eligible_signals += 1

        result = _simulate_trade(g, sig["signal_idx"], kind)
        if result is None:
            continue

        result["code"] = str(sig["code"])
        result["signal_date"] = sig["signal_date"]
        result["strategy"] = kind
        result.update(f)
        trades.append(result)

    logger.info(f"戦略{kind}: 対象シグナル {eligible_signals}件 / トレード候補 {len(trades)}件")
    return trades


def _safe_num(value, fallback=float("inf")):
    if value is None or pd.isna(value):
        return fallback
    return float(value)


def _strategy_rank_key(trade, kind):
    code = str(trade["code"])
    ma25 = _safe_num(trade.get("ma25_dev_pct"))
    if kind == "A":
        return (-ma25, code)
    if kind == "B":
        return (ma25, code)
    raise ValueError(kind)


def _cut_day_candidates(day_candidates, kind, cut_pct):
    ranked = sorted(day_candidates, key=lambda t: _strategy_rank_key(t, kind))
    if not ranked:
        return []

    keep_n = max(1, math.ceil(len(ranked) * (1 - cut_pct / 100)))
    kept = ranked[:keep_n]
    denom = max(1, len(kept) - 1)
    result = []
    for rank, tr in enumerate(kept):
        tr_copy = dict(tr)
        tr_copy["within_strategy_rank"] = rank + 1
        tr_copy["within_strategy_count"] = len(kept)
        tr_copy["normalized_rank"] = rank / denom if len(kept) > 1 else 0.0
        tr_copy["quality_cut_pct"] = cut_pct
        result.append(tr_copy)
    return result


def _merge_day_candidates(a_day, b_day, mode):
    a_kept = _cut_day_candidates(a_day, "A", A_CUT_PCT)
    b_kept = _cut_day_candidates(b_day, "B", B_CUT_PCT)

    if mode == "a_first":
        combined = sorted(a_kept, key=lambda t: (t["within_strategy_rank"], str(t["code"])))
        combined += sorted(b_kept, key=lambda t: (t["within_strategy_rank"], str(t["code"])))
    elif mode == "b_first":
        combined = sorted(b_kept, key=lambda t: (t["within_strategy_rank"], str(t["code"])))
        combined += sorted(a_kept, key=lambda t: (t["within_strategy_rank"], str(t["code"])))
    elif mode == "balanced_rank":
        combined = sorted(
            a_kept + b_kept,
            key=lambda t: (
                t["normalized_rank"],
                0 if t["strategy"] == "A" else 1,
                str(t["code"]),
            ),
        )
    else:
        raise ValueError(mode)

    # 同じ銘柄が同日にA/B両方へ入った場合、共有順位で先に来た方だけ残す。
    deduped = []
    seen_codes = set()
    duplicate_same_day = 0
    for tr in combined:
        code = str(tr["code"])
        if code in seen_codes:
            duplicate_same_day += 1
            continue
        seen_codes.add(code)
        deduped.append(tr)

    return deduped, len(a_kept), len(b_kept), duplicate_same_day


def _release_is_before_entry(trade, entry_date):
    exit_date = pd.Timestamp(trade["exit_date"])
    entry_date = pd.Timestamp(entry_date)
    if exit_date < entry_date:
        return True
    return exit_date == entry_date and trade.get("exit_reason") == "gakkuri_b"


def _portfolio_shared_max8(candidate_a, candidate_b, mode):
    a_by_date = {}
    b_by_date = {}
    all_dates = set()

    for tr in candidate_a:
        d = pd.Timestamp(tr["entry_date"])
        a_by_date.setdefault(d, []).append(tr)
        all_dates.add(d)
    for tr in candidate_b:
        d = pd.Timestamp(tr["entry_date"])
        b_by_date.setdefault(d, []).append(tr)
        all_dates.add(d)

    slots = [
        {"capital": INITIAL_CAPITAL / MAX_POSITIONS, "trade": None}
        for _ in range(MAX_POSITIONS)
    ]

    selected = []
    skipped_capacity = 0
    skipped_active_duplicate = 0
    duplicate_same_day = 0
    kept_after_cut_a = 0
    kept_after_cut_b = 0
    max_concurrent = 0

    def release_slots(entry_date):
        for slot in slots:
            tr = slot["trade"]
            if tr is not None and _release_is_before_entry(tr, entry_date):
                slot["capital"] *= 1 + tr["profit_pct"] / 100
                slot["trade"] = None

    for entry_date in sorted(all_dates):
        release_slots(entry_date)
        day_candidates, kept_a, kept_b, dup_day = _merge_day_candidates(
            a_by_date.get(entry_date, []),
            b_by_date.get(entry_date, []),
            mode,
        )
        kept_after_cut_a += kept_a
        kept_after_cut_b += kept_b
        duplicate_same_day += dup_day

        for tr in day_candidates:
            active_codes = {
                str(slot["trade"]["code"])
                for slot in slots
                if slot["trade"] is not None
            }
            if str(tr["code"]) in active_codes:
                skipped_active_duplicate += 1
                continue

            free_slot = next((slot for slot in slots if slot["trade"] is None), None)
            if free_slot is None:
                skipped_capacity += 1
                continue

            tr_copy = dict(tr)
            tr_copy["slot_entry_capital"] = round(free_slot["capital"], 2)
            tr_copy["merge_mode"] = mode
            free_slot["trade"] = tr_copy
            selected.append(tr_copy)
            max_concurrent = max(
                max_concurrent,
                sum(slot["trade"] is not None for slot in slots),
            )

    for slot in slots:
        tr = slot["trade"]
        if tr is not None:
            slot["capital"] *= 1 + tr["profit_pct"] / 100
            slot["trade"] = None

    ending_capital = sum(slot["capital"] for slot in slots)
    summary = backtest.summarize_trades(selected)
    a_entries = sum(1 for tr in selected if tr.get("strategy") == "A")
    b_entries = sum(1 for tr in selected if tr.get("strategy") == "B")
    summary.update({
        "candidate_trades_A": len(candidate_a),
        "candidate_trades_B": len(candidate_b),
        "kept_after_cut_A": kept_after_cut_a,
        "kept_after_cut_B": kept_after_cut_b,
        "actual_entries": len(selected),
        "entries_A": a_entries,
        "entries_B": b_entries,
        "skipped_capacity": skipped_capacity,
        "skipped_active_duplicate": skipped_active_duplicate,
        "duplicate_same_day_A_B": duplicate_same_day,
        "max_concurrent_positions": max_concurrent,
        "initial_capital": INITIAL_CAPITAL,
        "ending_capital": round(ending_capital, 0),
        "portfolio_return_pct": round((ending_capital / INITIAL_CAPITAL - 1) * 100, 2),
    })
    return selected, summary


def _yearly_records(trades):
    yearly = backtest.build_yearly_summary(trades)
    return yearly.to_dict("records") if not yearly.empty else []


def _recent_summary(trades):
    recent = [t for t in trades if pd.Timestamp(t["entry_date"]).year >= RECENT_START_YEAR]
    return backtest.summarize_trades(recent)


def _pf_num(summary):
    pf = summary.get("profit_factor")
    if isinstance(pf, (int, float)) and math.isfinite(float(pf)):
        return float(pf)
    return -1.0


def _log_mode(label, summary, recent):
    logger.info(
        f"{label}: entries {summary['actual_entries']} (A {summary['entries_A']} / B {summary['entries_B']}) / "
        f"PF {summary.get('profit_factor')} / 勝率 {summary.get('win_rate')}% / "
        f"資産 {summary['ending_capital']:,.0f}円 ({summary['portfolio_return_pct']:+.2f}%) / "
        f"容量見送り {summary['skipped_capacity']} / 同日AB重複 {summary['duplicate_same_day_A_B']} / "
        f"2025-26 PF {recent.get('profit_factor')} 勝率 {recent.get('win_rate')}%"
    )


def main():
    logger.info("=== A+B共有 最大8枠・本番寄りバックテスト開始 ===")
    logger.info(
        "Aは下位20%、Bは下位50%を同日ごとに除外後、同じ8枠を共有します。未来情報は順位付けに使いません。"
    )

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

    os.makedirs("output", exist_ok=True)

    results = {}
    ranking_rows = []
    for mode, label in MERGE_MODES:
        selected, summary = _portfolio_shared_max8(candidate_a, candidate_b, mode)
        recent = _recent_summary(selected)
        yearly = _yearly_records(selected)
        results[mode] = {
            "label": label,
            "summary": summary,
            "recent_2025_2026": recent,
            "yearly": yearly,
        }
        ranking_rows.append((mode, label, summary, recent))
        _log_mode(label, summary, recent)

        pd.DataFrame(selected).to_csv(
            f"output/max8_shared_ab_{mode}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    logger.info("\n--- 全期間PF順 ---")
    for i, (_, label, summary, recent) in enumerate(
        sorted(ranking_rows, key=lambda x: _pf_num(x[2]), reverse=True), 1
    ):
        logger.info(
            f"{i:02d}. {label}: PF {summary.get('profit_factor')} / "
            f"資産 {summary['ending_capital']:,.0f}円 / entries {summary['actual_entries']} "
            f"(A {summary['entries_A']} / B {summary['entries_B']}) / 2025-26 PF {recent.get('profit_factor')}"
        )

    logger.info("\n--- 2025-26 PF順 ---")
    for i, (_, label, summary, recent) in enumerate(
        sorted(ranking_rows, key=lambda x: _pf_num(x[3]), reverse=True), 1
    ):
        logger.info(
            f"{i:02d}. {label}: 2025-26 PF {recent.get('profit_factor')} / "
            f"全期間PF {summary.get('profit_factor')} / 資産 {summary['ending_capital']:,.0f}円"
        )

    output = {
        "settings": {
            "max_positions_shared": MAX_POSITIONS,
            "initial_capital": INITIAL_CAPITAL,
            "ranking_uses_future_information": False,
            "recent_start_year": RECENT_START_YEAR,
            "strategy_A_cut_pct": A_CUT_PCT,
            "strategy_B_cut_pct": B_CUT_PCT,
            "strategy_A_rank": "MA25 shallow (-5% side) first",
            "strategy_B_rank": "MA25 deep (-10% side) first",
            "merge_modes": [m for m, _ in MERGE_MODES],
        },
        "results": results,
    }
    with open("output/max8_shared_ab_comparison.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    logger.info("\n結果を output/max8_shared_ab_comparison.json に保存しました。")
    logger.info("=== A+B共有 最大8枠・本番寄りバックテスト完了 ===")


if __name__ == "__main__":
    main()
