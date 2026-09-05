"""有力2戦略で「同日候補のどれを最大8枠に入れるか」を単独指標ごとに比較する実験。

本番スクリーニング条件・通常バックテストは変更しない。
既存 ranking モードを実験枠として使う。

戦略A
- 入口: MA25乖離 -10〜-5% × 出来高前日比 <1.0
- 出口: がっくりB + -3%損切り + 最大30営業日

戦略B
- 入口: MA25乖離 -10〜-5% × RSI14 40〜50
- 出口: 純がっくりB（損切りなし・保有上限なし）

最大8ポジション。1銘柄は同時に1ポジションまで。
候補順位はシグナル当日の情報だけを使い、将来のexit_dateや損益は一切使わない。

比較する同日優先順位:
- code_asc: 銘柄コード昇順（前回の基準）
- ma25_deeper: MA25乖離がよりマイナス（-10%側）
- ma25_shallower: MA25乖離が浅い（-5%側）
- volume_lower: 出来高倍率が低い
- rsi_lower: RSIが低い
- rsi_near45: RSIが45に近い
- gain_lower: 当日上昇率が低い
- ma5_rise_lower: MA5上昇率が低い
- close_near80: 終値位置が80%に近い

資産推移は初期100万円を8スロットに等分し、各スロットを決済ごとに複利。
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

RANKING_MODES = [
    ("code_asc", "コード昇順（基準）"),
    ("ma25_deeper", "MA25乖離が深い順（-10%側）"),
    ("ma25_shallower", "MA25乖離が浅い順（-5%側）"),
    ("volume_lower", "出来高倍率が低い順"),
    ("rsi_lower", "RSIが低い順"),
    ("rsi_near45", "RSIが45に近い順"),
    ("gain_lower", "当日上昇率が低い順"),
    ("ma5_rise_lower", "MA5上昇率が低い順"),
    ("close_near80", "終値位置が80%に近い順"),
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
        pd.notna(row["H"]) and pd.notna(row["L"]) and pd.notna(row["C"])
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
        # 優先順位はこのシグナル当日の特徴だけを使う。
        result.update(f)
        trades.append(result)

    logger.info(f"戦略{kind}: 対象シグナル {eligible_signals}件 / トレード候補 {len(trades)}件")
    return trades


def _safe_num(value, fallback=float("inf")):
    if value is None or pd.isna(value):
        return fallback
    return float(value)


def _rank_key(trade, mode):
    code = str(trade["code"])
    if mode == "code_asc":
        return (code,)
    if mode == "ma25_deeper":
        return (_safe_num(trade.get("ma25_dev_pct")), code)
    if mode == "ma25_shallower":
        return (-_safe_num(trade.get("ma25_dev_pct"), fallback=-float("inf")), code)
    if mode == "volume_lower":
        return (_safe_num(trade.get("volume_ratio")), code)
    if mode == "rsi_lower":
        return (_safe_num(trade.get("rsi14")), code)
    if mode == "rsi_near45":
        rsi = _safe_num(trade.get("rsi14"))
        return (abs(rsi - 45), code)
    if mode == "gain_lower":
        return (_safe_num(trade.get("gain_pct")), code)
    if mode == "ma5_rise_lower":
        return (_safe_num(trade.get("ma5_rise_pct")), code)
    if mode == "close_near80":
        pos = _safe_num(trade.get("close_position_pct"))
        return (abs(pos - 80), code)
    raise ValueError(mode)


def _release_is_before_entry(trade, entry_date):
    exit_date = pd.Timestamp(trade["exit_date"])
    entry_date = pd.Timestamp(entry_date)
    if exit_date < entry_date:
        return True
    # がっくりBは翌日寄り決済なので、同日の新規寄り買いに枠を再利用できる。
    return exit_date == entry_date and trade.get("exit_reason") == "gakkuri_b"


def _portfolio_max8(candidate_trades, ranking_mode):
    """同日候補だけを指定指標で順位付けし、最大8同時保有を再現する。"""
    by_date = {}
    for tr in candidate_trades:
        d = pd.Timestamp(tr["entry_date"])
        by_date.setdefault(d, []).append(tr)

    slots = [
        {"capital": INITIAL_CAPITAL / MAX_POSITIONS, "trade": None}
        for _ in range(MAX_POSITIONS)
    ]
    selected = []
    skipped_capacity = 0
    skipped_duplicate_code = 0
    max_concurrent = 0

    def release_slots(entry_date):
        for slot in slots:
            tr = slot["trade"]
            if tr is None:
                continue
            if _release_is_before_entry(tr, entry_date):
                slot["capital"] *= 1 + tr["profit_pct"] / 100
                slot["trade"] = None

    for entry_date in sorted(by_date):
        release_slots(entry_date)
        day_candidates = sorted(by_date[entry_date], key=lambda t: _rank_key(t, ranking_mode))

        for tr in day_candidates:
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
            tr_copy["ranking_mode"] = ranking_mode
            free_slot["trade"] = tr_copy
            selected.append(tr_copy)
            concurrent = sum(slot["trade"] is not None for slot in slots)
            max_concurrent = max(max_concurrent, concurrent)

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
        f"{label}: entries {summary['actual_entries']} / PF {summary.get('profit_factor')} / "
        f"勝率 {summary.get('win_rate')}% / 資産 {summary['ending_capital']:,.0f}円 "
        f"({summary['portfolio_return_pct']:+.2f}%) / 2025-26 PF {recent.get('profit_factor')} "
        f"勝率 {recent.get('win_rate')}%"
    )


def _run_rankings(strategy_name, candidate_trades):
    logger.info("\n" + "=" * 116)
    logger.info(strategy_name)
    logger.info("=" * 116)

    results = {}
    rows_for_ranking = []

    for mode, label in RANKING_MODES:
        selected, summary = _portfolio_max8(candidate_trades, mode)
        yearly = _yearly_records(selected)
        recent = _recent_summary(selected)
        results[mode] = {
            "label": label,
            "summary": summary,
            "recent_2025_2026": recent,
            "yearly": yearly,
        }
        rows_for_ranking.append((mode, label, summary, recent))
        _log_mode(label, summary, recent)

        pd.DataFrame(selected).to_csv(
            f"output/max8_rank_{strategy_name.split(':')[0]}_{mode}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    logger.info("\n--- 全期間PF順 ---")
    for i, (mode, label, summary, recent) in enumerate(
        sorted(rows_for_ranking, key=lambda x: _pf_num(x[2]), reverse=True), 1
    ):
        logger.info(
            f"{i:02d}. {label}: PF {summary.get('profit_factor')} / "
            f"資産 {summary['ending_capital']:,.0f}円 / 2025-26 PF {recent.get('profit_factor')}"
        )

    logger.info("\n--- 2025-26 PF順 ---")
    for i, (mode, label, summary, recent) in enumerate(
        sorted(rows_for_ranking, key=lambda x: _pf_num(x[3]), reverse=True), 1
    ):
        logger.info(
            f"{i:02d}. {label}: 2025-26 PF {recent.get('profit_factor')} / "
            f"全期間PF {summary.get('profit_factor')} / 資産 {summary['ending_capital']:,.0f}円"
        )

    return results


def main():
    logger.info("=== 最大8枠 同日候補ランキング検証開始 ===")
    logger.info("候補順位はシグナル当日の情報だけを使用。未来の決済日・損益は使いません。")

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

    results_a = _run_rankings(
        "A: MA25 -10〜-5% × 出来高<1.0x → がっくりB + -3%SL + 30日",
        candidate_a,
    )
    results_b = _run_rankings(
        "B: MA25 -10〜-5% × RSI40-50 → 純がっくりB",
        candidate_b,
    )

    output = {
        "settings": {
            "max_positions": MAX_POSITIONS,
            "initial_capital": INITIAL_CAPITAL,
            "ranking_uses_future_information": False,
            "recent_start_year": RECENT_START_YEAR,
            "ranking_modes": [m for m, _ in RANKING_MODES],
        },
        "strategy_A": results_a,
        "strategy_B": results_b,
    }
    with open("output/max8_ranking_comparison.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    logger.info("\n結果を output/max8_ranking_comparison.json に保存しました。")
    logger.info("=== 最大8枠 同日候補ランキング検証完了 ===")


if __name__ == "__main__":
    main()
