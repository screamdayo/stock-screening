"""正規化混合ポートフォリオを、本番寄りにリスク分析する。

run_backtest_ranking.py が先に出力する固定8枠の正規化混合を日次終値で時価評価し、
東証プライム市場内部と主要DD局面を確認する。

さらに同じA/B条件・同じ正規化混合順位を固定したまま、前営業日の
「25日線上にいるプライム銘柄比率」だけで新規保有上限を 8 / 4 / 2 枠へ変える。

可変ルール:
- 前営業日の25日線上比率 40%以上: 最大8枠
- 20%以上40%未満: 最大4枠
- 20%未満: 最大2枠

重要:
- 当日の終値は寄り付き時点で未知なので、必ず前営業日の市場内部だけを使う。
- 枠数が下がっても既存ポジションは強制決済しない。
  現在保有数が上限以上なら、新規エントリーだけ止める。
- 本番スクリーニング条件は変更しない。分析専用。
"""

import json
import os

import pandas as pd

import backtest
import config
from logger import get_logger
from strategies import registry
import run_backtest_ranking as ranking

logger = get_logger(__name__)

TRADES_CSV = "output/max8_shared_ab_balanced_rank.csv"
EQUITY_CSV = "output/max8_shared_ab_balanced_mtm_equity.csv"
RISK_JSON = "output/max8_shared_ab_balanced_mtm_risk.json"
MARKET_BREADTH_CSV = "output/max8_shared_ab_market_breadth.csv"
MARKET_REGIME_JSON = "output/max8_shared_ab_market_regime.json"
VARIABLE_TRADES_CSV = "output/max8_shared_ab_balanced_variable_capacity.csv"
VARIABLE_EQUITY_CSV = "output/max8_shared_ab_balanced_variable_capacity_mtm_equity.csv"
VARIABLE_SUMMARY_JSON = "output/max8_shared_ab_balanced_variable_capacity_summary.json"

MAX_POSITIONS = 8
INITIAL_CAPITAL = 1_000_000
RECENT_START_YEAR = 2025

REGIME_WINDOWS = {
    "2022_realized_dd": {
        "label": "2022 決済ベースDD",
        "start": "2022-08-30",
        "trough": "2023-01-06",
        "end": "2023-01-06",
    },
    "2024_mtm_dd": {
        "label": "2024 時価評価DD",
        "start": "2024-06-14",
        "trough": "2024-08-05",
        "end": "2024-08-20",
    },
}


def _normalize_code(value):
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if len(text) == 5 and text.endswith("0"):
        text = text[:-1]
    return text


def _load_inputs():
    if not os.path.exists(TRADES_CSV):
        raise FileNotFoundError(
            f"{TRADES_CSV} がありません。先に run_backtest_ranking.py を実行してください。"
        )

    trades = pd.read_csv(
        TRADES_CSV,
        parse_dates=["entry_date", "exit_date"],
        dtype={"code": str},
    )
    if trades.empty:
        raise ValueError("正規化混合の採用トレードが0件です。")

    required = {
        "code", "slot_id", "slot_entry_capital", "entry_date", "entry_price",
        "exit_date", "profit_pct",
    }
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"採用トレードCSVに必要列がありません: {sorted(missing)}")

    trades["code"] = trades["code"].map(_normalize_code)
    trades["slot_id"] = trades["slot_id"].astype(int)
    trades["slot_entry_capital"] = pd.to_numeric(trades["slot_entry_capital"], errors="raise")
    trades["entry_price"] = pd.to_numeric(trades["entry_price"], errors="raise")
    trades["profit_pct"] = pd.to_numeric(trades["profit_pct"], errors="raise")
    trades = trades.sort_values(["slot_id", "entry_date", "exit_date", "code"]).reset_index(drop=True)

    cache_filename = f"backtest_prices_{config.TARGET_MARKET}_{config.BACKTEST_YEARS}y.csv"
    cache_path = os.path.join(config.DATA_DIR, cache_filename)
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"株価キャッシュがありません: {cache_path}")

    prices = pd.read_csv(cache_path, parse_dates=["Date"], dtype={"Code": str})
    prices["Code"] = prices["Code"].map(_normalize_code)
    for col in ["O", "H", "L", "C"]:
        if col in prices.columns:
            prices[col] = pd.to_numeric(prices[col], errors="coerce")
    prices = prices.dropna(subset=["Date", "Code", "C"]).sort_values(["Code", "Date"])
    return trades, prices


def _build_close_panel(trades, prices):
    codes = set(trades["code"].unique())
    start_date = trades["entry_date"].min()
    end_date = trades["exit_date"].max()
    filtered = prices[
        prices["Code"].isin(codes)
        & (prices["Date"] >= start_date)
        & (prices["Date"] <= end_date)
    ][["Date", "Code", "C"]].copy()
    if filtered.empty:
        raise ValueError("採用銘柄の株価データがキャッシュ内にありません。")
    trading_dates = pd.DatetimeIndex(sorted(filtered["Date"].unique()))
    panel = filtered.pivot_table(index="Date", columns="Code", values="C", aggfunc="last")
    panel = panel.reindex(trading_dates).sort_index().ffill()
    return trading_dates, panel


def _slot_value_on_date(slot_trades, date, close_panel):
    initial_slot_capital = INITIAL_CAPITAL / MAX_POSITIONS
    past = slot_trades[slot_trades["entry_date"] <= date]
    if past.empty:
        return initial_slot_capital, None

    latest_entry_date = past["entry_date"].max()
    tr = past[past["entry_date"] == latest_entry_date].iloc[-1]
    exit_date = pd.Timestamp(tr["exit_date"])
    entry_capital = float(tr["slot_entry_capital"])

    if date >= exit_date:
        realized = entry_capital * (1 + float(tr["profit_pct"]) / 100)
        return realized, None

    code = str(tr["code"])
    entry_price = float(tr["entry_price"])
    if code not in close_panel.columns:
        return entry_capital, code

    close_price = close_panel.at[date, code] if date in close_panel.index else float("nan")
    if pd.isna(close_price):
        history = close_panel.loc[:date, code].dropna()
        if history.empty:
            return entry_capital, code
        close_price = float(history.iloc[-1])

    return entry_capital * (float(close_price) / entry_price), code


def _build_daily_equity(trades, trading_dates, close_panel):
    by_slot = {
        slot_id: trades[trades["slot_id"] == slot_id].copy()
        for slot_id in range(1, MAX_POSITIONS + 1)
    }
    rows = []
    for date in trading_dates:
        total = 0.0
        active_positions = 0
        row = {"date": date}
        for slot_id in range(1, MAX_POSITIONS + 1):
            value, active_code = _slot_value_on_date(by_slot[slot_id], date, close_panel)
            row[f"slot_{slot_id}_value"] = round(value, 2)
            row[f"slot_{slot_id}_code"] = active_code
            total += value
            if active_code is not None:
                active_positions += 1
        row["active_positions"] = active_positions
        row["equity"] = round(total, 2)
        rows.append(row)

    curve = pd.DataFrame(rows)
    curve["peak_equity"] = curve["equity"].cummax()
    curve["drawdown_pct"] = (curve["equity"] / curve["peak_equity"] - 1) * 100
    return curve


def _risk_summary(curve):
    trough_idx = curve["drawdown_pct"].idxmin()
    trough = curve.loc[trough_idx]
    trough_date = pd.Timestamp(trough["date"])
    peak_equity = float(trough["peak_equity"])
    before = curve.loc[:trough_idx]
    peak_row = before[before["equity"] == peak_equity].iloc[-1]
    peak_date = pd.Timestamp(peak_row["date"])

    after = curve[curve["date"] > trough_date]
    recovered = after[after["equity"] >= peak_equity]
    if recovered.empty:
        recovery_date = None
        recovery_trading_days = None
        end_idx = len(curve) - 1
    else:
        recovery_row = recovered.iloc[0]
        recovery_date = pd.Timestamp(recovery_row["date"])
        end_idx = recovered.index[0]
        recovery_trading_days = int(end_idx - peak_row.name)

    final_equity = float(curve.iloc[-1]["equity"])
    min_equity_row = curve.loc[curve["equity"].idxmin()]
    return {
        "basis": "daily_close_mark_to_market",
        "max_drawdown_pct": round(float(trough["drawdown_pct"]), 2),
        "peak_equity": round(peak_equity, 0),
        "peak_date": str(peak_date.date()),
        "trough_equity": round(float(trough["equity"]), 0),
        "trough_date": str(trough_date.date()),
        "peak_to_trough_trading_days": int(trough_idx - peak_row.name),
        "recovery_date": str(recovery_date.date()) if recovery_date is not None else None,
        "recovery_trading_days_from_peak": recovery_trading_days,
        "underwater_trading_days": int(end_idx - peak_row.name),
        "final_equity": round(final_equity, 0),
        "portfolio_return_pct": round((final_equity / INITIAL_CAPITAL - 1) * 100, 2),
        "minimum_equity": round(float(min_equity_row["equity"]), 0),
        "minimum_equity_date": str(pd.Timestamp(min_equity_row["date"]).date()),
        "max_active_positions": int(curve["active_positions"].max()),
    }


def _build_market_breadth(prices):
    p = prices[["Date", "Code", "C"]].copy().sort_values(["Code", "Date"])
    p["ret_pct"] = p.groupby("Code")["C"].pct_change() * 100
    p["ma25"] = p.groupby("Code")["C"].transform(lambda s: s.rolling(25, min_periods=25).mean())
    p["above_ma25"] = p["C"] >= p["ma25"]

    rows = []
    for date, g in p.groupby("Date"):
        valid_ret = g["ret_pct"].dropna()
        valid_ma = g.dropna(subset=["ma25"])
        if valid_ret.empty:
            continue
        rows.append({
            "date": pd.Timestamp(date),
            "stocks": int(len(valid_ret)),
            "equal_weight_return_pct": round(float(valid_ret.mean()), 4),
            "median_return_pct": round(float(valid_ret.median()), 4),
            "advancers_pct": round(float((valid_ret > 0).mean() * 100), 2),
            "decliners_pct": round(float((valid_ret < 0).mean() * 100), 2),
            "above_ma25_pct": round(float(valid_ma["above_ma25"].mean() * 100), 2) if not valid_ma.empty else None,
            "below_ma25_pct": round(float((~valid_ma["above_ma25"]).mean() * 100), 2) if not valid_ma.empty else None,
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _window_market_summary(breadth, curve, spec):
    start = pd.Timestamp(spec["start"])
    end = pd.Timestamp(spec["end"])
    b = breadth[(breadth["date"] >= start) & (breadth["date"] <= end)].copy()
    c = curve[(curve["date"] >= start) & (curve["date"] <= end)].copy()
    if b.empty:
        return {"label": spec["label"], "start": spec["start"], "end": spec["end"], "error": "no_market_data"}

    synthetic = (1 + b["equal_weight_return_pct"] / 100).cumprod()
    worst = b.loc[b["equal_weight_return_pct"].idxmin()]
    narrowest = b.loc[b["advancers_pct"].idxmin()]
    weakest_ma = b.dropna(subset=["above_ma25_pct"])
    weakest_ma_row = weakest_ma.loc[weakest_ma["above_ma25_pct"].idxmin()] if not weakest_ma.empty else None

    portfolio_window_return = None
    portfolio_min_dd = None
    if not c.empty:
        portfolio_window_return = (float(c.iloc[-1]["equity"]) / float(c.iloc[0]["equity"]) - 1) * 100
        portfolio_min_dd = float(c["drawdown_pct"].min())

    return {
        "label": spec["label"],
        "start": str(start.date()),
        "trough": spec.get("trough"),
        "end": str(end.date()),
        "trading_days": int(len(b)),
        "prime_equal_weight_return_pct": round(float((synthetic.iloc[-1] - 1) * 100), 2),
        "negative_market_days": int((b["equal_weight_return_pct"] < 0).sum()),
        "avg_advancers_pct": round(float(b["advancers_pct"].mean()), 2),
        "avg_below_ma25_pct": round(float(b["below_ma25_pct"].dropna().mean()), 2),
        "worst_market_day": str(pd.Timestamp(worst["date"]).date()),
        "worst_market_day_return_pct": round(float(worst["equal_weight_return_pct"]), 2),
        "lowest_advancers_day": str(pd.Timestamp(narrowest["date"]).date()),
        "lowest_advancers_pct": round(float(narrowest["advancers_pct"]), 2),
        "lowest_above_ma25_day": str(pd.Timestamp(weakest_ma_row["date"]).date()) if weakest_ma_row is not None else None,
        "lowest_above_ma25_pct": round(float(weakest_ma_row["above_ma25_pct"]), 2) if weakest_ma_row is not None else None,
        "portfolio_window_return_pct": round(portfolio_window_return, 2) if portfolio_window_return is not None else None,
        "portfolio_worst_drawdown_pct_in_window": round(portfolio_min_dd, 2) if portfolio_min_dd is not None else None,
    }


def _log_market_regime(regime):
    logger.info("\n--- DD局面：東証プライム市場環境 ---")
    for _, r in regime.items():
        if "error" in r:
            logger.info("%s: 市場データなし", r["label"])
            continue
        logger.info(
            "%s (%s〜%s): Prime等金額 %+0.2f%% / 下落日 %s/%s / 平均値上がり比率 %.1f%% / 平均25日線割れ %.1f%%",
            r["label"], r["start"], r["end"], r["prime_equal_weight_return_pct"],
            r["negative_market_days"], r["trading_days"], r["avg_advancers_pct"], r["avg_below_ma25_pct"],
        )
        logger.info(
            "  最悪日 %s: Prime等金額 %+0.2f%% / 値上がり比率最低 %s: %.1f%% / 25日線上最低 %s: %.1f%%",
            r["worst_market_day"], r["worst_market_day_return_pct"],
            r["lowest_advancers_day"], r["lowest_advancers_pct"],
            r["lowest_above_ma25_day"], r["lowest_above_ma25_pct"],
        )
        logger.info(
            "  同期間の正規化混合: %+0.2f%% / 期間内最深DD %+.2f%%",
            r["portfolio_window_return_pct"], r["portfolio_worst_drawdown_pct_in_window"],
        )


def _capacity_from_above_ma25(value):
    if value is None or pd.isna(value):
        return 8
    value = float(value)
    if value >= 40:
        return 8
    if value >= 20:
        return 4
    return 2


def _previous_day_capacity_maps(breadth):
    cap_map = {}
    breadth_map = {}
    b = breadth.sort_values("date").reset_index(drop=True)
    for i in range(1, len(b)):
        current_date = pd.Timestamp(b.iloc[i]["date"])
        prev_value = b.iloc[i - 1]["above_ma25_pct"]
        cap_map[current_date] = _capacity_from_above_ma25(prev_value)
        breadth_map[current_date] = None if pd.isna(prev_value) else float(prev_value)
    return cap_map, breadth_map


def _variable_capacity_portfolio(candidate_a, candidate_b, cap_map, breadth_map):
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
        {"id": i + 1, "capital": INITIAL_CAPITAL / MAX_POSITIONS, "trade": None}
        for i in range(MAX_POSITIONS)
    ]
    selected = []
    skipped_capacity = 0
    skipped_duplicate = 0
    duplicate_same_day = 0
    capacity_days = {8: 0, 4: 0, 2: 0}
    max_concurrent = 0

    def release_slots(entry_date):
        for slot in slots:
            tr = slot["trade"]
            if tr is not None and ranking._release_is_before_entry(tr, entry_date):
                slot["capital"] *= 1 + float(tr["profit_pct"]) / 100
                slot["trade"] = None

    for entry_date in sorted(all_dates):
        release_slots(entry_date)
        capacity = int(cap_map.get(entry_date, 8))
        prev_breadth = breadth_map.get(entry_date)
        capacity_days[capacity] += 1

        day_candidates, _, _, dup_day = ranking._merge_day_candidates(
            a_by_date.get(entry_date, []),
            b_by_date.get(entry_date, []),
            "balanced_rank",
        )
        duplicate_same_day += dup_day

        for tr in day_candidates:
            active_slots = [slot for slot in slots if slot["trade"] is not None]
            active_codes = {str(slot["trade"]["code"]) for slot in active_slots}
            if str(tr["code"]) in active_codes:
                skipped_duplicate += 1
                continue

            # 枠数が下がっても既存ポジションは売らない。
            # 現在保有数がその日の上限以上なら新規だけ停止する。
            if len(active_slots) >= capacity:
                skipped_capacity += 1
                continue

            free_slot = next((slot for slot in slots if slot["trade"] is None), None)
            if free_slot is None:
                skipped_capacity += 1
                continue

            tr_copy = dict(tr)
            tr_copy["slot_id"] = free_slot["id"]
            tr_copy["slot_entry_capital"] = round(free_slot["capital"], 2)
            tr_copy["merge_mode"] = "balanced_rank_variable_capacity"
            tr_copy["entry_capacity"] = capacity
            tr_copy["prev_day_above_ma25_pct"] = prev_breadth
            free_slot["trade"] = tr_copy
            selected.append(tr_copy)
            max_concurrent = max(max_concurrent, len(active_slots) + 1)

    for slot in slots:
        tr = slot["trade"]
        if tr is not None:
            slot["capital"] *= 1 + float(tr["profit_pct"]) / 100
            slot["trade"] = None

    ending_capital = sum(slot["capital"] for slot in slots)
    summary = backtest.summarize_trades(selected)
    summary.update({
        "actual_entries": len(selected),
        "entries_A": sum(1 for tr in selected if tr.get("strategy") == "A"),
        "entries_B": sum(1 for tr in selected if tr.get("strategy") == "B"),
        "skipped_regime_capacity": skipped_capacity,
        "skipped_active_duplicate": skipped_duplicate,
        "duplicate_same_day_A_B": duplicate_same_day,
        "capacity_day_counts": capacity_days,
        "max_concurrent_positions": max_concurrent,
        "initial_capital": INITIAL_CAPITAL,
        "ending_capital": round(ending_capital, 0),
        "portfolio_return_pct": round((ending_capital / INITIAL_CAPITAL - 1) * 100, 2),
    })
    return selected, summary


def _recent_summary(trades):
    recent = [t for t in trades if pd.Timestamp(t["entry_date"]).year >= RECENT_START_YEAR]
    return backtest.summarize_trades(recent)


def _run_variable_capacity(prices, breadth):
    logger.info("\n=== 正規化混合：可変8/4/2枠 検証開始 ===")
    logger.info("前営業日の25日線上比率: 40%%以上=8枠 / 20〜40%%=4枠 / 20%%未満=2枠")
    logger.info("既存建玉は強制決済せず、上限超過中は新規だけ停止します。")

    target_codes = set(prices["Code"].dropna().astype(str).unique())
    strategy = registry.get_strategy("ma5_breakout")
    signals, price_data_by_code = strategy(prices, target_codes)
    candidate_a = ranking._build_strategy_trades(signals, price_data_by_code, "A")
    candidate_b = ranking._build_strategy_trades(signals, price_data_by_code, "B")

    cap_map, breadth_map = _previous_day_capacity_maps(breadth)
    selected, summary = _variable_capacity_portfolio(
        candidate_a, candidate_b, cap_map, breadth_map
    )
    recent = _recent_summary(selected)
    yearly_df = backtest.build_yearly_summary(selected)
    yearly = yearly_df.to_dict("records") if not yearly_df.empty else []

    selected_df = pd.DataFrame(selected)
    trading_dates, close_panel = _build_close_panel(selected_df, prices)
    curve = _build_daily_equity(selected_df, trading_dates, close_panel)
    risk = _risk_summary(curve)

    selected_df.to_csv(VARIABLE_TRADES_CSV, index=False, encoding="utf-8-sig")
    curve.to_csv(VARIABLE_EQUITY_CSV, index=False, encoding="utf-8-sig")
    output = {
        "settings": {
            "capacity_basis": "previous_trading_day_above_ma25_pct",
            "capacity_rule": {"gte_40": 8, "gte_20_lt_40": 4, "lt_20": 2},
            "force_exit_when_capacity_falls": False,
            "ranking_uses_future_information": False,
            "strategy_A_cut_pct": ranking.A_CUT_PCT,
            "strategy_B_cut_pct": ranking.B_CUT_PCT,
        },
        "summary": summary,
        "recent_2025_2026": recent,
        "yearly": yearly,
        "mark_to_market_risk": risk,
    }
    with open(VARIABLE_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    logger.info("\n--- 正規化混合：可変8/4/2枠 結果 ---")
    logger.info(
        "entries %s (A %s / B %s) / PF %s / 勝率 %s%% / 資産 %s円 (%+.2f%%) / 2025-26 PF %s",
        summary["actual_entries"], summary["entries_A"], summary["entries_B"],
        summary.get("profit_factor"), summary.get("win_rate"),
        f"{summary['ending_capital']:,.0f}", summary["portfolio_return_pct"],
        recent.get("profit_factor"),
    )
    logger.info(
        "日次時価最大DD %.2f%% / ピーク %s円 (%s) → ボトム %s円 (%s) / 回復日 %s / 水面下 %s営業日",
        risk["max_drawdown_pct"], f"{risk['peak_equity']:,.0f}", risk["peak_date"],
        f"{risk['trough_equity']:,.0f}", risk["trough_date"],
        risk["recovery_date"] or "未回復", risk["underwater_trading_days"],
    )
    logger.info(
        "上限別シグナル日数: 8枠=%s日 / 4枠=%s日 / 2枠=%s日 / 上限で新規見送り=%s件",
        summary["capacity_day_counts"].get(8, 0),
        summary["capacity_day_counts"].get(4, 0),
        summary["capacity_day_counts"].get(2, 0),
        summary["skipped_regime_capacity"],
    )
    logger.info("\n--- 可変枠：年別PF ---")
    for row in yearly:
        logger.info(
            "%s: trades %s / 勝率 %s%% / 平均損益 %+.3f%% / PF %s",
            int(row["year"]), int(row["total_trades"]), row["win_rate"],
            row["avg_profit_pct"], row["profit_factor"],
        )
    logger.info("可変枠トレード: %s", VARIABLE_TRADES_CSV)
    logger.info("可変枠日次曲線: %s", VARIABLE_EQUITY_CSV)
    logger.info("可変枠集計: %s", VARIABLE_SUMMARY_JSON)
    logger.info("=== 正規化混合：可変8/4/2枠 検証完了 ===")
    return output


def main():
    logger.info("=== 正規化混合 日次時価評価エクイティカーブ作成開始 ===")
    trades, prices = _load_inputs()
    trading_dates, close_panel = _build_close_panel(trades, prices)
    curve = _build_daily_equity(trades, trading_dates, close_panel)
    risk = _risk_summary(curve)

    breadth = _build_market_breadth(prices)
    regime = {
        key: _window_market_summary(breadth, curve, spec)
        for key, spec in REGIME_WINDOWS.items()
    }

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    curve.to_csv(EQUITY_CSV, index=False, encoding="utf-8-sig")
    breadth.to_csv(MARKET_BREADTH_CSV, index=False, encoding="utf-8-sig")
    with open(RISK_JSON, "w", encoding="utf-8") as f:
        json.dump(risk, f, ensure_ascii=False, indent=2)
    with open(MARKET_REGIME_JSON, "w", encoding="utf-8") as f:
        json.dump(regime, f, ensure_ascii=False, indent=2)

    logger.info("\n--- 正規化混合：日次時価評価リスク ---")
    logger.info(
        "日次時価評価最大DD: %.2f%% / ピーク %s円 (%s) → ボトム %s円 (%s)",
        risk["max_drawdown_pct"], f"{risk['peak_equity']:,.0f}", risk["peak_date"],
        f"{risk['trough_equity']:,.0f}", risk["trough_date"],
    )
    logger.info(
        "ピーク→ボトム: %s営業日 / 回復日: %s / 水面下期間: %s営業日",
        risk["peak_to_trough_trading_days"], risk["recovery_date"] or "未回復", risk["underwater_trading_days"],
    )
    logger.info(
        "最終時価評価資産: %s円 (%+.2f%%) / 最大同時保有 %s枠",
        f"{risk['final_equity']:,.0f}", risk["portfolio_return_pct"], risk["max_active_positions"],
    )
    _log_market_regime(regime)

    # 固定8枠の市場診断後、同じ候補条件・同じ正規化順位のまま可変枠を検証する。
    _run_variable_capacity(prices, breadth)

    logger.info("日次曲線: %s", EQUITY_CSV)
    logger.info("市場内部曲線: %s", MARKET_BREADTH_CSV)
    logger.info("市場局面集計: %s", MARKET_REGIME_JSON)
    logger.info("=== 正規化混合 日次時価評価エクイティカーブ作成完了 ===")


if __name__ == "__main__":
    main()
