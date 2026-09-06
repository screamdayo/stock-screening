"""正規化混合ポートフォリオの『日次時価評価』エクイティカーブを作る。

run_backtest_ranking.py が先に出力する
output/max8_shared_ab_balanced_rank.csv を読み、同じ8スロットの実採用トレードを
日次終値で時価評価する。

目的:
- 決済ベースでは見えない保有中の含み損益を含める
- 日次時価評価ベースの最大ドローダウンを出す
- ピーク→ボトム→回復までの期間を確認する

本番スクリーニング条件や候補順位は変更しない。分析専用。
"""

import json
import os

import pandas as pd

import config
from logger import get_logger

logger = get_logger(__name__)

TRADES_CSV = "output/max8_shared_ab_balanced_rank.csv"
EQUITY_CSV = "output/max8_shared_ab_balanced_mtm_equity.csv"
RISK_JSON = "output/max8_shared_ab_balanced_mtm_risk.json"
MAX_POSITIONS = 8
INITIAL_CAPITAL = 1_000_000


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
        "code",
        "slot_id",
        "slot_entry_capital",
        "entry_date",
        "entry_price",
        "exit_date",
        "profit_pct",
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

    prices = pd.read_csv(
        cache_path,
        parse_dates=["Date"],
        dtype={"Code": str},
    )
    prices["Code"] = prices["Code"].map(_normalize_code)
    prices["C"] = pd.to_numeric(prices["C"], errors="coerce")
    prices = prices.dropna(subset=["Date", "Code", "C"])

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

    # 同日に前の建玉を寄りで閉じて新規建玉へ入るケースがあるため、
    # entry_date が最も新しいトレードを優先する。
    latest_entry_date = past["entry_date"].max()
    latest_candidates = past[past["entry_date"] == latest_entry_date]
    tr = latest_candidates.iloc[-1]

    exit_date = pd.Timestamp(tr["exit_date"])
    entry_capital = float(tr["slot_entry_capital"])

    if date >= exit_date:
        realized = entry_capital * (1 + float(tr["profit_pct"]) / 100)
        return realized, None

    code = str(tr["code"])
    entry_price = float(tr["entry_price"])
    if code not in close_panel.columns:
        logger.warning("終値パネルに銘柄 %s がありません。取得額で据え置きます。", code)
        return entry_capital, code

    close_price = close_panel.at[date, code] if date in close_panel.index else float("nan")
    if pd.isna(close_price):
        # 売買停止などで当日の終値がない場合は、直近終値を使う。
        history = close_panel.loc[:date, code].dropna()
        if history.empty:
            logger.warning("%s は %s 時点で終値がなく、取得額で据え置きます。", code, date.date())
            return entry_capital, code
        close_price = float(history.iloc[-1])

    mtm_value = entry_capital * (float(close_price) / entry_price)
    return mtm_value, code


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
    peak_matches = before[before["equity"] == peak_equity]
    peak_row = peak_matches.iloc[-1]
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

    peak_to_trough_trading_days = int(trough_idx - peak_row.name)
    underwater_trading_days = int(end_idx - peak_row.name)

    final_equity = float(curve.iloc[-1]["equity"])
    min_equity_row = curve.loc[curve["equity"].idxmin()]

    return {
        "basis": "daily_close_mark_to_market",
        "max_drawdown_pct": round(float(trough["drawdown_pct"]), 2),
        "peak_equity": round(peak_equity, 0),
        "peak_date": str(peak_date.date()),
        "trough_equity": round(float(trough["equity"]), 0),
        "trough_date": str(trough_date.date()),
        "peak_to_trough_trading_days": peak_to_trough_trading_days,
        "recovery_date": str(recovery_date.date()) if recovery_date is not None else None,
        "recovery_trading_days_from_peak": recovery_trading_days,
        "underwater_trading_days": underwater_trading_days,
        "final_equity": round(final_equity, 0),
        "portfolio_return_pct": round((final_equity / INITIAL_CAPITAL - 1) * 100, 2),
        "minimum_equity": round(float(min_equity_row["equity"]), 0),
        "minimum_equity_date": str(pd.Timestamp(min_equity_row["date"]).date()),
        "max_active_positions": int(curve["active_positions"].max()),
    }


def main():
    logger.info("=== 正規化混合 日次時価評価エクイティカーブ作成開始 ===")
    trades, prices = _load_inputs()
    trading_dates, close_panel = _build_close_panel(trades, prices)
    curve = _build_daily_equity(trades, trading_dates, close_panel)
    risk = _risk_summary(curve)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    curve.to_csv(EQUITY_CSV, index=False, encoding="utf-8-sig")
    with open(RISK_JSON, "w", encoding="utf-8") as f:
        json.dump(risk, f, ensure_ascii=False, indent=2)

    logger.info("\n--- 正規化混合：日次時価評価リスク ---")
    logger.info(
        "日次時価評価最大DD: %.2f%% / ピーク %s円 (%s) → ボトム %s円 (%s)",
        risk["max_drawdown_pct"],
        f"{risk['peak_equity']:,.0f}",
        risk["peak_date"],
        f"{risk['trough_equity']:,.0f}",
        risk["trough_date"],
    )
    logger.info(
        "ピーク→ボトム: %s営業日 / 回復日: %s / 水面下期間: %s営業日",
        risk["peak_to_trough_trading_days"],
        risk["recovery_date"] or "未回復",
        risk["underwater_trading_days"],
    )
    logger.info(
        "最終時価評価資産: %s円 (%+.2f%%) / 最大同時保有 %s枠",
        f"{risk['final_equity']:,.0f}",
        risk["portfolio_return_pct"],
        risk["max_active_positions"],
    )
    logger.info("日次曲線: %s", EQUITY_CSV)
    logger.info("リスク集計: %s", RISK_JSON)
    logger.info("=== 正規化混合 日次時価評価エクイティカーブ作成完了 ===")


if __name__ == "__main__":
    main()
