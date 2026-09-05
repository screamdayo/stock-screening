"""本の「くいっと→がっくり」を出口まで含めて比較する実験。

本番スクリーニング条件・通常バックテストは変更しない。
既存の ranking モードを実験枠として使い、現行 ma5_breakout の入口に対して
以下4種類の出口を比較する。

- fixed: 現行 +5% / -3% / 最大10営業日
- gakkuri_a: 前日までMA5上向き → 当日下向き + 陰線
- gakkuri_b: MA5上昇が減速 → 前日非マイナス → 当日下向き + 陰線 + 終値<MA5
- gakkuri_c: B + 当日高値>=MA5 + 長い下ヒゲ除外

がっくりは当日終値まで見て成立するため、決済は翌営業日始値。
がっくり3種は -3% 損切りを残し、最大30営業日で時間切れ決済する。

入口は以下3グループで別集計する。
- 全シグナル
- MA25乖離 -10〜-5% × 出来高前日比 <1.0
- MA25乖離 -10〜-5% × RSI14 40〜50
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
GAKKURI_MAX_HOLD_DAYS = 30
LOWER_WICK_MAX_BODY_MULTIPLE = 1.5


def _rsi_series(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(100).where(avg_gain.notna())
    return rsi


def _entry_features(g, idx):
    row = g.iloc[idx]
    prev = g.iloc[idx - 1] if idx > 0 else None

    prev_volume = prev.get("Vo") if prev is not None else None
    today_volume = row.get("Vo")
    volume_ratio = (
        today_volume / prev_volume
        if prev_volume is not None
        and pd.notna(prev_volume)
        and prev_volume > 0
        and pd.notna(today_volume)
        else None
    )

    ma25 = g["MA_LONG"].iloc[idx]
    ma25_dev_pct = (
        (row["C"] / ma25 - 1) * 100
        if pd.notna(ma25) and ma25 != 0
        else None
    )

    if "RSI14" not in g.columns:
        g["RSI14"] = _rsi_series(g["C"], RSI_PERIOD)
    rsi14 = g["RSI14"].iloc[idx]

    return {
        "volume_ratio": volume_ratio,
        "ma25_dev_pct": ma25_dev_pct,
        "rsi14": rsi14,
    }


def _ma5_slope(g, idx):
    if idx <= 0:
        return None
    prev = g["MA_SHORT"].iloc[idx - 1]
    today = g["MA_SHORT"].iloc[idx]
    if pd.isna(prev) or pd.isna(today):
        return None
    return today - prev


def _is_bearish(row):
    return pd.notna(row["O"]) and pd.notna(row["C"]) and row["C"] < row["O"]


def _is_gakkuri(g, idx, mode):
    """終値確定時点で、指定モードの「がっくり」が成立したか。"""
    if idx < 2:
        return False

    row = g.iloc[idx]
    if not _is_bearish(row):
        return False

    slope_today = _ma5_slope(g, idx)
    slope_prev = _ma5_slope(g, idx - 1)
    if slope_today is None or slope_prev is None:
        return False

    if mode == "gakkuri_a":
        return slope_prev > 0 and slope_today < 0

    slope_two_days_ago = _ma5_slope(g, idx - 2)
    if slope_two_days_ago is None:
        return False

    # 本の「上向きの傾きが段々と緩やか→水平→下向き」を数値化。
    # 2日前より前日の傾きが弱く、前日はまだ横ばい以上、当日初めて下向き。
    decelerating = slope_two_days_ago > slope_prev >= 0 and slope_today < 0
    ma5 = g["MA_SHORT"].iloc[idx]
    if not decelerating or pd.isna(ma5) or row["C"] >= ma5:
        return False

    if mode == "gakkuri_b":
        return True

    if mode == "gakkuri_c":
        # 「5日線にぶら下がる陰線」: 当日の値幅がMA5に触れ、終値は下。
        if pd.isna(row["H"]) or row["H"] < ma5:
            return False

        body = row["O"] - row["C"]  # 陰線なので正
        lower_wick = row["C"] - row["L"] if pd.notna(row["L"]) else None
        if body <= 0 or lower_wick is None:
            return False
        if lower_wick > body * LOWER_WICK_MAX_BODY_MULTIPLE:
            return False
        return True

    raise ValueError(f"unknown gakkuri mode: {mode}")


def _simulate_gakkuri_trade(g, signal_idx, mode):
    """翌日寄り買い、-3%損切り、がっくり翌日寄り売り、最大30日。"""
    entry_idx = signal_idx + 1
    if entry_idx >= len(g):
        return None

    entry_row = g.iloc[entry_idx]
    entry_price = entry_row["O"]
    if pd.isna(entry_price) or entry_price <= 0:
        return None

    stop_loss_price = entry_price * (1 - config.BACKTEST_STOP_LOSS_PCT / 100)
    hold_end_idx = min(entry_idx + GAKKURI_MAX_HOLD_DAYS - 1, len(g) - 1)

    for idx in range(entry_idx, hold_end_idx + 1):
        row = g.iloc[idx]
        holding_days = idx - entry_idx + 1

        # 当日中の損切りは、引け後にしか分からない「がっくり」より先に判定。
        low = row.get("L")
        if pd.notna(low) and low <= stop_loss_price:
            return {
                "entry_date": entry_row["Date"],
                "entry_price": round(entry_price, 2),
                "exit_date": row["Date"],
                "exit_price": round(stop_loss_price, 2),
                "exit_reason": "stop_loss",
                "holding_days": holding_days,
                "profit_pct": round((stop_loss_price - entry_price) / entry_price * 100, 3),
            }

        if _is_gakkuri(g, idx, mode):
            exit_idx = idx + 1
            if exit_idx < len(g):
                exit_row = g.iloc[exit_idx]
                exit_price = exit_row["O"]
                if pd.notna(exit_price) and exit_price > 0:
                    # がっくり判定日の翌日寄りまでを保有日数に含める。
                    exit_holding_days = exit_idx - entry_idx + 1
                    return {
                        "entry_date": entry_row["Date"],
                        "entry_price": round(entry_price, 2),
                        "exit_date": exit_row["Date"],
                        "exit_price": round(exit_price, 2),
                        "exit_reason": mode,
                        "holding_days": exit_holding_days,
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
        "exit_reason": "time_exit",
        "holding_days": hold_end_idx - entry_idx + 1,
        "profit_pct": round((exit_price - entry_price) / entry_price * 100, 3),
    }


def _run_gakkuri_backtest(signals, price_data_by_code, mode):
    trades = []
    for sig in signals:
        g = price_data_by_code.get(sig["code"])
        if g is None:
            continue
        result = _simulate_gakkuri_trade(g, sig["signal_idx"], mode)
        if result is None:
            continue
        result["code"] = sig["code"]
        result["signal_date"] = sig["signal_date"]
        trades.append(result)
    logger.info(f"{mode}: シミュレートしたトレード数 {len(trades)}件")
    return trades


def _trade_key(x):
    return str(x["code"]), pd.Timestamp(x["signal_date"])


def _extended_summary(trades):
    summary = dict(backtest.summarize_trades(trades))
    if not trades:
        summary.update({
            "max_holding_days": None,
            "gakkuri_exit_count": 0,
            "stop_loss_count": 0,
            "time_exit_count": 0,
        })
        return summary

    df = pd.DataFrame(trades)
    summary["max_holding_days"] = int(df["holding_days"].max())
    summary["gakkuri_exit_count"] = int(df["exit_reason"].astype(str).str.startswith("gakkuri_").sum())
    summary["stop_loss_count"] = int((df["exit_reason"] == "stop_loss").sum())
    summary["time_exit_count"] = int((df["exit_reason"] == "time_exit").sum())
    return summary


def _fmt(summary):
    return (
        f"{summary.get('total_trades', 0)}件 / 勝率{summary.get('win_rate')}% / "
        f"PF{summary.get('profit_factor')} / 平均勝ち{summary.get('avg_profit_pct')}% / "
        f"平均負け{summary.get('avg_loss_pct')}% / 平均保有{summary.get('avg_holding_days')}日"
    )


def _yearly_records(trades):
    yearly = backtest.build_yearly_summary(trades)
    return yearly.to_dict("records") if not yearly.empty else []


def _filter_by_keys(trades, keys):
    return [t for t in trades if _trade_key(t) in keys]


def main():
    logger.info("=== くいっと→がっくり 出口比較バックテスト開始 ===")
    logger.info("本番条件は変更せず、現行入口に対して fixed / がっくりA/B/C を比較します。")

    strategy = registry.get_strategy("ma5_breakout")
    target_codes = download.get_target_codes()
    cache_filename = f"backtest_prices_{config.TARGET_MARKET}_{config.BACKTEST_YEARS}y.csv"
    price_df = download.get_price_history_incremental(
        cache_filename=cache_filename,
        years=config.BACKTEST_YEARS,
    )

    signals, price_data_by_code = strategy(price_df, target_codes)

    # 現行出口は既存実装をそのまま利用。
    fixed_trades = backtest.run_backtest(signals, price_data_by_code)
    exit_trades = {
        "fixed_5tp_3sl_10d": fixed_trades,
        "gakkuri_a": _run_gakkuri_backtest(signals, price_data_by_code, "gakkuri_a"),
        "gakkuri_b": _run_gakkuri_backtest(signals, price_data_by_code, "gakkuri_b"),
        "gakkuri_c": _run_gakkuri_backtest(signals, price_data_by_code, "gakkuri_c"),
    }

    # 入口条件はシグナル当日時点の情報だけで分類する。
    all_keys = set()
    volume_keys = set()
    rsi_keys = set()
    for sig in signals:
        g = price_data_by_code.get(sig["code"])
        if g is None:
            continue
        key = (str(sig["code"]), pd.Timestamp(sig["signal_date"]))
        all_keys.add(key)
        f = _entry_features(g, sig["signal_idx"])
        in_ma25_band = (
            pd.notna(f["ma25_dev_pct"])
            and -10 <= f["ma25_dev_pct"] < -5
        )
        if in_ma25_band and pd.notna(f["volume_ratio"]) and f["volume_ratio"] < 1.0:
            volume_keys.add(key)
        if in_ma25_band and pd.notna(f["rsi14"]) and 40 <= f["rsi14"] < 50:
            rsi_keys.add(key)

    groups = {
        "全シグナル": all_keys,
        "MA25乖離-10〜-5% × 出来高<1.0x": volume_keys,
        "MA25乖離-10〜-5% × RSI40-50": rsi_keys,
    }

    output = {
        "settings": {
            "gakkuri_max_hold_days": GAKKURI_MAX_HOLD_DAYS,
            "stop_loss_pct": config.BACKTEST_STOP_LOSS_PCT,
            "lower_wick_max_body_multiple": LOWER_WICK_MAX_BODY_MULTIPLE,
            "gakkuri_exit_timing": "signal next business day open",
        },
        "groups": {},
    }

    logger.info("\n" + "=" * 104)
    logger.info("出口比較")
    logger.info("=" * 104)

    for group_name, keys in groups.items():
        logger.info(f"\n### 入口: {group_name}（対象キー {len(keys)}件）")
        group_result = {}

        for exit_name, trades in exit_trades.items():
            selected = _filter_by_keys(trades, keys)
            summary = _extended_summary(selected)
            yearly = _yearly_records(selected)
            group_result[exit_name] = {
                "summary": summary,
                "yearly": yearly,
            }

            logger.info(f"{exit_name}: {_fmt(summary)}")
            logger.info(
                "  決済内訳: "
                f"がっくり {summary.get('gakkuri_exit_count', 0)} / "
                f"損切り {summary.get('stop_loss_count', 0)} / "
                f"期日 {summary.get('time_exit_count', 0)} / "
                f"最大保有 {summary.get('max_holding_days')}日"
            )
            if yearly:
                logger.info("  年別: " + " / ".join(
                    f"{int(r['year'])}: {int(r['total_trades'])}件 PF{r['profit_factor']} 勝率{r['win_rate']}%"
                    for r in yearly
                ))

        output["groups"][group_name] = group_result

    os.makedirs("output", exist_ok=True)
    with open("output/gakkuri_exit_comparison.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    # 後から個別トレードも確認できるよう、出口別にCSV保存。
    for exit_name, trades in exit_trades.items():
        pd.DataFrame(trades).to_csv(
            f"output/gakkuri_trades_{exit_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    logger.info("\n結果を output/gakkuri_exit_comparison.json に保存しました。")
    logger.info("=== くいっと→がっくり 出口比較バックテスト完了 ===")


if __name__ == "__main__":
    main()
