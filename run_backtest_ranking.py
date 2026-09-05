"""本命の「がっくりB」について、損切り・保有上限の有無を比較する実験。

本番スクリーニング条件・通常バックテストは変更しない。
既存の ranking モードを実験枠として使い、現行 ma5_breakout の入口に対して
以下の出口を比較する。

- fixed_5tp_3sl_10d: 現行 +5% / -3% / 最大10営業日（参考）
- b_sl3_30d: がっくりB + -3%損切り + 最大30営業日（前回の本命）
- b_no_sl_30d: がっくりB + 損切りなし + 最大30営業日
- b_sl3_unlimited: がっくりB + -3%損切り + 保有上限なし
- b_pure: がっくりBのみ。損切りなし・保有上限なし

がっくりBは、MA5の上昇が減速 → 前日非マイナス → 当日下向き + 陰線 + 終値<MA5。
がっくりは当日終値まで見て成立するため、決済は翌営業日始値。
「保有上限なし」でデータ末尾までがっくりが出ない場合だけ、最終日の終値で data_end_exit とする。

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


def _is_gakkuri_b(g, idx):
    """終値確定時点で本命の「がっくりB」が成立したか。"""
    if idx < 2:
        return False

    row = g.iloc[idx]
    if not _is_bearish(row):
        return False

    slope_today = _ma5_slope(g, idx)
    slope_prev = _ma5_slope(g, idx - 1)
    slope_two_days_ago = _ma5_slope(g, idx - 2)
    if slope_today is None or slope_prev is None or slope_two_days_ago is None:
        return False

    # 上向きの傾きが弱まる → 前日はまだ横ばい以上 → 当日下向き。
    decelerating = slope_two_days_ago > slope_prev >= 0 and slope_today < 0
    ma5 = g["MA_SHORT"].iloc[idx]
    return decelerating and pd.notna(ma5) and row["C"] < ma5


def _simulate_gakkuri_b_trade(g, signal_idx, use_stop_loss=True, max_hold_days=30):
    """翌日寄り買い。がっくりB成立の翌日寄りで売却する。"""
    entry_idx = signal_idx + 1
    if entry_idx >= len(g):
        return None

    entry_row = g.iloc[entry_idx]
    entry_price = entry_row["O"]
    if pd.isna(entry_price) or entry_price <= 0:
        return None

    stop_loss_price = entry_price * (1 - config.BACKTEST_STOP_LOSS_PCT / 100)

    if max_hold_days is None:
        hold_end_idx = len(g) - 1
    else:
        hold_end_idx = min(entry_idx + max_hold_days - 1, len(g) - 1)

    for idx in range(entry_idx, hold_end_idx + 1):
        row = g.iloc[idx]
        holding_days = idx - entry_idx + 1

        # 損切りを使うパターンでは、日中の-3%到達を先に判定する。
        if use_stop_loss:
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

    # 30日上限なら time_exit。上限なしならデータ末尾到達。
    exit_reason = "data_end_exit" if max_hold_days is None else "time_exit"
    return {
        "entry_date": entry_row["Date"],
        "entry_price": round(entry_price, 2),
        "exit_date": final_row["Date"],
        "exit_price": round(exit_price, 2),
        "exit_reason": exit_reason,
        "holding_days": hold_end_idx - entry_idx + 1,
        "profit_pct": round((exit_price - entry_price) / entry_price * 100, 3),
    }


def _run_b_backtest(signals, price_data_by_code, label, use_stop_loss=True, max_hold_days=30):
    trades = []
    for sig in signals:
        g = price_data_by_code.get(sig["code"])
        if g is None:
            continue
        result = _simulate_gakkuri_b_trade(
            g,
            sig["signal_idx"],
            use_stop_loss=use_stop_loss,
            max_hold_days=max_hold_days,
        )
        if result is None:
            continue
        result["code"] = sig["code"]
        result["signal_date"] = sig["signal_date"]
        trades.append(result)
    logger.info(f"{label}: シミュレートしたトレード数 {len(trades)}件")
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
            "data_end_exit_count": 0,
        })
        return summary

    df = pd.DataFrame(trades)
    summary["max_holding_days"] = int(df["holding_days"].max())
    summary["gakkuri_exit_count"] = int((df["exit_reason"] == "gakkuri_b").sum())
    summary["stop_loss_count"] = int((df["exit_reason"] == "stop_loss").sum())
    summary["time_exit_count"] = int((df["exit_reason"] == "time_exit").sum())
    summary["data_end_exit_count"] = int((df["exit_reason"] == "data_end_exit").sum())
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
    logger.info("=== がっくりB 出口条件比較バックテスト開始 ===")
    logger.info("本番条件は変更せず、がっくりBの損切り・保有上限だけを比較します。")

    strategy = registry.get_strategy("ma5_breakout")
    target_codes = download.get_target_codes()
    cache_filename = f"backtest_prices_{config.TARGET_MARKET}_{config.BACKTEST_YEARS}y.csv"
    price_df = download.get_price_history_incremental(
        cache_filename=cache_filename,
        years=config.BACKTEST_YEARS,
    )

    signals, price_data_by_code = strategy(price_df, target_codes)

    fixed_trades = backtest.run_backtest(signals, price_data_by_code)
    exit_trades = {
        "fixed_5tp_3sl_10d": fixed_trades,
        "b_sl3_30d": _run_b_backtest(
            signals, price_data_by_code, "b_sl3_30d", use_stop_loss=True, max_hold_days=30
        ),
        "b_no_sl_30d": _run_b_backtest(
            signals, price_data_by_code, "b_no_sl_30d", use_stop_loss=False, max_hold_days=30
        ),
        "b_sl3_unlimited": _run_b_backtest(
            signals, price_data_by_code, "b_sl3_unlimited", use_stop_loss=True, max_hold_days=None
        ),
        "b_pure": _run_b_backtest(
            signals, price_data_by_code, "b_pure", use_stop_loss=False, max_hold_days=None
        ),
    }

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
        in_ma25_band = pd.notna(f["ma25_dev_pct"]) and -10 <= f["ma25_dev_pct"] < -5
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
            "gakkuri_mode": "B",
            "stop_loss_pct": config.BACKTEST_STOP_LOSS_PCT,
            "limited_hold_days": GAKKURI_MAX_HOLD_DAYS,
            "gakkuri_exit_timing": "signal next business day open",
            "unlimited_fallback": "last available close as data_end_exit",
        },
        "groups": {},
    }

    logger.info("\n" + "=" * 108)
    logger.info("がっくりB 出口条件比較")
    logger.info("=" * 108)

    for group_name, keys in groups.items():
        logger.info(f"\n### 入口: {group_name}（対象キー {len(keys)}件）")
        group_result = {}

        for exit_name, trades in exit_trades.items():
            selected = _filter_by_keys(trades, keys)
            summary = _extended_summary(selected)
            yearly = _yearly_records(selected)
            group_result[exit_name] = {"summary": summary, "yearly": yearly}

            logger.info(f"{exit_name}: {_fmt(summary)}")
            logger.info(
                "  決済内訳: "
                f"がっくり {summary.get('gakkuri_exit_count', 0)} / "
                f"損切り {summary.get('stop_loss_count', 0)} / "
                f"30日期日 {summary.get('time_exit_count', 0)} / "
                f"データ末尾 {summary.get('data_end_exit_count', 0)} / "
                f"最大保有 {summary.get('max_holding_days')}日"
            )
            if yearly:
                logger.info("  年別: " + " / ".join(
                    f"{int(r['year'])}: {int(r['total_trades'])}件 PF{r['profit_factor']} 勝率{r['win_rate']}%"
                    for r in yearly
                ))

        output["groups"][group_name] = group_result

    os.makedirs("output", exist_ok=True)
    with open("output/gakkuri_b_variant_comparison.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    for exit_name, trades in exit_trades.items():
        pd.DataFrame(trades).to_csv(
            f"output/gakkuri_b_trades_{exit_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    logger.info("\n結果を output/gakkuri_b_variant_comparison.json に保存しました。")
    logger.info("=== がっくりB 出口条件比較バックテスト完了 ===")


if __name__ == "__main__":
    main()
