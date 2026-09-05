"""MA5ブレイクアウトの各指標を単独で区切り、PF・勝率・件数を比較する実験。

本番スクリーニング条件は変更しない。
現行 ma5_breakout が出した全シグナルについて、以下6指標をそれぞれ単独でビン分けする。
- RSI(14)
- 出来高倍率（前日比）
- 当日上昇率
- MA5上昇率
- MA25乖離率
- 終値位置（当日レンジ内。100%に近いほど上ヒゲが短い）

各ビンについて全期間と直近2025-2026の件数・勝率・PFを表示する。
ランキングや上位8件選択は行わない。
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


def _feature_row(g, idx):
    row = g.iloc[idx]
    prev = g.iloc[idx - 1] if idx > 0 else None

    prev_volume = prev.get("Vo") if prev is not None else None
    today_volume = row.get("Vo")
    volume_ratio = (
        today_volume / prev_volume
        if prev_volume is not None and pd.notna(prev_volume) and prev_volume > 0
        and pd.notna(today_volume)
        else None
    )

    gain_pct = (row["C"] / row["O"] - 1) * 100 if row["O"] > 0 else None

    ma5_prev = g["MA_SHORT"].iloc[idx - 1] if idx > 0 else None
    ma5_today = g["MA_SHORT"].iloc[idx]
    ma5_rise_pct = (
        (ma5_today / ma5_prev - 1) * 100
        if ma5_prev is not None and pd.notna(ma5_prev) and ma5_prev != 0
        and pd.notna(ma5_today)
        else None
    )

    ma25 = g["MA_LONG"].iloc[idx]
    ma25_dev_pct = (
        (row["C"] / ma25 - 1) * 100
        if pd.notna(ma25) and ma25 != 0 else None
    )

    if "RSI14" not in g.columns:
        g["RSI14"] = _rsi_series(g["C"], RSI_PERIOD)
    rsi14 = g["RSI14"].iloc[idx]

    high, low, close = row["H"], row["L"], row["C"]
    close_position_pct = None
    if pd.notna(high) and pd.notna(low) and pd.notna(close) and high > low:
        close_position_pct = (close - low) / (high - low) * 100

    return {
        "rsi14": rsi14,
        "volume_ratio": volume_ratio,
        "gain_pct": gain_pct,
        "ma5_rise_pct": ma5_rise_pct,
        "ma25_dev_pct": ma25_dev_pct,
        "close_position_pct": close_position_pct,
    }


BIN_SPECS = {
    "RSI14": ("rsi14", [-float("inf"), 30, 40, 50, 60, 70, float("inf")],
              ["<30", "30-40", "40-50", "50-60", "60-70", ">=70"]),
    "出来高倍率": ("volume_ratio", [-float("inf"), 1.0, 1.2, 1.5, 2.0, 3.0, float("inf")],
               ["<1.0x", "1.0-1.2x", "1.2-1.5x", "1.5-2.0x", "2.0-3.0x", ">=3.0x"]),
    "当日上昇率": ("gain_pct", [-float("inf"), 2.0, 3.0, 5.0, 8.0, float("inf")],
               ["<2%", "2-3%", "3-5%", "5-8%", ">=8%"]),
    "MA5上昇率": ("ma5_rise_pct", [-float("inf"), 0.1, 0.3, 0.6, 1.0, float("inf")],
                ["<0.1%", "0.1-0.3%", "0.3-0.6%", "0.6-1.0%", ">=1.0%"]),
    "MA25乖離率": ("ma25_dev_pct", [-float("inf"), -10, -5, 0, 5, 10, float("inf")],
                 ["<-10%", "-10--5%", "-5-0%", "0-5%", "5-10%", ">=10%"]),
    "終値位置": ("close_position_pct", [-float("inf"), 60, 75, 85, 95, float("inf")],
              ["<60%", "60-75%", "75-85%", "85-95%", ">=95%"]),
}


def _trade_key(x):
    return str(x["code"]), pd.Timestamp(x["signal_date"])


def _summarize(rows):
    trades = [r["trade"] for r in rows]
    return backtest.summarize_trades(trades)


def _summarize_recent(rows):
    trades = [
        r["trade"] for r in rows
        if pd.Timestamp(r["trade"]["signal_date"]).year >= 2025
    ]
    return backtest.summarize_trades(trades)


def _fmt(s):
    return f"{s.get('total_trades', 0)}件 / 勝率{s.get('win_rate')}% / PF{s.get('profit_factor')}"


def main():
    logger.info("=== MA5 指標別・単独性能バックテスト開始 ===")
    logger.info("ランキングなし。現行全シグナルを各指標だけでビン分けして比較します。")

    strategy = registry.get_strategy("ma5_breakout")
    target_codes = download.get_target_codes()
    cache_filename = f"backtest_prices_{config.TARGET_MARKET}_{config.BACKTEST_YEARS}y.csv"
    price_df = download.get_price_history_incremental(
        cache_filename=cache_filename,
        years=config.BACKTEST_YEARS,
    )

    signals, price_data_by_code = strategy(price_df, target_codes)
    all_trades = backtest.run_backtest(signals, price_data_by_code)
    baseline = backtest.summarize_trades(all_trades)
    logger.info("全シグナル基準: " + _fmt(baseline))

    trades_by_key = {_trade_key(t): t for t in all_trades}
    rows = []
    for sig in signals:
        key = (str(sig["code"]), pd.Timestamp(sig["signal_date"]))
        trade = trades_by_key.get(key)
        if trade is None:
            continue
        g = price_data_by_code.get(sig["code"])
        if g is None:
            continue
        features = _feature_row(g, sig["signal_idx"])
        rows.append({"trade": trade, **features})

    results = {}
    logger.info("\n" + "=" * 78)
    logger.info("単独性能（全期間 | 2025-2026）")
    logger.info("=" * 78)

    for display_name, (field, edges, labels) in BIN_SPECS.items():
        logger.info(f"\n--- {display_name} ---")
        values = pd.Series([r[field] for r in rows], dtype="float64")
        bins = pd.cut(values, bins=edges, labels=labels, right=False)

        metric_results = []
        for label in labels:
            selected = [rows[i] for i in range(len(rows)) if bins.iloc[i] == label]
            if not selected:
                continue
            full = _summarize(selected)
            recent = _summarize_recent(selected)
            logger.info(f"{label}: 全期間 {_fmt(full)} | 2025-26 {_fmt(recent)}")
            metric_results.append({
                "bin": label,
                "full": full,
                "recent_2025_2026": recent,
            })

        missing = [r for r in rows if pd.isna(r[field])]
        if missing:
            full = _summarize(missing)
            recent = _summarize_recent(missing)
            logger.info(f"欠損: 全期間 {_fmt(full)} | 2025-26 {_fmt(recent)}")
            metric_results.append({
                "bin": "missing",
                "full": full,
                "recent_2025_2026": recent,
            })

        results[display_name] = metric_results

    os.makedirs("output", exist_ok=True)
    with open("output/indicator_standalone_performance.json", "w", encoding="utf-8") as f:
        json.dump({
            "baseline": baseline,
            "metrics": results,
        }, f, ensure_ascii=False, indent=2, default=str)

    feature_rows = []
    for r in rows:
        out = {k: v for k, v in r.items() if k != "trade"}
        out.update({
            "code": r["trade"]["code"],
            "signal_date": r["trade"]["signal_date"],
            "profit_pct": r["trade"]["profit_pct"],
            "exit_reason": r["trade"]["exit_reason"],
        })
        feature_rows.append(out)
    pd.DataFrame(feature_rows).to_csv(
        "output/indicator_features.csv", index=False, encoding="utf-8-sig"
    )

    logger.info("\n結果を output/indicator_standalone_performance.json に保存しました。")
    logger.info("=== MA5 指標別・単独性能バックテスト完了 ===")


if __name__ == "__main__":
    main()
