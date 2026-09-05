"""MA5ブレイクアウトの同日候補を複数指標で採点し、上位8銘柄だけを比較する実験。

本番スクリーニング条件は変更しない。
現行 ma5_breakout が出した全シグナルに対して、以下6項目を0〜1に正規化して採点する。
- 出来高倍率
- 当日上昇率
- 上ヒゲの短さ（高値引け度）
- MA5上昇率
- RSI(14)
- MA25乖離率

5種類の重み付けを比較し、各シグナル日ごとに上位8件だけを残してPF・勝率・年別PFを出す。
さらに2021〜2024と2025〜2026に分け、直近でも再現するか確認する。
"""

import os
import json

import pandas as pd

import backtest
import config
import download
from logger import get_logger
from strategies import registry

logger = get_logger(__name__)

TOP_N = 8
RSI_PERIOD = 14

WEIGHT_PATTERNS = {
    "equal": {
        "volume": 1.0, "gain": 1.0, "wick": 1.0,
        "ma5": 1.0, "rsi": 1.0, "ma25_dev": 1.0,
    },
    "volume_focus": {
        "volume": 2.5, "gain": 1.0, "wick": 1.0,
        "ma5": 1.0, "rsi": 0.75, "ma25_dev": 0.75,
    },
    "ma5_focus": {
        "volume": 1.0, "gain": 1.0, "wick": 1.0,
        "ma5": 2.5, "rsi": 0.75, "ma25_dev": 0.75,
    },
    "candle_focus": {
        "volume": 1.0, "gain": 2.0, "wick": 2.0,
        "ma5": 1.0, "rsi": 0.75, "ma25_dev": 0.75,
    },
    "overheat_avoid": {
        "volume": 1.0, "gain": 1.0, "wick": 1.0,
        "ma5": 1.0, "rsi": 2.0, "ma25_dev": 2.0,
    },
}


def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(v)))


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


def _score_volume(ratio):
    if pd.isna(ratio):
        return 0.0
    return _clamp((ratio - 1.0) / 1.5)


def _score_gain(gain_pct):
    if pd.isna(gain_pct):
        return 0.0
    return _clamp((gain_pct - 1.5) / 3.5)


def _score_wick(row):
    high, low, close = row["H"], row["L"], row["C"]
    if pd.isna(high) or pd.isna(low) or pd.isna(close) or high <= low:
        return 0.5
    return _clamp((close - low) / (high - low))


def _score_ma5(rise_pct):
    if pd.isna(rise_pct):
        return 0.0
    return _clamp(rise_pct / 0.8)


def _score_rsi(rsi):
    if pd.isna(rsi):
        return 0.0
    rsi = float(rsi)
    if 40 <= rsi <= 55:
        return 1.0
    if 30 <= rsi < 40:
        return (rsi - 30) / 10
    if 55 < rsi <= 70:
        return (70 - rsi) / 15
    return 0.0


def _score_ma25_dev(dev_pct):
    if pd.isna(dev_pct):
        return 0.0
    d = abs(float(dev_pct))
    if d <= 5:
        return 1.0
    if d >= 15:
        return 0.0
    return (15 - d) / 10


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
    ma25_dev_pct = (row["C"] / ma25 - 1) * 100 if pd.notna(ma25) and ma25 != 0 else None

    if "RSI14" not in g.columns:
        g["RSI14"] = _rsi_series(g["C"], RSI_PERIOD)
    rsi = g["RSI14"].iloc[idx]

    scores = {
        "volume": _score_volume(volume_ratio),
        "gain": _score_gain(gain_pct),
        "wick": _score_wick(row),
        "ma5": _score_ma5(ma5_rise_pct),
        "rsi": _score_rsi(rsi),
        "ma25_dev": _score_ma25_dev(ma25_dev_pct),
    }
    raw = {
        "volume_ratio": volume_ratio,
        "gain_pct": gain_pct,
        "ma5_rise_pct": ma5_rise_pct,
        "rsi14": rsi,
        "ma25_dev_pct": ma25_dev_pct,
    }
    return scores, raw


def _weighted_score(scores, weights):
    denom = sum(weights.values())
    if denom <= 0:
        return 0.0
    return sum(scores[k] * weights[k] for k in weights) / denom * 100


def _select_top_n(scored_signals, pattern_name):
    df = pd.DataFrame(scored_signals)
    score_col = f"score_{pattern_name}"
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df = df.sort_values(["signal_date", score_col, "code"], ascending=[True, False, True])
    return df.groupby("signal_date", group_keys=False).head(TOP_N)


def _trade_key(x):
    return str(x["code"]), pd.Timestamp(x["signal_date"])


def _summary_for_period(trades, start_year=None, end_year=None):
    if start_year is None:
        return backtest.summarize_trades(trades)
    filtered = []
    for t in trades:
        y = pd.Timestamp(t["signal_date"]).year
        if start_year <= y <= end_year:
            filtered.append(t)
    return backtest.summarize_trades(filtered)


def _fmt_summary(s):
    return f"{s.get('total_trades', 0)}件 / 勝率{s.get('win_rate')}% / PF{s.get('profit_factor')}"


def main():
    logger.info("=== MA5ランキング比較バックテスト開始 ===")
    logger.info(f"同日候補から上位{TOP_N}件を選択 / 指標6種 / 重み5パターン")

    strategy = registry.get_strategy("ma5_breakout")
    target_codes = download.get_target_codes()
    cache_filename = f"backtest_prices_{config.TARGET_MARKET}_{config.BACKTEST_YEARS}y.csv"
    price_df = download.get_price_history_incremental(
        cache_filename=cache_filename,
        years=config.BACKTEST_YEARS,
    )

    signals, price_data_by_code = strategy(price_df, target_codes)
    all_trades = backtest.run_backtest(signals, price_data_by_code)
    all_summary = backtest.summarize_trades(all_trades)
    logger.info("全シグナル: " + _fmt_summary(all_summary))

    scored = []
    for sig in signals:
        g = price_data_by_code.get(sig["code"])
        if g is None:
            continue
        scores, raw = _feature_row(g, sig["signal_idx"])
        item = dict(sig)
        item.update(raw)
        for pattern_name, weights in WEIGHT_PATTERNS.items():
            item[f"score_{pattern_name}"] = round(_weighted_score(scores, weights), 4)
        scored.append(item)

    trades_by_key = {_trade_key(t): t for t in all_trades}
    comparison = {}
    output_rows = []

    logger.info("\n" + "=" * 72)
    logger.info("ランキング結果（全期間 / 2021-2024 / 2025-2026）")
    logger.info("=" * 72)

    for pattern_name in WEIGHT_PATTERNS:
        selected_df = _select_top_n(scored, pattern_name)
        selected_keys = {
            (str(r["code"]), pd.Timestamp(r["signal_date"]))
            for _, r in selected_df.iterrows()
        }
        selected_trades = [trades_by_key[k] for k in selected_keys if k in trades_by_key]

        full = backtest.summarize_trades(selected_trades)
        train = _summary_for_period(selected_trades, 2021, 2024)
        test = _summary_for_period(selected_trades, 2025, 2026)
        yearly_df = backtest.build_yearly_summary(selected_trades)
        yearly = yearly_df.to_dict("records") if not yearly_df.empty else []

        logger.info(
            f"{pattern_name}: 全期間 {_fmt_summary(full)} | "
            f"2021-24 {_fmt_summary(train)} | 2025-26 {_fmt_summary(test)}"
        )
        if yearly:
            logger.info("  年別: " + " / ".join(
                f"{int(r['year'])}: {int(r['total_trades'])}件 PF{r['profit_factor']} 勝率{r['win_rate']}%"
                for r in yearly
            ))

        comparison[pattern_name] = {
            "weights": WEIGHT_PATTERNS[pattern_name],
            "full": full,
            "train_2021_2024": train,
            "test_2025_2026": test,
            "yearly": yearly,
        }

        for _, row in selected_df.iterrows():
            out = row.to_dict()
            out["ranking_pattern"] = pattern_name
            output_rows.append(out)

    os.makedirs("output", exist_ok=True)
    with open("output/ranking_comparison.json", "w", encoding="utf-8") as f:
        json.dump({
            "top_n": TOP_N,
            "baseline": all_summary,
            "patterns": comparison,
        }, f, ensure_ascii=False, indent=2, default=str)

    pd.DataFrame(output_rows).to_csv(
        "output/ranking_selected_signals.csv", index=False, encoding="utf-8-sig"
    )
    logger.info("比較結果を output/ranking_comparison.json に保存しました。")
    logger.info("=== MA5ランキング比較バックテスト完了 ===")


if __name__ == "__main__":
    main()
