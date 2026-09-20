import argparse
import os
from pathlib import Path

import pandas as pd

import download
from compare_gc import YEARS, DEEP_DROP_START_PCT, DEEP_DROP_RESET_PCT, prepare, summarize, fmt

HOLD_DAYS = [5, 7, 10, 12, 15]


def px(g, i, col):
    v = pd.to_numeric(g.loc[i, col], errors="coerce")
    return None if pd.isna(v) else float(v)


def collect_for_hold(price_df, target_codes, hold_days):
    targets = set(map(str, target_codes))
    trades = []
    unresolved = 0

    for code, group in price_df.groupby("Code"):
        code = str(code)
        if code not in targets:
            continue

        g = prepare(group)
        deep_episode = False
        gc_count = 0

        for i in range(25, len(g) - 1):
            dd = g.loc[i, "DRAWDOWN60_PCT"]

            if not pd.isna(dd):
                if not deep_episode and float(dd) <= DEEP_DROP_START_PCT:
                    deep_episode = True
                    gc_count = 0
                elif deep_episode and float(dd) > DEEP_DROP_RESET_PCT:
                    deep_episode = False
                    gc_count = 0

            ma5_prev, ma25_prev = g.loc[i - 1, ["MA5", "MA25"]]
            ma5_now, ma25_now = g.loc[i, ["MA5", "MA25"]]
            if any(pd.isna(x) for x in [ma5_prev, ma25_prev, ma5_now, ma25_now]):
                continue
            if not (ma5_prev <= ma25_prev and ma5_now > ma25_now):
                continue

            if not deep_episode:
                continue

            gc_count += 1
            if gc_count != 3:
                continue

            slope = g.loc[i, "MA25_5D_SLOPE_PCT"]
            if pd.isna(slope) or float(slope) <= 0.25:
                continue

            entry_idx = i + 1
            exit_idx = entry_idx + hold_days - 1
            if exit_idx >= len(g):
                unresolved += 1
                continue

            entry = px(g, entry_idx, "O")
            exit_px = px(g, exit_idx, "C")
            if entry is None or exit_px is None or entry <= 0:
                continue

            closes = pd.to_numeric(g.loc[entry_idx:exit_idx, "C"], errors="coerce").dropna()
            if closes.empty:
                continue

            ret = (exit_px / entry - 1) * 100
            mfe = (float(closes.max()) / entry - 1) * 100

            trades.append({
                "code": code,
                "signal_date": str(pd.Timestamp(g.loc[i, "Date"]).date()),
                "ret_pct": ret,
                "mfe_pct": mfe,
                "giveback_pct": mfe - ret,
            })

    return trades, unresolved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, choices=HOLD_DAYS, required=True)
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)
    target_codes = download.get_target_codes()
    price_df = download.get_price_history_incremental(
        cache_filename="backtest_prices_prime_5y.csv", years=YEARS
    )
    price_df["Date"] = pd.to_datetime(price_df["Date"])

    rows, unresolved = collect_for_hold(price_df, target_codes, args.days)
    s = summarize([r["ret_pct"] for r in rows])

    if rows:
        avg_mfe = float(pd.Series([r["mfe_pct"] for r in rows], dtype=float).mean())
        avg_giveback = float(pd.Series([r["giveback_pct"] for r in rows], dtype=float).mean())
    else:
        avg_mfe = None
        avg_giveback = None

    lines = [
        f"GC fixed-exit grid: {args.days} trading days / Prime / {YEARS} years through {price_df['Date'].max().date()}",
        "Entry = exactly 3rd GC in deep-drop episode + MA25 rising (> +0.25% over 5d)",
        "Buy = next open after GC",
        f"Sell = close on holding day {args.days}",
        "",
        "[Overall]",
        f"n={s['n']} unresolved={unresolved} win={fmt(s['win_rate'])}% avg={fmt(s['avg'])}% median={fmt(s['median'])}%",
        f"avg_MFE={fmt(avg_mfe)}% avg_giveback_from_best_close={fmt(avg_giveback)}%",
    ]

    text = "\n".join(lines)
    Path(f"output/gc_fixed_exit_{args.days}d.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
