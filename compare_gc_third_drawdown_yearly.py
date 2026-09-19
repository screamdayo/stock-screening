import os
from collections import defaultdict
from pathlib import Path

import pandas as pd

import download
from compare_gc import YEARS, FORWARD_DAYS, collect, summarize, fmt


def summarize_rows(rows):
    return {
        f"{d}d": summarize([r[f"ret_{d}d"] for r in rows])
        for d in FORWARD_DAYS
    }


def add_summary(lines, title, rows):
    s = summarize_rows(rows)
    lines += ["", title]
    for d in FORWARD_DAYS:
        x = s[f"{d}d"]
        lines.append(
            f"{d}d n={x['n']} win={fmt(x['win_rate'])}% "
            f"avg={fmt(x['avg'])}% median={fmt(x['median'])}%"
        )
    return s


def add_yearly(lines, title, rows):
    lines += ["", title]
    by_year = defaultdict(list)
    for r in rows:
        by_year[pd.Timestamp(r["signal_date"]).year].append(r)

    for year in sorted(by_year):
        s = summarize_rows(by_year[year])
        lines.append(f"{year}:")
        for d in FORWARD_DAYS:
            x = s[f"{d}d"]
            lines.append(
                f"  {d}d n={x['n']} win={fmt(x['win_rate'])}% "
                f"avg={fmt(x['avg'])}% median={fmt(x['median'])}%"
            )


def main():
    os.makedirs("output", exist_ok=True)

    target_codes = download.get_target_codes()
    price_df = download.get_price_history_incremental(
        cache_filename="backtest_prices_prime_5y.csv", years=YEARS
    )
    price_df["Date"] = pd.to_datetime(price_df["Date"])
    rows = collect(price_df, target_codes)

    target = [
        r for r in rows
        if r.get("deep_episode_gc_number") == 3
        and r.get("ma25_slope_bucket") == "rising"
        and r.get("drawdown60_pct") is not None
        and -20.0 < float(r["drawdown60_pct"]) <= -15.0
    ]

    lines = [
        f"3rd GC + MA25 rising + 15-20% drawdown yearly / Prime / {YEARS} years through {price_df['Date'].max().date()}",
        "Condition = exactly 3rd GC in deep-drop episode + MA25 rising (> +0.25% over 5d)",
        "Drawdown filter = GC-day close is 15-20% below rolling 60-day high",
        "Entry = next open after GC; returns = close after 5/10/15 trading days",
        "Year split = calendar year of GC signal; first and last years may be partial.",
    ]

    add_summary(lines, "[Overall]", target)
    add_yearly(lines, "[Yearly]", target)

    text = "\n".join(lines)
    Path("output/gc_third_drawdown_yearly.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
