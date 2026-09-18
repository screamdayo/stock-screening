import os
from pathlib import Path

import pandas as pd

import download
from compare_gc import YEARS, FORWARD_DAYS, collect, summarize, fmt


def summarize_rows(rows):
    return {
        f"{d}d": summarize([r[f"ret_{d}d"] for r in rows])
        for d in FORWARD_DAYS
    }


def add_section(lines, title, rows):
    s = summarize_rows(rows)
    lines += ["", title]
    for d in FORWARD_DAYS:
        x = s[f"{d}d"]
        lines.append(
            f"{d}d n={x['n']} win={fmt(x['win_rate'])}% avg={fmt(x['avg'])}% median={fmt(x['median'])}%"
        )
    return s


def main():
    os.makedirs("output", exist_ok=True)

    target_codes = download.get_target_codes()
    price_df = download.get_price_history_incremental(
        cache_filename="backtest_prices_prime_5y.csv", years=YEARS
    )
    price_df["Date"] = pd.to_datetime(price_df["Date"])
    rows = collect(price_df, target_codes)

    episode = [r for r in rows if r.get("deep_episode_gc_number") is not None]
    skip_first = [r for r in episode if r["deep_episode_gc_number"] >= 2]

    skip_falling = [r for r in skip_first if r.get("ma25_slope_bucket") == "falling"]
    skip_flat = [r for r in skip_first if r.get("ma25_slope_bucket") == "flat"]
    skip_rising = [r for r in skip_first if r.get("ma25_slope_bucket") == "rising"]
    skip_flat_or_rising = [
        r for r in skip_first if r.get("ma25_slope_bucket") in ("flat", "rising")
    ]

    lines = [
        f"GC skip-first + MA25 slope comparison / Prime / {YEARS} years through {price_df['Date'].max().date()}",
        "Deep-drop episode = starts at 15%+ below 60d high, ends after recovery to within 5%",
        "Condition = 2nd GC or later in episode; entry = next open after GC",
        "MA25 slope = 5-day change: falling < -0.25%, flat -0.25%..+0.25%, rising > +0.25%",
        "Returns = close after 5/10/15 trading days",
    ]

    base = add_section(lines, "[Skip 1st: all MA25 slopes]", skip_first)
    falling = add_section(lines, "[Skip 1st + MA25 falling]", skip_falling)
    flat = add_section(lines, "[Skip 1st + MA25 flat]", skip_flat)
    rising = add_section(lines, "[Skip 1st + MA25 rising]", skip_rising)
    fr = add_section(lines, "[Skip 1st + MA25 flat OR rising]", skip_flat_or_rising)

    lines += ["", "[Improvement vs skip-1st baseline]"]
    for d in FORWARD_DAYS:
        b = base[f"{d}d"]
        x = fr[f"{d}d"]
        lines.append(
            f"{d}d: n {b['n']} -> {x['n']}, "
            f"win {fmt(b['win_rate'])}% -> {fmt(x['win_rate'])}% "
            f"({fmt(x['win_rate'] - b['win_rate'])}pt), "
            f"avg {fmt(b['avg'])}% -> {fmt(x['avg'])}% "
            f"({fmt(x['avg'] - b['avg'])}pt), "
            f"median {fmt(b['median'])}% -> {fmt(x['median'])}%"
        )

    text = "\n".join(lines)
    Path("output/gc_skip_first_slope_comparison.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
