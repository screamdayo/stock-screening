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


def add_section(lines, title, summary):
    lines += ["", title]
    for d in FORWARD_DAYS:
        s = summary[f"{d}d"]
        lines.append(
            f"{d}d n={s['n']} win={fmt(s['win_rate'])}% avg={fmt(s['avg'])}% median={fmt(s['median'])}%"
        )


def main():
    os.makedirs("output", exist_ok=True)
    target_codes = download.get_target_codes()
    price_df = download.get_price_history_incremental(
        cache_filename="backtest_prices_prime_5y.csv", years=YEARS
    )
    price_df["Date"] = pd.to_datetime(price_df["Date"])
    rows = collect(price_df, target_codes)

    episode_rows = [r for r in rows if r.get("deep_episode_gc_number") is not None]
    first_rows = [r for r in episode_rows if r["deep_episode_gc_number"] == 1]
    skip_first_rows = [r for r in episode_rows if r["deep_episode_gc_number"] >= 2]

    episode_falling = [r for r in episode_rows if r.get("ma25_slope_bucket") == "falling"]
    first_falling = [r for r in episode_falling if r["deep_episode_gc_number"] == 1]
    skip_first_falling = [r for r in episode_falling if r["deep_episode_gc_number"] >= 2]

    all_summary = summarize_rows(episode_rows)
    first_summary = summarize_rows(first_rows)
    skip_summary = summarize_rows(skip_first_rows)
    all_falling_summary = summarize_rows(episode_falling)
    first_falling_summary = summarize_rows(first_falling)
    skip_falling_summary = summarize_rows(skip_first_falling)

    lines = [
        f"GC skip-first comparison / Prime / {YEARS} years through {price_df['Date'].max().date()}",
        "Deep-drop episode = starts at 15%+ below 60d high, ends after recovery to within 5%",
        "Entry = next open after GC; returns = close after 5/10/15 trading days",
    ]
    add_section(lines, "[All GCs inside deep-drop episodes]", all_summary)
    add_section(lines, "[1st GC only]", first_summary)
    add_section(lines, "[Skip 1st: 2nd GC and later only]", skip_summary)
    add_section(lines, "[All episode GC when MA25 still falling]", all_falling_summary)
    add_section(lines, "[1st GC only, MA25 falling]", first_falling_summary)
    add_section(lines, "[Skip 1st, MA25 falling]", skip_falling_summary)

    lines += ["", "[Improvement from skipping 1st GC]"]
    for d in FORWARD_DAYS:
        a = all_summary[f"{d}d"]
        s = skip_summary[f"{d}d"]
        lines.append(
            f"{d}d: win {fmt(a['win_rate'])}% -> {fmt(s['win_rate'])}% "
            f"({fmt(s['win_rate'] - a['win_rate'])}pt), "
            f"avg {fmt(a['avg'])}% -> {fmt(s['avg'])}% "
            f"({fmt(s['avg'] - a['avg'])}pt)"
        )

    text = "\n".join(lines)
    Path("output/gc_skip_first_comparison.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
