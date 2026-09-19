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
            f"{d}d n={x['n']} win={fmt(x['win_rate'])}% "
            f"avg={fmt(x['avg'])}% median={fmt(x['median'])}%"
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

    rising = [
        r for r in rows
        if r.get("deep_episode_gc_number") is not None
        and r.get("ma25_slope_bucket") == "rising"
    ]

    second = [r for r in rising if r["deep_episode_gc_number"] == 2]
    third = [r for r in rising if r["deep_episode_gc_number"] == 3]
    fourth_plus = [r for r in rising if r["deep_episode_gc_number"] >= 4]

    lines = [
        f"GC 2nd vs 3rd vs 4th+ with MA25 rising / Prime / {YEARS} years through {price_df['Date'].max().date()}",
        "Deep-drop episode = starts at 15%+ below 60d high, ends after recovery to within 5%",
        "MA25 rising = 5-day slope > +0.25%",
        "Entry = next open after GC; returns = close after 5/10/15 trading days",
    ]

    s2 = add_section(lines, "[2nd GC + MA25 rising]", second)
    s3 = add_section(lines, "[3rd GC + MA25 rising]", third)
    s4 = add_section(lines, "[4th+ GC + MA25 rising]", fourth_plus)

    lines += ["", "[Direct comparison: 2nd vs 3rd vs 4th+]"]
    for d in FORWARD_DAYS:
        a = s2[f"{d}d"]
        b = s3[f"{d}d"]
        c = s4[f"{d}d"]
        lines.append(
            f"{d}d: "
            f"2nd n={a['n']} win={fmt(a['win_rate'])}% avg={fmt(a['avg'])}% median={fmt(a['median'])}% | "
            f"3rd n={b['n']} win={fmt(b['win_rate'])}% avg={fmt(b['avg'])}% median={fmt(b['median'])}% | "
            f"4th+ n={c['n']} win={fmt(c['win_rate'])}% avg={fmt(c['avg'])}% median={fmt(c['median'])}%"
        )

    lines += ["", "[3rd minus 2nd / 4th+ minus 3rd]"]
    for d in FORWARD_DAYS:
        a = s2[f"{d}d"]
        b = s3[f"{d}d"]
        c = s4[f"{d}d"]
        lines.append(
            f"{d}d: "
            f"3rd-2nd win={fmt(b['win_rate'] - a['win_rate'])}pt avg={fmt(b['avg'] - a['avg'])}pt | "
            f"4th+-3rd win={fmt(c['win_rate'] - b['win_rate'])}pt avg={fmt(c['avg'] - b['avg'])}pt"
        )

    text = "\n".join(lines)
    Path("output/gc_third_vs_fourth.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
