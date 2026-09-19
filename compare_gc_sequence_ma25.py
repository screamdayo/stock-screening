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

    episode = [r for r in rows if r.get("deep_episode_gc_number") is not None]

    lines = [
        f"GC sequence x MA25 slope / Prime / {YEARS} years through {price_df['Date'].max().date()}",
        "Deep-drop episode = starts at 15%+ below 60d high, ends after recovery to within 5%",
        "Sequence = 1st / 2nd / 3rd+ GC inside same deep-drop episode",
        "MA25 slope = 5-day change: falling < -0.25%, flat -0.25%..+0.25%, rising > +0.25%",
        "Entry = next open after GC; returns = close after 5/10/15 trading days",
    ]

    for slope in ["falling", "flat", "rising"]:
        slope_rows = [r for r in episode if r.get("ma25_slope_bucket") == slope]
        add_section(lines, f"[All sequence, MA25 {slope}]", slope_rows)

        for seq_label, fn in [
            ("1st", lambda r: r["deep_episode_gc_number"] == 1),
            ("2nd", lambda r: r["deep_episode_gc_number"] == 2),
            ("3rd+", lambda r: r["deep_episode_gc_number"] >= 3),
        ]:
            subset = [r for r in slope_rows if fn(r)]
            add_section(lines, f"[{seq_label} GC + MA25 {slope}]", subset)

    # Focus comparison requested: 2nd vs 3rd+ under MA25 rising.
    rising = [r for r in episode if r.get("ma25_slope_bucket") == "rising"]
    second = [r for r in rising if r["deep_episode_gc_number"] == 2]
    third_plus = [r for r in rising if r["deep_episode_gc_number"] >= 3]

    second_s = summarize_rows(second)
    third_s = summarize_rows(third_plus)

    lines += ["", "[Focus: MA25 rising, 2nd vs 3rd+]"]
    for d in FORWARD_DAYS:
        a = second_s[f"{d}d"]
        b = third_s[f"{d}d"]
        lines.append(
            f"{d}d: 2nd n={a['n']} win={fmt(a['win_rate'])}% avg={fmt(a['avg'])}% median={fmt(a['median'])}% | "
            f"3rd+ n={b['n']} win={fmt(b['win_rate'])}% avg={fmt(b['avg'])}% median={fmt(b['median'])}% | "
            f"diff win={fmt(b['win_rate'] - a['win_rate'])}pt avg={fmt(b['avg'] - a['avg'])}pt"
        )

    text = "\n".join(lines)
    Path("output/gc_sequence_ma25.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
