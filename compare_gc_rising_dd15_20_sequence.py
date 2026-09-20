import os
from pathlib import Path

import pandas as pd

import download
from compare_gc import YEARS, FORWARD_DAYS, collect, summarize, fmt


def seq_bucket(n):
    if n is None:
        return "outside_deep_episode"
    if n == 1:
        return "1st"
    if n == 2:
        return "2nd"
    if n == 3:
        return "3rd"
    return "4th+"


def add_group(lines, label, rows):
    parts = []
    for d in FORWARD_DAYS:
        s = summarize([r[f"ret_{d}d"] for r in rows])
        parts.append(
            f"{d}d n={s['n']} win={fmt(s['win_rate'])}% "
            f"avg={fmt(s['avg'])}% median={fmt(s['median'])}%"
        )
    lines.append(f"{label}: " + " | ".join(parts))


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
        if r.get("ma25_slope_bucket") == "rising"
        and r.get("drawdown60_pct") is not None
        and -20.0 < float(r["drawdown60_pct"]) <= -15.0
    ]

    for r in target:
        r["seq_exact_bucket"] = seq_bucket(r.get("deep_episode_gc_number"))

    lines = [
        f"MA25 rising + 15-20% drawdown GC by sequence / Prime / {YEARS} years through {price_df['Date'].max().date()}",
        "Condition = GC-day MA25 rising (> +0.25% over 5d) AND close is 15-20% below rolling 60-day high",
        "GC = MA5 crosses from <= MA25 to > MA25; entry = next open",
        "Sequence = GC number within deep-drop episode that began at 15%+ drawdown and resets after recovery above -5%",
        "Returns = close after 5/10/15 trading days",
        "",
        "[Overall target]",
    ]
    add_group(lines, "all", target)

    lines += ["", "[By GC sequence]"]
    for label in ["1st", "2nd", "3rd", "4th+", "outside_deep_episode"]:
        sub = [r for r in target if r["seq_exact_bucket"] == label]
        add_group(lines, label, sub)

    text = "\n".join(lines)
    Path("output/gc_rising_dd15_20_sequence.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
