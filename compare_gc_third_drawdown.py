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


def dd_bin(dd):
    if dd is None or pd.isna(dd):
        return None
    fall = -float(dd)
    if 15 <= fall < 20:
        return "15-20%"
    if 20 <= fall < 30:
        return "20-30%"
    if fall >= 30:
        return "30%+"
    return None


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

    target = [
        r for r in rows
        if r.get("deep_episode_gc_number") == 3
        and r.get("ma25_slope_bucket") == "rising"
    ]

    for r in target:
        r["third_gc_drawdown_bin"] = dd_bin(r.get("drawdown60_pct"))

    lines = [
        f"3rd GC + MA25 rising by drawdown / Prime / {YEARS} years through {price_df['Date'].max().date()}",
        "Condition = exactly 3rd GC in deep-drop episode + MA25 rising (> +0.25% over 5d)",
        "Drawdown = current GC-day close vs rolling 60-day high",
        "Bins = 15-20%, 20-30%, 30%+ below 60-day high",
        "Entry = next open after GC; returns = close after 5/10/15 trading days",
        "Note: target episode began at 15%+ drawdown, but by the 3rd GC price may have recovered above -15%; those cases are shown separately.",
    ]

    base = add_section(lines, "[All 3rd GC + MA25 rising]", target)

    recovered = [r for r in target if dd_bin(r.get("drawdown60_pct")) is None]
    add_section(lines, "[Recovered to less than 15% below 60d high by 3rd GC]", recovered)

    summaries = {}
    for label in ["15-20%", "20-30%", "30%+"]:
        subset = [r for r in target if r.get("third_gc_drawdown_bin") == label]
        summaries[label] = add_section(lines, f"[Drawdown {label}]", subset)

    lines += ["", "[Direct comparison by drawdown bin]"]
    for d in FORWARD_DAYS:
        parts = []
        for label in ["15-20%", "20-30%", "30%+"]:
            s = summaries[label][f"{d}d"]
            parts.append(
                f"{label} n={s['n']} win={fmt(s['win_rate'])}% avg={fmt(s['avg'])}% median={fmt(s['median'])}%"
            )
        lines.append(f"{d}d: " + " | ".join(parts))

    text = "\n".join(lines)
    Path("output/gc_third_drawdown_bins.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
