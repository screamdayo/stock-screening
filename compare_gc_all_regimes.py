import argparse
import os
from pathlib import Path

import pandas as pd

import download
from compare_gc import YEARS, FORWARD_DAYS, collect, summarize, fmt

SLOPES = ["all", "rising", "flat", "falling"]


def dd_bucket(dd):
    if dd is None or pd.isna(dd):
        return None
    fall = -float(dd)
    if fall < 5:
        return "0-5%"
    if fall < 10:
        return "5-10%"
    if fall < 15:
        return "10-15%"
    if fall < 20:
        return "15-20%"
    if fall < 30:
        return "20-30%"
    return "30%+"


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--slope", choices=SLOPES, required=True)
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)

    target_codes = download.get_target_codes()
    price_df = download.get_price_history_incremental(
        cache_filename="backtest_prices_prime_5y.csv", years=YEARS
    )
    price_df["Date"] = pd.to_datetime(price_df["Date"])

    rows = collect(price_df, target_codes)

    if args.slope != "all":
        rows = [r for r in rows if r.get("ma25_slope_bucket") == args.slope]

    for r in rows:
        r["dd_bucket_fine"] = dd_bucket(r.get("drawdown60_pct"))

    lines = [
        f"All-GC drawdown-regime test / slope={args.slope} / Prime / {YEARS} years through {price_df['Date'].max().date()}",
        "GC = MA5 crosses from <= MA25 to > MA25; entry = next open",
        "No deep-drop episode restriction and no GC-sequence restriction",
        "Drawdown = GC-day close vs rolling 60-day high",
        "Returns = close after 5/10/15 trading days",
        "",
        "[Overall]",
    ]

    add_group(lines, "all", rows)

    lines += ["", "[By current drawdown from 60-day high]"]
    order = ["0-5%", "5-10%", "10-15%", "15-20%", "20-30%", "30%+"]
    for bucket in order:
        sub = [r for r in rows if r.get("dd_bucket_fine") == bucket]
        add_group(lines, bucket, sub)

    text = "\n".join(lines)
    Path(f"output/gc_all_drawdown_{args.slope}.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
