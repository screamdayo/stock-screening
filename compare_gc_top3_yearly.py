import argparse
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd

import download
from compare_gc import YEARS, FORWARD_DAYS, collect as collect_gc, summarize, fmt, prepare


TESTS = ["ma25_strength", "below_days", "cross_gap"]


def summarize_rows(rows):
    return {
        f"{d}d": summarize([r[f"ret_{d}d"] for r in rows])
        for d in FORWARD_DAYS
    }


def build_feature_map(price_df, target_codes):
    targets = set(map(str, target_codes))
    out = {}

    for code, group in price_df.groupby("Code"):
        code = str(code)
        if code not in targets:
            continue

        g = prepare(group)

        for i in range(25, len(g)):
            ma5 = g.loc[i, "MA5"]
            ma25 = g.loc[i, "MA25"]
            if pd.isna(ma5) or pd.isna(ma25) or not float(ma25) > 0:
                continue

            date = str(pd.Timestamp(g.loc[i, "Date"]).date())

            cross_gap_pct = (float(ma5) / float(ma25) - 1) * 100

            below_days = 0
            j = i - 1
            while j >= 0:
                m5 = g.loc[j, "MA5"]
                m25 = g.loc[j, "MA25"]
                if pd.isna(m5) or pd.isna(m25) or float(m5) > float(m25):
                    break
                below_days += 1
                j -= 1

            slope = g.loc[i, "MA25_5D_SLOPE_PCT"]
            ma25_strength = None if pd.isna(slope) else float(slope)

            out[(code, date)] = {
                "cross_gap_pct": cross_gap_pct,
                "below_days": below_days,
                "ma25_strength": ma25_strength,
            }

    return out


def keep(test, row):
    if test == "ma25_strength":
        v = row.get("ma25_strength")
        return v is not None and v >= 1.0
    if test == "below_days":
        v = row.get("below_days")
        return v is not None and 6 <= v <= 10
    if test == "cross_gap":
        v = row.get("cross_gap_pct")
        return v is not None and v >= 1.0
    raise ValueError(test)


def add_summary(lines, title, rows):
    s = summarize_rows(rows)
    lines += ["", title]
    for d in FORWARD_DAYS:
        x = s[f"{d}d"]
        lines.append(
            f"{d}d n={x['n']} win={fmt(x['win_rate'])}% "
            f"avg={fmt(x['avg'])}% median={fmt(x['median'])}%"
        )


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=TESTS, required=True)
    args = parser.parse_args()
    test = args.test

    os.makedirs("output", exist_ok=True)

    target_codes = download.get_target_codes()
    price_df = download.get_price_history_incremental(
        cache_filename="backtest_prices_prime_5y.csv", years=YEARS
    )
    price_df["Date"] = pd.to_datetime(price_df["Date"])

    gc_rows = collect_gc(price_df, target_codes)
    feature_map = build_feature_map(price_df, target_codes)

    base = [
        dict(r) for r in gc_rows
        if r.get("deep_episode_gc_number") == 3
        and r.get("ma25_slope_bucket") == "rising"
    ]

    selected = []
    for r in base:
        f = feature_map.get((r["code"], r["signal_date"]))
        if not f:
            continue
        r.update(f)
        if keep(test, r):
            selected.append(r)

    labels = {
        "ma25_strength": "MA25 5-day slope >= +1.0%",
        "below_days": "MA5 stayed <= MA25 for 6-10 consecutive trading days before GC",
        "cross_gap": "GC-day MA5 is >= 1.0% above MA25",
    }

    lines = [
        f"GC top-3 yearly check: {test} / Prime / {YEARS} years through {price_df['Date'].max().date()}",
        "Base = exactly 3rd GC in deep-drop episode + MA25 rising (> +0.25% over 5d)",
        f"Extra condition = {labels[test]}",
        "Entry = next open after GC; returns = close after 5/10/15 trading days",
        "Year split = calendar year of GC signal; first and last years may be partial.",
    ]

    add_summary(lines, "[Overall]", selected)
    add_yearly(lines, "[Yearly]", selected)

    text = "\n".join(lines)
    Path(f"output/gc_top3_yearly_{test}.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
