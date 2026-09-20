import os
from collections import defaultdict
from pathlib import Path

import pandas as pd

import download
from compare_gc import YEARS, FORWARD_DAYS, collect as collect_gc, summarize, fmt, prepare


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
            if pd.isna(ma5) or pd.isna(ma25):
                continue

            date = str(pd.Timestamp(g.loc[i, "Date"]).date())

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
            out[(code, date)] = {
                "below_days": below_days,
                "ma25_strength": None if pd.isna(slope) else float(slope),
            }

    return out


def add_summary(lines, title, rows):
    lines += ["", title]
    s = summarize_rows(rows)
    for d in FORWARD_DAYS:
        x = s[f"{d}d"]
        lines.append(
            f"{d}d n={x['n']} win={fmt(x['win_rate'])}% "
            f"avg={fmt(x['avg'])}% median={fmt(x['median'])}%"
        )


def add_yearly(lines, rows):
    lines += ["", "[Yearly]"]
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
        if 6 <= r["below_days"] <= 10 and r["ma25_strength"] is not None and r["ma25_strength"] >= 1.0:
            selected.append(r)

    lines = [
        f"GC combo / Prime / {YEARS} years through {price_df['Date'].max().date()}",
        "Base = exactly 3rd GC in deep-drop episode + MA25 rising",
        "Extra = MA5 stayed <= MA25 for 6-10 trading days before GC",
        "Extra = MA25 5-day slope >= +1.0%",
        "Entry = next open after GC; returns = close after 5/10/15 trading days",
        "Year split = calendar year of GC signal; first and last years may be partial.",
    ]

    add_summary(lines, "[Overall]", selected)
    add_yearly(lines, selected)

    text = "\n".join(lines)
    Path("output/gc_combo_below6_10_ma25plus1.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
