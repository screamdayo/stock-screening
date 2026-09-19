import argparse
import os
from pathlib import Path

import pandas as pd

import download
from compare_gc import YEARS, FORWARD_DAYS, collect as collect_gc, summarize, fmt, prepare


TESTS = ["volume", "candle", "cross_gap", "below_days", "ma25_strength"]


def summarize_rows(rows):
    return {
        f"{d}d": summarize([r[f"ret_{d}d"] for r in rows])
        for d in FORWARD_DAYS
    }


def add_group(lines, label, rows):
    s = summarize_rows(rows)
    parts = []
    for d in FORWARD_DAYS:
        x = s[f"{d}d"]
        parts.append(
            f"{d}d n={x['n']} win={fmt(x['win_rate'])}% "
            f"avg={fmt(x['avg'])}% median={fmt(x['median'])}%"
        )
    lines.append(f"{label}: " + " | ".join(parts))


def build_feature_map(price_df, target_codes):
    targets = set(map(str, target_codes))
    out = {}

    for code, group in price_df.groupby("Code"):
        code = str(code)
        if code not in targets:
            continue

        g = prepare(group)
        if "Vo" in g.columns:
            g["Vo"] = pd.to_numeric(g["Vo"], errors="coerce")
        else:
            g["Vo"] = pd.NA

        for i in range(25, len(g)):
            ma5 = g.loc[i, "MA5"]
            ma25 = g.loc[i, "MA25"]
            if pd.isna(ma5) or pd.isna(ma25) or not ma25 > 0:
                continue

            date = str(pd.Timestamp(g.loc[i, "Date"]).date())
            o = pd.to_numeric(g.loc[i, "O"], errors="coerce")
            c = pd.to_numeric(g.loc[i, "C"], errors="coerce")

            candle_pct = None
            if pd.notna(o) and pd.notna(c) and float(o) > 0:
                candle_pct = (float(c) / float(o) - 1) * 100

            volume_ratio = None
            if i >= 1:
                prev_v = g.loc[i - 1, "Vo"]
                cur_v = g.loc[i, "Vo"]
                if pd.notna(prev_v) and pd.notna(cur_v) and float(prev_v) > 0:
                    volume_ratio = float(cur_v) / float(prev_v)

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
                "volume_ratio": volume_ratio,
                "candle_pct": candle_pct,
                "cross_gap_pct": cross_gap_pct,
                "below_days": below_days,
                "ma25_strength": ma25_strength,
            }

    return out


def bucket(test, row):
    if test == "volume":
        v = row.get("volume_ratio")
        if v is None:
            return None
        if v < 1.0:
            return "<1.0x"
        if v < 1.2:
            return "1.0-1.2x"
        if v < 1.5:
            return "1.2-1.5x"
        return "1.5x+"

    if test == "candle":
        v = row.get("candle_pct")
        if v is None:
            return None
        if v <= 0:
            return "<=0%"
        if v < 1:
            return "0-1%"
        if v < 2:
            return "1-2%"
        return "2%+"

    if test == "cross_gap":
        v = row.get("cross_gap_pct")
        if v is None:
            return None
        if v < 0.25:
            return "0-0.25%"
        if v < 0.5:
            return "0.25-0.5%"
        if v < 1.0:
            return "0.5-1.0%"
        return "1.0%+"

    if test == "below_days":
        v = row.get("below_days")
        if v is None:
            return None
        if v <= 5:
            return "1-5d"
        if v <= 10:
            return "6-10d"
        if v <= 20:
            return "11-20d"
        return "21d+"

    if test == "ma25_strength":
        v = row.get("ma25_strength")
        if v is None:
            return None
        if v < 0.5:
            return "0.25-0.5%"
        if v < 1.0:
            return "0.5-1.0%"
        return "1.0%+"

    raise ValueError(test)


def order_for(test):
    return {
        "volume": ["<1.0x", "1.0-1.2x", "1.2-1.5x", "1.5x+"],
        "candle": ["<=0%", "0-1%", "1-2%", "2%+"],
        "cross_gap": ["0-0.25%", "0.25-0.5%", "0.5-1.0%", "1.0%+"],
        "below_days": ["1-5d", "6-10d", "11-20d", "21d+"],
        "ma25_strength": ["0.25-0.5%", "0.5-1.0%", "1.0%+"],
    }[test]


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

    # Keep one stable base while comparing individual filters:
    # exactly 3rd GC in a deep-drop episode + MA25 rising.
    base = [
        dict(r) for r in gc_rows
        if r.get("deep_episode_gc_number") == 3
        and r.get("ma25_slope_bucket") == "rising"
    ]

    enriched = []
    for r in base:
        f = feature_map.get((r["code"], r["signal_date"]))
        if not f:
            continue
        r.update(f)
        r["bucket"] = bucket(test, r)
        if r["bucket"] is not None:
            enriched.append(r)

    labels = {
        "volume": "GC-day volume ratio vs previous day",
        "candle": "GC-day candle return (close/open - 1)",
        "cross_gap": "GC-day MA5 above MA25 gap",
        "below_days": "Consecutive trading days MA5 stayed <= MA25 before GC",
        "ma25_strength": "MA25 5-day slope strength",
    }

    lines = [
        f"Parallel GC condition test: {test} / Prime / {YEARS} years through {price_df['Date'].max().date()}",
        "Base = exactly 3rd GC in deep-drop episode + MA25 rising (> +0.25% over 5d)",
        labels[test],
        "Entry = next open after GC; returns = close after 5/10/15 trading days",
        "",
        "[Baseline]",
    ]
    add_group(lines, "all", enriched)
    lines += ["", "[By bucket]"]

    for label in order_for(test):
        rows = [r for r in enriched if r["bucket"] == label]
        add_group(lines, label, rows)

    text = "\n".join(lines)
    out = Path(f"output/gc_parallel_{test}.txt")
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
