import os
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import pandas as pd

import download
from compare_gc import YEARS, FORWARD_DAYS, collect as collect_gc, summarize, fmt
from strategies import kuitto_pullback_auto as kuitto

WINDOWS = [3, 5]
TURNOVER_MIN = kuitto.AVG_TURNOVER_20_MIN


def summarize_rows(rows):
    return {
        f"{d}d": summarize([r[f"ret_{d}d"] for r in rows])
        for d in FORWARD_DAYS
    }


def collect_kuitto_indices(price_df, target_codes):
    targets = set(map(str, target_codes))
    signal_indices = {}
    date_to_idx = {}

    for code, group in price_df.groupby("Code"):
        code = str(code)
        if code not in targets:
            continue

        g = kuitto._prepare(group)
        if len(g) <= kuitto.MA_LONG:
            continue

        date_to_idx[code] = {
            str(pd.Timestamp(dt).date()): i
            for i, dt in enumerate(g["Date"])
        }

        idxs = []
        for idx in range(kuitto.MA_LONG, len(g)):
            f = kuitto._features(g, idx)
            if not f:
                continue

            avg_turnover = g["AVG_TURNOVER_20"].iloc[idx]
            if pd.isna(avg_turnover) or float(avg_turnover) < TURNOVER_MIN:
                continue

            idxs.append(idx)

        signal_indices[code] = idxs

    return signal_indices, date_to_idx


def has_prior_kuitto(kuitto_idxs, gc_idx, window):
    if not kuitto_idxs:
        return False
    pos = bisect_left(kuitto_idxs, gc_idx) - 1
    if pos < 0:
        return False
    diff = gc_idx - kuitto_idxs[pos]
    return 1 <= diff <= window


def attach_overlap(rows, kuitto_idxs, date_to_idx):
    out = []
    for r in rows:
        x = dict(r)
        code = x["code"]
        gc_idx = date_to_idx.get(code, {}).get(x["signal_date"])
        if gc_idx is None:
            continue

        x["signal_year"] = pd.Timestamp(x["signal_date"]).year
        for w in WINDOWS:
            x[f"kuitto_prior_{w}d"] = has_prior_kuitto(
                kuitto_idxs.get(code, []), gc_idx, w
            )
        out.append(x)
    return out


def append_summary(lines, title, rows):
    s = summarize_rows(rows)
    lines += ["", title]
    for d in FORWARD_DAYS:
        x = s[f"{d}d"]
        lines.append(
            f"{d}d n={x['n']} win={fmt(x['win_rate'])}% "
            f"avg={fmt(x['avg'])}% median={fmt(x['median'])}%"
        )
    return s


def append_yearly(lines, title, rows):
    lines += ["", title]
    by_year = defaultdict(list)
    for r in rows:
        by_year[r["signal_year"]].append(r)

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
    kuitto_idxs, date_to_idx = collect_kuitto_indices(price_df, target_codes)

    common = [
        r for r in gc_rows
        if r.get("deep_episode_gc_number") is not None
        and r["deep_episode_gc_number"] >= 2
    ]

    strict = attach_overlap(
        [r for r in common if r.get("ma25_slope_bucket") == "rising"],
        kuitto_idxs,
        date_to_idx,
    )
    relaxed = attach_overlap(
        [r for r in common if r.get("ma25_slope_bucket") in ("flat", "rising")],
        kuitto_idxs,
        date_to_idx,
    )

    lines = [
        f"GC + Kuitto yearly check / Prime / {YEARS} years through {price_df['Date'].max().date()}",
        "GC common condition = deep-drop episode + 2nd GC or later",
        "Strict = MA25 rising (> +0.25% over 5d)",
        "Relaxed = MA25 flat or rising (>= -0.25% over 5d)",
        "Kuitto = current kuitto_pullback_auto + 20d avg turnover >= 500m over all years",
        "Overlap = prior Kuitto in same stock within 3 or 5 trading days before GC",
        "Entry = next open after GC; returns = close after 5/10/15 trading days",
        "Year split = calendar year of GC signal; first and last calendar years may be partial.",
    ]

    for w in WINDOWS:
        strict_rows = [r for r in strict if r[f"kuitto_prior_{w}d"]]
        relaxed_rows = [r for r in relaxed if r[f"kuitto_prior_{w}d"]]

        append_summary(lines, f"[Strict overall: prior Kuitto within {w}d]", strict_rows)
        append_yearly(lines, f"[Strict yearly: prior Kuitto within {w}d]", strict_rows)

        append_summary(lines, f"[Relaxed overall: prior Kuitto within {w}d]", relaxed_rows)
        append_yearly(lines, f"[Relaxed yearly: prior Kuitto within {w}d]", relaxed_rows)

    text = "\n".join(lines)
    Path("output/gc_kuitto_yearly.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
