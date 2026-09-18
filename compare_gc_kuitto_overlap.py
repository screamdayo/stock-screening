import os
from bisect import bisect_left
from pathlib import Path

import pandas as pd

import download
from compare_gc import YEARS, FORWARD_DAYS, collect as collect_gc, summarize, fmt
from strategies import kuitto_pullback_auto as kuitto

WINDOWS = [3, 5, 10, 20]
TURNOVER_MIN = kuitto.AVG_TURNOVER_20_MIN


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

            # Today's production rule uses 20d average turnover >= 500m.
            # Apply it consistently over the full 5-year history for this comparison.
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


def main():
    os.makedirs("output", exist_ok=True)

    target_codes = download.get_target_codes()
    price_df = download.get_price_history_incremental(
        cache_filename="backtest_prices_prime_5y.csv", years=YEARS
    )
    price_df["Date"] = pd.to_datetime(price_df["Date"])

    gc_rows = collect_gc(price_df, target_codes)
    kuitto_idxs, date_to_idx = collect_kuitto_indices(price_df, target_codes)

    # Previous best GC condition:
    # deep-drop episode, 2nd GC or later, MA25 5d slope is rising (> +0.25%).
    best_gc = [
        r for r in gc_rows
        if r.get("deep_episode_gc_number") is not None
        and r["deep_episode_gc_number"] >= 2
        and r.get("ma25_slope_bucket") == "rising"
    ]

    for r in best_gc:
        code = r["code"]
        gc_idx = date_to_idx.get(code, {}).get(r["signal_date"])
        r["gc_idx"] = gc_idx
        for w in WINDOWS:
            r[f"kuitto_prior_{w}d"] = (
                gc_idx is not None
                and has_prior_kuitto(kuitto_idxs.get(code, []), gc_idx, w)
            )

    baseline = [r for r in best_gc if r.get("gc_idx") is not None]

    lines = [
        f"GC + Kuitto overlap / Prime / {YEARS} years through {price_df['Date'].max().date()}",
        "GC condition = deep-drop episode + 2nd GC or later + MA25 rising (> +0.25% over 5d)",
        "Kuitto = current kuitto_pullback_auto conditions + 20d avg turnover >= 500m applied to all years",
        "Overlap = a Kuitto signal occurred in the same stock BEFORE GC within N trading days",
        "Entry = next open after GC; returns = close after 5/10/15 trading days",
        "Note: exact same-day overlap is structurally impossible because Kuitto requires MA5 <= MA25 while GC requires MA5 > MA25.",
    ]

    base_s = add_section(lines, "[Best GC baseline]", baseline)

    for w in WINDOWS:
        rows = [r for r in baseline if r[f"kuitto_prior_{w}d"]]
        s = add_section(lines, f"[Best GC + prior Kuitto within {w} trading days]", rows)
        lines += [f"[Improvement vs baseline: {w}d window]"]
        for d in FORWARD_DAYS:
            b = base_s[f"{d}d"]
            x = s[f"{d}d"]
            if x["n"] == 0:
                lines.append(f"{d}d: n=0")
                continue
            lines.append(
                f"{d}d: n {b['n']} -> {x['n']}, "
                f"win {fmt(b['win_rate'])}% -> {fmt(x['win_rate'])}% "
                f"({fmt(x['win_rate'] - b['win_rate'])}pt), "
                f"avg {fmt(b['avg'])}% -> {fmt(x['avg'])}% "
                f"({fmt(x['avg'] - b['avg'])}pt), "
                f"median {fmt(b['median'])}% -> {fmt(x['median'])}%"
            )

    text = "\n".join(lines)
    Path("output/gc_kuitto_overlap.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
