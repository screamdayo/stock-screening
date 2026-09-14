import json
import os
from pathlib import Path

import pandas as pd

import download
from compare_ma25_filter import (
    YEARS, collect_candidates, select_daily, simulate_gakutto,
    summarize, build_windows, fmt,
)


def main():
    os.makedirs("output", exist_ok=True)
    target_codes = download.get_target_codes()
    price_df = download.get_price_history_incremental(
        cache_filename="backtest_prices_prime_5y.csv", years=YEARS
    )
    price_df["Date"] = pd.to_datetime(price_df["Date"])
    candidates, groups = collect_candidates(price_df, target_codes)
    last_date = price_df["Date"].max()
    windows = build_windows(last_date)

    filters = {
        "baseline_current": lambda x: True,
        "abs_gap_le_1pct": lambda x: abs(x["ma_gap"]) <= 1.0,
        "abs_gap_le_2pct": lambda x: abs(x["ma_gap"]) <= 2.0,
        "abs_gap_le_3pct": lambda x: abs(x["ma_gap"]) <= 3.0,
        "bucket_0_to_1pct": lambda x: abs(x["ma_gap"]) <= 1.0,
        "bucket_1_to_2pct": lambda x: 1.0 < abs(x["ma_gap"]) <= 2.0,
        "bucket_2_to_3pct": lambda x: 2.0 < abs(x["ma_gap"]) <= 3.0,
        "bucket_3_to_5pct": lambda x: 3.0 < abs(x["ma_gap"]) <= 5.0,
    }

    report = {"last_date": str(pd.Timestamp(last_date).date()), "filters": {}}
    lines = [
        f"MA5/MA25 distance filter / 5-year backtest through {report['last_date']}",
        "Exit: stop -5%, strict gakutto -> next open, max 15d; entry gap <= +0.5%; volume >= 1.25x",
        "Daily selection: up to 5, nearest MA5/MA25 distance first",
    ]

    for label, fn in filters.items():
        selected = select_daily([x for x in candidates if fn(x)])
        rows = []
        for s in selected:
            r = simulate_gakutto(groups[s["code"]], s["signal_idx"])
            if r is None:
                continue
            r.update({"code": s["code"], "signal_date": s["signal_date"], "ma_gap": s["ma_gap"]})
            rows.append(r)

        overall = summarize(rows)
        yearly = []
        for start, stop in windows:
            subset = [r for r in rows if start <= pd.Timestamp(r["signal_date"]) < stop]
            yearly.append({"start": str(start.date()), "end": str((stop - pd.Timedelta(days=1)).date()), **summarize(subset)})

        report["filters"][label] = {"overall": overall, "yearly": yearly}
        lines.append("")
        lines.append(
            f"[{label}] n={overall['trades']} win={fmt(overall['win_rate'],2)}% "
            f"avg={fmt(overall['avg'])}% PF={fmt(overall['pf'])} hold={fmt(overall['avg_hold'],2)}d"
        )
        for y in yearly:
            lines.append(
                f"{y['start']}..{y['end']} n={y['trades']} win={fmt(y['win_rate'],2)}% "
                f"avg={fmt(y['avg'])}% PF={fmt(y['pf'])}"
            )

    Path("output/ma_distance_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    text = "\n".join(lines)
    Path("output/ma_distance_comparison.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
