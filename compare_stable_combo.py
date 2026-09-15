import json
import os
from pathlib import Path

import pandas as pd

import download
from compare_ma25_filter import (
    YEARS, collect_candidates, select_daily, simulate_gakutto,
    summarize, build_windows, fmt,
)


def run_case(label, candidates, groups, windows, fn):
    selected = select_daily([x for x in candidates if fn(x)])
    rows = []
    for s in selected:
        r = simulate_gakutto(groups[s["code"]], s["signal_idx"])
        if r is None:
            continue
        r.update({
            "code": s["code"],
            "signal_date": s["signal_date"],
            "ma_gap": s["ma_gap"],
            "ma25_slope_5d": s["ma25_slope_5d"],
        })
        rows.append(r)

    overall = summarize(rows)
    yearly = []
    for start, stop in windows:
        subset = [r for r in rows if start <= pd.Timestamp(r["signal_date"]) < stop]
        yearly.append({
            "start": str(start.date()),
            "end": str((stop - pd.Timedelta(days=1)).date()),
            **summarize(subset),
        })
    return {"overall": overall, "yearly": yearly}


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

    flat5 = lambda x: x["ma25_slope_5d"] is not None and x["ma25_slope_5d"] >= -0.5
    rising1 = lambda x: x["ma25_slope_1d"] is not None and x["ma25_slope_1d"] > 0
    gap01 = lambda x: abs(x["ma_gap"]) <= 1.0
    gap02 = lambda x: abs(x["ma_gap"]) <= 2.0
    gap35 = lambda x: 3.0 < abs(x["ma_gap"]) <= 5.0

    cases = {
        "baseline": lambda x: True,
        "ma25_flat_or_rising_5d": flat5,
        "distance_3_to_5pct": gap35,
        "combo_flat5_and_3to5": lambda x: flat5(x) and gap35(x),
        "combo_flat5_and_le2": lambda x: flat5(x) and gap02(x),
        "combo_flat5_and_le1": lambda x: flat5(x) and gap01(x),
        "combo_rising1d_and_3to5": lambda x: rising1(x) and gap35(x),
    }

    report = {
        "last_date": str(pd.Timestamp(last_date).date()),
        "rules": "stop -5%, strict gakutto next open, max 15d, gap <= +0.5%, volume >=1.25x, daily max5",
        "cases": {},
    }
    lines = [
        f"Stable-condition combo / 5-year backtest through {report['last_date']}",
        "Exit: stop -5%, strict gakutto -> next open, max 15d; entry gap <= +0.5%; volume >= 1.25x",
        "Daily selection: up to 5 after filtering, nearest MA5/MA25 distance first",
    ]

    for label, fn in cases.items():
        res = run_case(label, candidates, groups, windows, fn)
        report["cases"][label] = res
        s = res["overall"]
        lines.append("")
        lines.append(
            f"[{label}] n={s['trades']} win={fmt(s['win_rate'],2)}% avg={fmt(s['avg'])}% "
            f"PF={fmt(s['pf'])} hold={fmt(s['avg_hold'],2)}d"
        )
        for y in res["yearly"]:
            lines.append(
                f"{y['start']}..{y['end']} n={y['trades']} win={fmt(y['win_rate'],2)}% "
                f"avg={fmt(y['avg'])}% PF={fmt(y['pf'])}"
            )

    Path("output/stable_combo_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    text = "\n".join(lines)
    Path("output/stable_combo_comparison.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
