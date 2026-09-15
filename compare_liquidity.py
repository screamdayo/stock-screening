import json
import os
from pathlib import Path

import pandas as pd

import download
from forward_test import _prepare, _base_features
from compare_ma25_filter import YEARS, select_daily, simulate_gakutto, summarize, build_windows, fmt

VOLUME_RATIO_MIN = 1.25
MAX_ENTRY_GAP_PCT = 0.5


def collect_candidates(price_df, target_codes):
    groups = {}
    raw = []
    targets = set(map(str, target_codes))

    for code, group in price_df.groupby("Code"):
        code = str(code)
        if code not in targets:
            continue

        g = _prepare(group)
        groups[code] = g
        g["turnover"] = g["C"] * g["Vo"]
        g["avg_turnover_20"] = g["turnover"].rolling(20).mean()

        for i in range(25, len(g) - 1):
            f = _base_features(g, i)
            if not f:
                continue
            if f.get("volume_ratio") is None or f["volume_ratio"] < VOLUME_RATIO_MIN:
                continue

            signal_close = float(g.C.iloc[i])
            next_open = float(g.O.iloc[i + 1])
            if not signal_close > 0 or not next_open > 0:
                continue
            entry_gap = (next_open / signal_close - 1) * 100
            if entry_gap > MAX_ENTRY_GAP_PCT:
                continue

            avg_turnover_20 = g.avg_turnover_20.iloc[i]
            if pd.isna(avg_turnover_20):
                continue

            raw.append({
                "code": code,
                "signal_idx": i,
                "signal_date": g.Date.iloc[i],
                "ma_gap": float(f["ma5_vs_ma25_pct"]),
                "avg_turnover_20": float(avg_turnover_20),
            })

    return raw, groups


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
        "baseline": lambda x: True,
        "avg_turnover_ge_50m": lambda x: x["avg_turnover_20"] >= 50_000_000,
        "avg_turnover_ge_100m": lambda x: x["avg_turnover_20"] >= 100_000_000,
        "avg_turnover_ge_300m": lambda x: x["avg_turnover_20"] >= 300_000_000,
        "avg_turnover_ge_500m": lambda x: x["avg_turnover_20"] >= 500_000_000,
    }

    report = {"last_date": str(pd.Timestamp(last_date).date()), "filters": {}}
    lines = [
        f"Liquidity filter / 5-year backtest through {report['last_date']}",
        "Metric: 20-day average turnover = close x volume",
        "Exit: stop -5%, strict gakutto -> next open, max 15d; entry gap <= +0.5%; volume ratio >= 1.25x",
        "Daily selection: up to 5, nearest MA5/MA25 distance first after liquidity filtering",
    ]

    for label, fn in filters.items():
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
                "avg_turnover_20": s["avg_turnover_20"],
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

    Path("output/liquidity_filter_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    text = "\n".join(lines)
    Path("output/liquidity_filter_comparison.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
