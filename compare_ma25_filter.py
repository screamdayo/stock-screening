import json
import os
from pathlib import Path

import pandas as pd

import download
from forward_test import _prepare, _base_features, _strict_gakutto

YEARS = 5
VOLUME_RATIO_MIN = 1.25
MAX_ENTRY_GAP_PCT = 0.5
MAX_DAILY_BUYS = 5
STOP_PCT = 5.0
MAX_HOLD = 15


def pf(values):
    wins = sum(v for v in values if v > 0)
    losses = -sum(v for v in values if v <= 0)
    if losses > 0:
        return wins / losses
    return float("inf") if wins > 0 else None


def summarize(rows):
    vals = [r["pnl_pct"] for r in rows]
    if not vals:
        return {"trades": 0, "win_rate": None, "avg": None, "pf": None, "avg_hold": None}
    return {
        "trades": len(vals),
        "win_rate": sum(v > 0 for v in vals) / len(vals) * 100,
        "avg": sum(vals) / len(vals),
        "pf": pf(vals),
        "avg_hold": sum(r["hold_days"] for r in rows) / len(rows),
    }


def simulate_gakutto(g, signal_idx):
    ent = signal_idx + 1
    if ent >= len(g):
        return None
    entry = float(g.O.iloc[ent])
    signal_close = float(g.C.iloc[signal_idx])
    if not entry > 0 or not signal_close > 0:
        return None
    gap = (entry / signal_close - 1) * 100
    if gap > MAX_ENTRY_GAP_PCT:
        return None

    stop = entry * (1 - STOP_PCT / 100)
    pending_gakutto = False
    last = min(ent + MAX_HOLD - 1, len(g) - 1)

    for j in range(ent, last + 1):
        if pending_gakutto:
            px = float(g.O.iloc[j])
            return {"pnl_pct": (px / entry - 1) * 100, "hold_days": j - ent + 1}

        day_open = float(g.O.iloc[j])
        day_low = float(g.L.iloc[j])
        if day_open <= stop:
            return {"pnl_pct": (day_open / entry - 1) * 100, "hold_days": j - ent + 1}
        if day_low <= stop:
            return {"pnl_pct": -STOP_PCT, "hold_days": j - ent + 1}

        if j < last and _strict_gakutto(g, j):
            pending_gakutto = True

    px = float(g.C.iloc[last])
    return {"pnl_pct": (px / entry - 1) * 100, "hold_days": last - ent + 1}


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
            gap = (next_open / signal_close - 1) * 100
            if gap > MAX_ENTRY_GAP_PCT:
                continue

            ma25_today = float(g.MA25.iloc[i])
            ma25_prev = float(g.MA25.iloc[i - 1])
            ma25_5d_ago = float(g.MA25.iloc[i - 5]) if i >= 5 else float("nan")
            slope_1d = (ma25_today / ma25_prev - 1) * 100 if ma25_prev > 0 else None
            slope_5d = (ma25_today / ma25_5d_ago - 1) * 100 if pd.notna(ma25_5d_ago) and ma25_5d_ago > 0 else None

            raw.append({
                "code": code,
                "signal_idx": i,
                "signal_date": g.Date.iloc[i],
                "ma_gap": float(f["ma5_vs_ma25_pct"]),
                "ma25_slope_1d": slope_1d,
                "ma25_slope_5d": slope_5d,
            })
    return raw, groups


def select_daily(candidates):
    selected = []
    by_date = {}
    for x in candidates:
        by_date.setdefault(x["signal_date"], []).append(x)
    for xs in by_date.values():
        xs.sort(key=lambda x: abs(x["ma_gap"]))
        selected.extend(xs[:MAX_DAILY_BUYS])
    return selected


def build_windows(last_date):
    end = pd.Timestamp(last_date).normalize()
    windows = []
    for n in range(YEARS, 0, -1):
        start = end - pd.DateOffset(years=n)
        stop = end - pd.DateOffset(years=n - 1)
        windows.append((start, stop))
    return windows


def fmt(v, digits=3):
    if v is None:
        return "-"
    if v == float("inf"):
        return "inf"
    return f"{v:.{digits}f}"


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
        "ma25_rising_1d": lambda x: x["ma25_slope_1d"] is not None and x["ma25_slope_1d"] > 0,
        "ma25_rising_5d": lambda x: x["ma25_slope_5d"] is not None and x["ma25_slope_5d"] > 0,
        "ma25_flat_or_rising_5d": lambda x: x["ma25_slope_5d"] is not None and x["ma25_slope_5d"] >= -0.5,
    }

    report = {"last_date": str(pd.Timestamp(last_date).date()), "filters": {}}
    lines = [f"MA25 direction filter / 5-year backtest through {report['last_date']}",
             "Exit: stop -5%, strict gakutto -> next open, max 15d; daily top5 by MA25 distance"]

    for label, fn in filters.items():
        filtered = [x for x in candidates if fn(x)]
        selected = select_daily(filtered)
        rows = []
        for s in selected:
            r = simulate_gakutto(groups[s["code"]], s["signal_idx"])
            if r is None:
                continue
            r.update({"code": s["code"], "signal_date": s["signal_date"]})
            rows.append(r)

        overall = summarize(rows)
        yearly = []
        for start, stop in windows:
            subset = [r for r in rows if start <= pd.Timestamp(r["signal_date"]) < stop]
            yearly.append({"start": str(start.date()), "end": str((stop - pd.Timedelta(days=1)).date()), **summarize(subset)})

        report["filters"][label] = {"overall": overall, "yearly": yearly}
        lines.append("")
        lines.append(f"[{label}] n={overall['trades']} win={fmt(overall['win_rate'],2)}% avg={fmt(overall['avg'])}% PF={fmt(overall['pf'])} hold={fmt(overall['avg_hold'],2)}d")
        for y in yearly:
            lines.append(f"{y['start']}..{y['end']} n={y['trades']} win={fmt(y['win_rate'],2)}% avg={fmt(y['avg'])}% PF={fmt(y['pf'])}")

    Path("output/ma25_filter_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    text = "\n".join(lines)
    Path("output/ma25_filter_comparison.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
