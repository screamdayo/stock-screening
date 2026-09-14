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


def simulate(g, signal_idx, *, stop_pct, max_hold, exit_mode, take_profit_pct=5.0):
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

    stop = entry * (1 - stop_pct / 100)
    tp = entry * (1 + take_profit_pct / 100)
    pending_gakutto = False
    last = min(ent + max_hold - 1, len(g) - 1)

    for j in range(ent, last + 1):
        if pending_gakutto:
            px = float(g.O.iloc[j])
            return {
                "entry_date": g.Date.iloc[ent], "exit_date": g.Date.iloc[j],
                "pnl_pct": (px / entry - 1) * 100, "hold_days": j - ent + 1,
                "reason": "gakutto_next_open",
            }

        day_open = float(g.O.iloc[j])
        day_low = float(g.L.iloc[j])
        day_high = float(g.H.iloc[j])

        if day_open <= stop:
            return {"entry_date": g.Date.iloc[ent], "exit_date": g.Date.iloc[j],
                    "pnl_pct": (day_open / entry - 1) * 100, "hold_days": j - ent + 1,
                    "reason": "stop_gap"}
        if day_low <= stop:
            return {"entry_date": g.Date.iloc[ent], "exit_date": g.Date.iloc[j],
                    "pnl_pct": -stop_pct, "hold_days": j - ent + 1, "reason": "stop"}

        if exit_mode == "tp5" and day_high >= tp:
            return {"entry_date": g.Date.iloc[ent], "exit_date": g.Date.iloc[j],
                    "pnl_pct": take_profit_pct, "hold_days": j - ent + 1, "reason": "tp5"}

        if exit_mode == "gakutto" and j < last and _strict_gakutto(g, j):
            pending_gakutto = True

    px = float(g.C.iloc[last])
    return {"entry_date": g.Date.iloc[ent], "exit_date": g.Date.iloc[last],
            "pnl_pct": (px / entry - 1) * 100, "hold_days": last - ent + 1,
            "reason": "time_exit"}


def collect_signals(price_df, target_codes):
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
            raw.append({
                "code": code,
                "signal_idx": i,
                "signal_date": g.Date.iloc[i],
                "ma_gap": float(f["ma5_vs_ma25_pct"]),
            })

    selected = []
    by_date = {}
    for x in raw:
        by_date.setdefault(x["signal_date"], []).append(x)
    for xs in by_date.values():
        xs.sort(key=lambda x: abs(x["ma_gap"]))
        selected.extend(xs[:MAX_DAILY_BUYS])
    return selected, groups


def build_windows(last_date):
    end = pd.Timestamp(last_date).normalize()
    windows = []
    for n in range(YEARS, 0, -1):
        start = end - pd.DateOffset(years=n)
        stop = end - pd.DateOffset(years=n - 1)
        windows.append((start, stop))
    return windows


def fmt(v, digits=2):
    if v is None:
        return "-"
    if v == float("inf"):
        return "inf"
    return f"{v:.{digits}f}"


def main():
    os.makedirs("output", exist_ok=True)
    target_codes = download.get_target_codes()
    price_df = download.get_price_history_incremental(
        cache_filename="backtest_prices_prime_5y.csv",
        years=YEARS,
    )
    price_df["Date"] = pd.to_datetime(price_df["Date"])
    signals, groups = collect_signals(price_df, target_codes)
    last_date = price_df["Date"].max()
    windows = build_windows(last_date)

    scenarios = [
        ("legacy_3pct_10d", 3.0, 10),
        ("current_5pct_15d", 5.0, 15),
    ]
    report = {"last_date": str(pd.Timestamp(last_date).date()), "signal_count": len(signals), "scenarios": {}}

    for label, stop_pct, max_hold in scenarios:
        rows_by_mode = {"tp5": [], "gakutto": []}
        for s in signals:
            g = groups[s["code"]]
            for mode in rows_by_mode:
                r = simulate(g, s["signal_idx"], stop_pct=stop_pct, max_hold=max_hold, exit_mode=mode)
                if r is not None:
                    r.update({"code": s["code"], "signal_date": s["signal_date"]})
                    rows_by_mode[mode].append(r)

        scenario = {"overall": {}, "yearly": []}
        for mode, rows in rows_by_mode.items():
            scenario["overall"][mode] = summarize(rows)

        for start, stop in windows:
            item = {"start": str(start.date()), "end": str((stop - pd.Timedelta(days=1)).date())}
            for mode, rows in rows_by_mode.items():
                subset = [r for r in rows if start <= pd.Timestamp(r["signal_date"]) < stop]
                item[mode] = summarize(subset)
            scenario["yearly"].append(item)
        report["scenarios"][label] = scenario

    Path("output/yearly_exit_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    lines = []
    lines.append(f"5-year exit comparison through {report['last_date']} / selected signals={report['signal_count']}")
    for label, scenario in report["scenarios"].items():
        lines.append("")
        lines.append(f"[{label}]")
        for mode in ("tp5", "gakutto"):
            s = scenario["overall"][mode]
            lines.append(f"OVERALL {mode}: n={s['trades']} win={fmt(s['win_rate'])}% avg={fmt(s['avg'],3)}% PF={fmt(s['pf'],3)} hold={fmt(s['avg_hold'])}d")
        for y in scenario["yearly"]:
            a, b = y["tp5"], y["gakutto"]
            lines.append(
                f"{y['start']}..{y['end']} | TP5 n={a['trades']} PF={fmt(a['pf'],3)} avg={fmt(a['avg'],3)}% | "
                f"GAK n={b['trades']} PF={fmt(b['pf'],3)} avg={fmt(b['avg'],3)}%"
            )
    text = "\n".join(lines)
    print(text)
    Path("output/yearly_exit_comparison.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
