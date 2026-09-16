import json
import os
from pathlib import Path

import pandas as pd

import download
from compare_capital_risk import collect_candidates, simulate_trade
from compare_ma25_filter import YEARS, select_daily

INITIAL_CAPITAL = 1_000_000.0
MAX_POSITIONS = 11
LIQUIDITY_MIN = 500_000_000
TRIGGER_COUNT = 5
FRESH_CACHE_FILENAME = "backtest_prices_prime_5y_rotation_fresh.csv"
DIAGNOSTIC_CODES = {"6036", "3097", "3046", "8136", "8830"}


def build_current_trades(candidates, groups):
    eligible = [x for x in candidates if x["avg_turnover_20"] >= LIQUIDITY_MIN]
    selected = select_daily(eligible)
    rows = []
    for s in selected:
        t = simulate_trade(groups[s["code"]], s["signal_idx"])
        if t is None:
            continue
        t.update({"code": s["code"], "ma_gap": s["ma_gap"], "signal_date": s["signal_date"]})
        rows.append(t)
    rows.sort(key=lambda x: (x["entry_date"], abs(x["ma_gap"]), x["code"]))
    return rows


def close_on(groups, code, date, field="C"):
    g = groups[code]
    r = g[g.Date == date]
    if r.empty:
        return None
    return float(r.iloc[0][field])


def baseline(trades):
    slots = [{"capital": INITIAL_CAPITAL / MAX_POSITIONS, "trade": None} for _ in range(MAX_POSITIONS)]
    skipped = 0
    dates = sorted(set(t["entry_date"] for t in trades) | set(t["exit_date"] for t in trades))
    by_entry = {}
    for t in trades:
        by_entry.setdefault(t["entry_date"], []).append(t)
    for d in dates:
        for s in slots:
            tr = s["trade"]
            if tr and tr["exit_date"] == d:
                s["capital"] *= 1 + tr["pnl_pct"] / 100
                s["trade"] = None
        for tr in by_entry.get(d, []):
            free = next((s for s in slots if s["trade"] is None), None)
            if free is None:
                skipped += 1
            else:
                free["trade"] = tr
    for s in slots:
        tr = s["trade"]
        if tr:
            s["capital"] *= 1 + tr["pnl_pct"] / 100
            s["trade"] = None
    return sum(s["capital"] for s in slots), skipped


def rotation(trades, groups):
    slots = [{"capital": INITIAL_CAPITAL / MAX_POSITIONS, "trade": None} for _ in range(MAX_POSITIONS)]
    rotations = []
    skipped = 0
    by_entry = {}
    for t in trades:
        by_entry.setdefault(t["entry_date"], []).append(t)
    dates = sorted(set(t["entry_date"] for t in trades) | set(t["exit_date"] for t in trades))

    for d in dates:
        for s in slots:
            tr = s["trade"]
            if tr and tr["exit_date"] == d:
                s["capital"] *= 1 + tr["pnl_pct"] / 100
                s["trade"] = None

        entrants = sorted(by_entry.get(d, []), key=lambda x: (abs(x["ma_gap"]), x["code"]))
        do_rotate = len(entrants) >= TRIGGER_COUNT
        held_codes = {s["trade"]["code"] for s in slots if s["trade"]}

        for tr in entrants:
            if tr["code"] in held_codes:
                continue
            free = next((s for s in slots if s["trade"] is None), None)
            if free is not None:
                free["trade"] = tr
                held_codes.add(tr["code"])
                continue
            if not do_rotate:
                skipped += 1
                continue

            choices = []
            for s in slots:
                old = s["trade"]
                px = close_on(groups, old["code"], d, "O")
                if px is None:
                    continue
                upct = (px / old["entry_price"] - 1) * 100
                if upct > 0:
                    choices.append((upct, s, px, old))
            if not choices:
                skipped += 1
                continue
            upct, s, px, old = max(choices, key=lambda x: x[0])
            old_normal_remaining = (old["exit_price"] / px - 1) * 100 if px > 0 else None
            s["capital"] *= px / old["entry_price"]
            rotations.append({
                "date": str(d.date()), "sold_code": old["code"], "sold_unrealized_pct": upct,
                "sold_entry_date": str(pd.Timestamp(old["entry_date"]).date()),
                "sold_entry_price": old["entry_price"], "sold_rotation_open": px,
                "sold_normal_exit_date": str(pd.Timestamp(old["exit_date"]).date()),
                "sold_normal_exit_price": old["exit_price"], "sold_normal_pnl_pct": old["pnl_pct"],
                "sold_if_held_from_rotation_pct": old_normal_remaining,
                "replacement_code": tr["code"], "replacement_normal_pct": tr["pnl_pct"],
                "replacement_rank_gap_abs": abs(tr["ma_gap"]),
            })
            held_codes.discard(old["code"])
            s["trade"] = tr
            held_codes.add(tr["code"])

    for s in slots:
        tr = s["trade"]
        if tr:
            s["capital"] *= 1 + tr["pnl_pct"] / 100
            s["trade"] = None
    return sum(s["capital"] for s in slots), skipped, rotations


def load_fresh_price_history():
    cache_path = Path("data") / FRESH_CACHE_FILENAME
    if cache_path.exists():
        cache_path.unlink()
        print(f"Deleted restored rotation cache: {cache_path}")
    print("Rebuilding full 5-year price history from J-Quants (no incremental cache)...")
    return download.get_price_history_range(years=YEARS, cache_filename=FRESH_CACHE_FILENAME)


def diagnostic_lines(rots, groups):
    lines = ["", "PRICE DIAGNOSTICS (raw O/H/L/C around suspicious rotations):"]
    for x in rots:
        code = x["sold_code"]
        if code not in DIAGNOSTIC_CODES:
            continue
        lines.append(
            f"{code}: entry={x['sold_entry_date']} entry_price={x['sold_entry_price']:.4f}; "
            f"rotation={x['date']} open={x['sold_rotation_open']:.4f}; "
            f"normal_exit={x['sold_normal_exit_date']} exit_price={x['sold_normal_exit_price']:.4f} "
            f"normal_pnl={x['sold_normal_pnl_pct']:+.2f}%"
        )
        g = groups[code]
        start = pd.Timestamp(x["sold_entry_date"]) - pd.Timedelta(days=3)
        end = pd.Timestamp(x["sold_normal_exit_date"]) + pd.Timedelta(days=3)
        sample = g[(g.Date >= start) & (g.Date <= end)][["Date", "O", "H", "L", "C"]]
        for _, r in sample.iterrows():
            lines.append(
                f"  {pd.Timestamp(r['Date']).date()} O={float(r['O']):.4f} H={float(r['H']):.4f} "
                f"L={float(r['L']):.4f} C={float(r['C']):.4f}"
            )
    return lines


def main():
    os.makedirs("output", exist_ok=True)
    targets = download.get_target_codes()
    price_df = load_fresh_price_history()
    price_df["Date"] = pd.to_datetime(price_df["Date"])
    candidates, groups = collect_candidates(price_df, targets)
    trades = build_current_trades(candidates, groups)

    b_final, b_skip = baseline(trades)
    r_final, r_skip, rots = rotation(trades, groups)
    report = {
        "through": str(pd.Timestamp(price_df.Date.max()).date()),
        "price_source": "fresh_full_jquants_rebuild",
        "initial_capital": INITIAL_CAPITAL, "max_positions": MAX_POSITIONS,
        "trigger_candidate_count": TRIGGER_COUNT,
        "liquidity_min_yen": LIQUIDITY_MIN,
        "baseline": {"final_equity": b_final, "return_pct": (b_final / INITIAL_CAPITAL - 1) * 100, "skipped": b_skip},
        "rotation": {"final_equity": r_final, "return_pct": (r_final / INITIAL_CAPITAL - 1) * 100, "skipped": r_skip, "rotations": len(rots)},
        "difference_yen": r_final - b_final,
        "difference_pct_points": (r_final - b_final) / INITIAL_CAPITAL * 100,
        "rotation_details": rots,
        "notes": "FRESH full J-Quants rebuild; diagnostics include raw OHLC around suspicious rotations"
    }
    Path("output/profit_rotation_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"Profit rotation comparison / through {report['through']}",
        "Price source=FRESH full J-Quants rebuild (restored/incremental cache ignored)",
        f"Baseline final={b_final:,.0f} return={report['baseline']['return_pct']:+.2f}% skipped={b_skip}",
        f"Rotation final={r_final:,.0f} return={report['rotation']['return_pct']:+.2f}% skipped={r_skip} rotations={len(rots)}",
        f"Rotation - baseline={r_final-b_final:+,.0f} yen ({report['difference_pct_points']:+.2f} pt)",
        "", "Rotation details:"
    ]
    for x in rots:
        lines.append(
            f"{x['date']} sold {x['sold_code']} entry={x['sold_entry_date']}@{x['sold_entry_price']:.2f} "
            f"rotation_open={x['sold_rotation_open']:.2f} unreal={x['sold_unrealized_pct']:+.2f}% "
            f"normal_exit={x['sold_normal_exit_date']}@{x['sold_normal_exit_price']:.2f} "
            f"hold_remaining={x['sold_if_held_from_rotation_pct']:+.2f}% -> "
            f"{x['replacement_code']} replacement={x['replacement_normal_pct']:+.2f}%"
        )
    lines.extend(diagnostic_lines(rots, groups))
    text = "\n".join(lines)
    Path("output/profit_rotation_comparison.txt").write_text(text, encoding="utf-8")
    print(text)

if __name__ == "__main__":
    main()
