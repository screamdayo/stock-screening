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
    realized = 0.0
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
                realized += 1
                s["trade"] = None
        for tr in by_entry.get(d, []):
            free = next((s for s in slots if s["trade"] is None), None)
            if free is None:
                skipped += 1
            else:
                free["trade"] = tr
    # realize any still-open simulated trades at their predefined normal exits
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
        # normal exits happen first at the date's normal modeled exit price
        for s in slots:
            tr = s["trade"]
            if tr and tr["exit_date"] == d:
                s["capital"] *= 1 + tr["pnl_pct"] / 100
                s["trade"] = None

        entrants = sorted(by_entry.get(d, []), key=lambda x: (abs(x["ma_gap"]), x["code"]))
        # only rotate on days whose eligible candidate cohort has >=5 names
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

            # sell the currently profitable holding with the largest unrealized percentage gain at this open
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


def main():
    os.makedirs("output", exist_ok=True)
    targets = download.get_target_codes()
    price_df = download.get_price_history_incremental(cache_filename="backtest_prices_prime_5y.csv", years=YEARS)
    price_df["Date"] = pd.to_datetime(price_df["Date"])
    candidates, groups = collect_candidates(price_df, targets)
    trades = build_current_trades(candidates, groups)

    b_final, b_skip = baseline(trades)
    r_final, r_skip, rots = rotation(trades, groups)
    report = {
        "through": str(pd.Timestamp(price_df.Date.max()).date()),
        "initial_capital": INITIAL_CAPITAL, "max_positions": MAX_POSITIONS,
        "trigger_candidate_count": TRIGGER_COUNT,
        "liquidity_min_yen": LIQUIDITY_MIN,
        "baseline": {"final_equity": b_final, "return_pct": (b_final / INITIAL_CAPITAL - 1) * 100, "skipped": b_skip},
        "rotation": {"final_equity": r_final, "return_pct": (r_final / INITIAL_CAPITAL - 1) * 100, "skipped": r_skip, "rotations": len(rots)},
        "difference_yen": r_final - b_final,
        "difference_pct_points": (r_final - b_final) / INITIAL_CAPITAL * 100,
        "rotation_details": rots,
        "notes": "5-year current-entry approximation; 11 equal slots; on >=5-entry days, when full, sell largest positive unrealized pct at next open and buy highest-ranked candidate; normal exits -5% stop/strict gakutto/max15; fees tax slippage excluded"
    }
    Path("output/profit_rotation_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"Profit rotation comparison / through {report['through']}",
        f"Baseline final={b_final:,.0f} return={report['baseline']['return_pct']:+.2f}% skipped={b_skip}",
        f"Rotation final={r_final:,.0f} return={report['rotation']['return_pct']:+.2f}% skipped={r_skip} rotations={len(rots)}",
        f"Rotation - baseline={r_final-b_final:+,.0f} yen ({report['difference_pct_points']:+.2f} pt)",
        "",
        "Rotation details:"
    ]
    for x in rots:
        lines.append(f"{x['date']} sold {x['sold_code']} unreal={x['sold_unrealized_pct']:+.2f}% hold_remaining={x['sold_if_held_from_rotation_pct']:+.2f}% -> {x['replacement_code']} replacement={x['replacement_normal_pct']:+.2f}%")
    text = "\n".join(lines)
    Path("output/profit_rotation_comparison.txt").write_text(text, encoding="utf-8")
    print(text)

if __name__ == "__main__":
    main()
