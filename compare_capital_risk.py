import json
import os
from pathlib import Path

import pandas as pd

import download
from forward_test import _prepare, _base_features, _strict_gakutto
from compare_ma25_filter import YEARS, select_daily

INITIAL_CAPITAL = 1_000_000.0
MAX_POSITIONS = 8
VOLUME_RATIO_MIN = 1.25
MAX_ENTRY_GAP_PCT = 0.5
STOP_PCT = 5.0
MAX_HOLD = 15


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
            if not f or f.get("volume_ratio") is None or f["volume_ratio"] < VOLUME_RATIO_MIN:
                continue
            signal_close = float(g.C.iloc[i])
            next_open = float(g.O.iloc[i + 1])
            if not signal_close > 0 or not next_open > 0:
                continue
            gap = (next_open / signal_close - 1) * 100
            if gap > MAX_ENTRY_GAP_PCT:
                continue
            avg_turnover = g.avg_turnover_20.iloc[i]
            if pd.isna(avg_turnover):
                continue
            ma25_today = float(g.MA25.iloc[i])
            ma25_prev = float(g.MA25.iloc[i - 1])
            raw.append({
                "code": code,
                "signal_idx": i,
                "signal_date": pd.Timestamp(g.Date.iloc[i]),
                "ma_gap": float(f["ma5_vs_ma25_pct"]),
                "ma25_slope_1d": (ma25_today / ma25_prev - 1) * 100 if ma25_prev > 0 else None,
                "avg_turnover_20": float(avg_turnover),
            })
    return raw, groups


def simulate_trade(g, signal_idx):
    ent = signal_idx + 1
    if ent >= len(g):
        return None
    entry = float(g.O.iloc[ent])
    signal_close = float(g.C.iloc[signal_idx])
    if not entry > 0 or not signal_close > 0:
        return None
    if (entry / signal_close - 1) * 100 > MAX_ENTRY_GAP_PCT:
        return None

    stop = entry * (1 - STOP_PCT / 100)
    pending = False
    last = min(ent + MAX_HOLD - 1, len(g) - 1)
    for j in range(ent, last + 1):
        if pending:
            px = float(g.O.iloc[j])
            return {
                "entry_date": pd.Timestamp(g.Date.iloc[ent]), "exit_date": pd.Timestamp(g.Date.iloc[j]),
                "entry_price": entry, "exit_price": px, "pnl_pct": (px / entry - 1) * 100,
                "exit_reason": "strict_gakutto_next_open",
            }
        day_open = float(g.O.iloc[j])
        day_low = float(g.L.iloc[j])
        if day_open <= stop:
            return {
                "entry_date": pd.Timestamp(g.Date.iloc[ent]), "exit_date": pd.Timestamp(g.Date.iloc[j]),
                "entry_price": entry, "exit_price": day_open, "pnl_pct": (day_open / entry - 1) * 100,
                "exit_reason": "stop_gap_open",
            }
        if day_low <= stop:
            return {
                "entry_date": pd.Timestamp(g.Date.iloc[ent]), "exit_date": pd.Timestamp(g.Date.iloc[j]),
                "entry_price": entry, "exit_price": stop, "pnl_pct": -STOP_PCT,
                "exit_reason": "stop_loss",
            }
        if j < last and _strict_gakutto(g, j):
            pending = True

    px = float(g.C.iloc[last])
    return {
        "entry_date": pd.Timestamp(g.Date.iloc[ent]), "exit_date": pd.Timestamp(g.Date.iloc[last]),
        "entry_price": entry, "exit_price": px, "pnl_pct": (px / entry - 1) * 100,
        "exit_reason": "max15_close",
    }


def build_trades(candidates, groups, fn):
    selected = select_daily([x for x in candidates if fn(x)])
    rows = []
    for s in selected:
        t = simulate_trade(groups[s["code"]], s["signal_idx"])
        if t is None:
            continue
        t.update({"code": s["code"], "ma_gap": s["ma_gap"]})
        rows.append(t)
    rows.sort(key=lambda x: (x["entry_date"], abs(x["ma_gap"]), x["code"]))
    return rows


def assign_slots(trades):
    slots = [{"capital": INITIAL_CAPITAL / MAX_POSITIONS, "available_after": pd.Timestamp.min, "trades": []}
             for _ in range(MAX_POSITIONS)]
    skipped = 0
    for tr in trades:
        free = [i for i, s in enumerate(slots) if s["available_after"] < tr["entry_date"]]
        if not free:
            skipped += 1
            continue
        i = free[0]
        s = slots[i]
        entry_capital = s["capital"]
        exit_capital = entry_capital * (1 + tr["pnl_pct"] / 100)
        rec = dict(tr)
        rec.update({"slot": i + 1, "entry_capital": entry_capital, "exit_capital": exit_capital})
        s["trades"].append(rec)
        s["capital"] = exit_capital
        s["available_after"] = tr["exit_date"]
    return slots, skipped


def slot_value_on_date(slot, date, close_map):
    capital = INITIAL_CAPITAL / MAX_POSITIONS
    for tr in slot["trades"]:
        if date < tr["entry_date"]:
            return capital
        if tr["entry_date"] <= date < tr["exit_date"]:
            close = close_map.get((tr["code"], date))
            if close is None:
                return tr["entry_capital"]
            return tr["entry_capital"] * float(close) / tr["entry_price"]
        if date >= tr["exit_date"]:
            capital = tr["exit_capital"]
    return capital


def summarize_portfolio(slots, skipped, price_df):
    executed = [t for s in slots for t in s["trades"]]
    final_equity = sum(s["capital"] for s in slots)
    gross_gains = sum(t["exit_capital"] - t["entry_capital"] for t in executed if t["pnl_pct"] > 0)
    gross_losses = sum(t["exit_capital"] - t["entry_capital"] for t in executed if t["pnl_pct"] <= 0)

    if not executed:
        return {
            "executed_trades": 0, "skipped_capacity": skipped, "final_equity": INITIAL_CAPITAL,
            "net_profit": 0.0, "return_pct": 0.0, "max_dd_pct": 0.0,
            "gross_gains": 0.0, "gross_losses": 0.0,
        }

    start = min(t["entry_date"] for t in executed)
    end = max(t["exit_date"] for t in executed)
    dates = sorted(pd.to_datetime(price_df.loc[(price_df["Date"] >= start) & (price_df["Date"] <= end), "Date"].unique()))
    close_map = {(str(r.Code), pd.Timestamp(r.Date)): float(r.C) for r in price_df.itertuples() if pd.notna(r.C)}
    curve = []
    for d in dates:
        d = pd.Timestamp(d)
        eq = sum(slot_value_on_date(s, d, close_map) for s in slots)
        curve.append((d, eq))
    peak = 0.0
    max_dd = 0.0
    for _, eq in curve:
        peak = max(peak, eq)
        if peak > 0:
            max_dd = min(max_dd, (eq / peak - 1) * 100)

    return {
        "executed_trades": len(executed),
        "skipped_capacity": skipped,
        "final_equity": final_equity,
        "net_profit": final_equity - INITIAL_CAPITAL,
        "return_pct": (final_equity / INITIAL_CAPITAL - 1) * 100,
        "max_dd_pct": max_dd,
        "gross_gains": gross_gains,
        "gross_losses": gross_losses,
    }


def main():
    os.makedirs("output", exist_ok=True)
    target_codes = download.get_target_codes()
    price_df = download.get_price_history_incremental(cache_filename="backtest_prices_prime_5y.csv", years=YEARS)
    price_df["Date"] = pd.to_datetime(price_df["Date"])
    candidates, groups = collect_candidates(price_df, target_codes)

    gap35 = lambda x: 3.0 < abs(x["ma_gap"]) <= 5.0
    rising1 = lambda x: x["ma25_slope_1d"] is not None and x["ma25_slope_1d"] > 0
    liquid500 = lambda x: x["avg_turnover_20"] >= 500_000_000

    cases = {
        "baseline_current": lambda x: True,
        "distance_3_to_5pct": gap35,
        "ma25_rising1d_and_3to5": lambda x: rising1(x) and gap35(x),
        "turnover_ge_500m": liquid500,
        "turnover500m_and_3to5": lambda x: liquid500(x) and gap35(x),
        "turnover500m_rising1d_3to5": lambda x: liquid500(x) and rising1(x) and gap35(x),
    }

    report = {
        "through": str(pd.Timestamp(price_df["Date"].max()).date()),
        "initial_capital": INITIAL_CAPITAL,
        "max_positions": MAX_POSITIONS,
        "notes": "8 equal capital slots, per-slot compounding, daily close mark-to-market DD; fees/tax/slippage excluded",
        "cases": {},
    }
    lines = [
        f"Capital/risk comparison / 5-year backtest through {report['through']}",
        f"Initial capital: {INITIAL_CAPITAL:,.0f} yen / max concurrent positions: {MAX_POSITIONS}",
        "Position sizing: 8 equal slots, each slot reinvests its own realized capital",
        "Max DD: daily close mark-to-market; fees/tax/slippage excluded",
    ]

    for label, fn in cases.items():
        trades = build_trades(candidates, groups, fn)
        slots, skipped = assign_slots(trades)
        s = summarize_portfolio(slots, skipped, price_df)
        report["cases"][label] = s
        lines.append("")
        lines.append(
            f"[{label}] trades={s['executed_trades']} skipped_capacity={s['skipped_capacity']} "
            f"final={s['final_equity']:,.0f} yen net={s['net_profit']:+,.0f} yen "
            f"return={s['return_pct']:+.2f}% maxDD={s['max_dd_pct']:.2f}% "
            f"gross_gain={s['gross_gains']:,.0f} gross_loss={s['gross_losses']:,.0f}"
        )

    Path("output/capital_risk_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    text = "\n".join(lines)
    Path("output/capital_risk_comparison.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
