"""Unified forward-test journal for production signals.

Kuitto, GC strong breakout, Pinch-to-Chance, and Selling Climax signals are stored in one log.
Existing rows are never re-selected with hindsight; later daily runs only fill in
facts that became available after the signal (next-open entry, future returns,
MFE/MAE, and each strategy's frozen exit result).
"""

import json
from pathlib import Path

import pandas as pd

from logger import get_logger

logger = get_logger(__name__)

FORWARD_TEST_START = pd.Timestamp("2026-09-09")
RULE_EFFECTIVE_FROM = pd.Timestamp("2026-09-16")
RULE_VERSION = "kuitto_v2_2026-09-16_liquidity500m"
LOG_PATH = Path("docs/forward_test_log.json")
RULES_PATH = Path("docs/forward_test_rules.json")
HOLDINGS_PATH = Path("holdings.json")

VOLUME_RATIO_MIN = 1.25
AVG_TURNOVER_20_MIN = 500_000_000
MAX_ENTRY_GAP_PCT = 0.5
MAX_DAILY_BUYS = 5
STOP_LOSS_PCT = 5.0
MAX_HOLD_DAYS = 15

GC_RULE_VERSION = "gc_strong_breakout_v1_2026-09-22"
GC_HOLD_DAYS = 10
PINCH_RULE_VERSION = "pinch_to_chance_v1_2026-09-22_top3_41d"
PINCH_HOLD_DAYS = 41
SELLING_RULE_VERSION = "selling_climax_v1_2026-09-23_top3_41d"
SELLING_HOLD_DAYS = 41

GC_RULE_SNAPSHOT = {
    "version": GC_RULE_VERSION,
    "effective_from": "2026-09-22",
    "strategy": "gc_strong_breakout",
    "entry": {"signal": "production gc_strong_breakout", "entry": "next trading-day open", "gap_cap": None},
    "exit": {"fixed_hold_days": GC_HOLD_DAYS, "exit": "open after 10 trading days from entry", "stop_loss": None},
}

SELLING_RULE_SNAPSHOT = {
    "version": SELLING_RULE_VERSION,
    "effective_from": "2026-09-23",
    "strategy": "selling_climax",
    "entry": {"signal": "selling climax market sensor active + individual anchor", "ranking": "DD20 deepest first", "selected": "top 3", "single_stock_cap_pct": 80, "lot_size": 100, "entry": "next trading-day open", "gap_cap": None},
    "exit": {"fixed_hold_days": SELLING_HOLD_DAYS, "exit": "open after 41 trading days from entry", "stop_loss": None},
}

PINCH_RULE_SNAPSHOT = {
    "version": PINCH_RULE_VERSION,
    "effective_from": "2026-09-22",
    "strategy": "pinch_to_chance",
    "entry": {"signal": "market sensor active + individual anchor", "ranking": "DD20 deepest first", "selected": "top 3", "entry": "next trading-day open", "gap_cap": None},
    "exit": {"fixed_hold_days": PINCH_HOLD_DAYS, "exit": "open after 41 trading days from entry", "stop_loss": None},
}

RULE_SNAPSHOT = {
    "version": RULE_VERSION,
    "effective_from": "2026-09-16",
    "entry": {
        "signal": "kuitto_pullback_auto base shape",
        "volume_ratio_min": VOLUME_RATIO_MIN,
        "avg_turnover_20_min_yen": AVG_TURNOVER_20_MIN,
        "avg_turnover_20_definition": "20-day mean of close x volume",
        "next_open_gap_max_pct": MAX_ENTRY_GAP_PCT,
        "daily_max": MAX_DAILY_BUYS,
        "ranking": "abs(MA5/MA25 gap) ascending when selecting top 5",
        "sector_limit": None,
    },
    "exit": {
        "stop_loss_pct": STOP_LOSS_PCT,
        "stop_fill": "if open <= stop: open, elif low <= stop: stop price",
        "gakutto": "strict gakutto detected at close -> next trading-day open",
        "max_hold_days": MAX_HOLD_DAYS,
        "max_hold_exit": "15th trading-day close alert -> next trading-day open",
    },
}


def _load_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _prepare(group):
    g = group.sort_values("Date").reset_index(drop=True).copy()
    g["Date"] = pd.to_datetime(g["Date"])
    for c in ["O", "H", "L", "C", "Vo"]:
        if c in g.columns:
            g[c] = pd.to_numeric(g[c], errors="coerce")
    if "Vo" not in g.columns:
        g["Vo"] = pd.NA
    g["MA5"] = g["C"].rolling(5).mean()
    g["MA25"] = g["C"].rolling(25).mean()
    g["turnover"] = g["C"] * g["Vo"]
    g["avg_turnover_20"] = g["turnover"].rolling(20).mean()
    return g


def _base_features(g, i):
    if i < 25:
        return None
    r = g.iloc[i]
    if pd.isna(r.MA5) or pd.isna(r.MA25) or not r.O > 0:
        return None
    bull = (r.C / r.O - 1) * 100
    if not (1.5 <= bull <= 3.5):
        return None
    ma_gap = (r.MA5 / r.MA25 - 1) * 100
    if not (-5.0 <= ma_gap <= 0.0):
        return None
    past = i - 1 - 5
    if past < 0 or pd.isna(g.MA5.iloc[past]) or not g.MA5.iloc[past] > 0:
        return None
    decline = (g.MA5.iloc[i - 1] / g.MA5.iloc[past] - 1) * 100
    if not (-5.5 <= decline <= -2.0):
        return None
    if any(g.MA5.iloc[j] > g.MA5.iloc[j - 1] for j in range(i - 2, i)):
        return None
    if not g.MA5.iloc[i] > g.MA5.iloc[i - 1]:
        return None
    close_ma5 = (r.C / r.MA5 - 1) * 100
    if close_ma5 > 4.0:
        return None
    prev_v = g.Vo.iloc[i - 1]
    today_v = r.Vo
    volume_ratio = None
    if pd.notna(prev_v) and pd.notna(today_v) and float(prev_v) > 0:
        volume_ratio = float(today_v) / float(prev_v)
    avg_turnover_20 = g.avg_turnover_20.iloc[i]
    return {
        "bull_candle_pct": round(float(bull), 3),
        "ma5_prior5d_decline_pct": round(float(decline), 3),
        "ma5_vs_ma25_pct": round(float(ma_gap), 3),
        "close_vs_ma5_pct": round(float(close_ma5), 3),
        "volume_ratio": round(float(volume_ratio), 3) if volume_ratio is not None else None,
        "avg_turnover_20": round(float(avg_turnover_20), 3) if pd.notna(avg_turnover_20) else None,
    }


def _strict_gakutto(g, j):
    if j < 3:
        return False
    if not g.MA5.iloc[j] < g.MA5.iloc[j - 1]:
        return False
    if g.MA5.iloc[j - 1] < g.MA5.iloc[j - 2]:
        return False
    if g.MA5.iloc[j - 2] < g.MA5.iloc[j - 3]:
        return False
    return float(g.C.iloc[j]) < float(g.O.iloc[j])


def _pct(px, entry):
    return round((float(px) / float(entry) - 1) * 100, 3)


def _update_future(row, g, signal_idx):
    ent = signal_idx + 1
    if ent >= len(g):
        return

    entry = float(g.O.iloc[ent])
    signal_close = float(g.C.iloc[signal_idx])
    if not entry > 0 or not signal_close > 0:
        return

    row["entry_date"] = g.Date.iloc[ent].strftime("%Y-%m-%d")
    row["entry_open"] = round(entry, 3)
    row["next_open_gap_pct"] = round((entry / signal_close - 1) * 100, 3)
    row["gap_pass"] = row["next_open_gap_pct"] <= MAX_ENTRY_GAP_PCT

    available_end = len(g) - 1
    for days in (5, 10, 15):
        idx = ent + days - 1
        if idx <= available_end:
            row[f"return_{days}d_pct"] = _pct(g.C.iloc[idx], entry)

    observed = g.iloc[ent:available_end + 1]
    if not observed.empty:
        row["mfe_pct"] = round((float(observed.H.max()) / entry - 1) * 100, 3)
        row["mae_pct"] = round((float(observed.L.min()) / entry - 1) * 100, 3)

    if row.get("exit_date"):
        return

    stop = entry * (1 - STOP_LOSS_PCT / 100)
    pending_gakutto = False
    max_j = min(ent + MAX_HOLD_DAYS - 1, available_end)

    for j in range(ent, max_j + 1):
        if pending_gakutto:
            px = float(g.O.iloc[j])
            row.update({
                "exit_date": g.Date.iloc[j].strftime("%Y-%m-%d"),
                "exit_price": round(px, 3),
                "exit_reason": "strict_gakutto_next_open",
                "exit_pnl_pct": _pct(px, entry),
                "hold_days": j - ent + 1,
            })
            return

        day_open = float(g.O.iloc[j])
        day_low = float(g.L.iloc[j])
        if day_open <= stop:
            px = day_open
            row.update({
                "exit_date": g.Date.iloc[j].strftime("%Y-%m-%d"),
                "exit_price": round(px, 3),
                "exit_reason": "stop_gap_open",
                "exit_pnl_pct": _pct(px, entry),
                "hold_days": j - ent + 1,
            })
            return
        if day_low <= stop:
            row.update({
                "exit_date": g.Date.iloc[j].strftime("%Y-%m-%d"),
                "exit_price": round(stop, 3),
                "exit_reason": "stop_loss",
                "exit_pnl_pct": round(-STOP_LOSS_PCT, 3),
                "hold_days": j - ent + 1,
            })
            return

        if j < ent + MAX_HOLD_DAYS - 1 and _strict_gakutto(g, j):
            pending_gakutto = True

    timeout_day = ent + MAX_HOLD_DAYS
    if timeout_day <= available_end:
        px = float(g.O.iloc[timeout_day])
        row.update({
            "exit_date": g.Date.iloc[timeout_day].strftime("%Y-%m-%d"),
            "exit_price": round(px, 3),
            "exit_reason": "max15_next_open",
            "exit_pnl_pct": _pct(px, entry),
            "hold_days": MAX_HOLD_DAYS + 1,
        })



def _update_fixed_exit_future(row, g, signal_idx, hold_days, exit_reason):
    """Fill common OOS fields for fixed-time GC / Pinch rules."""
    ent = signal_idx + 1
    if ent >= len(g):
        return

    entry = float(g.O.iloc[ent])
    signal_close = float(g.C.iloc[signal_idx])
    if not entry > 0 or not signal_close > 0:
        return

    row["entry_date"] = g.Date.iloc[ent].strftime("%Y-%m-%d")
    row["entry_open"] = round(entry, 3)
    row["next_open_gap_pct"] = round((entry / signal_close - 1) * 100, 3)
    row["gap_pass"] = True

    available_end = len(g) - 1
    for days in (5, 10, 15, 20, 41):
        idx = ent + days
        if idx <= available_end:
            row[f"return_{days}d_pct"] = _pct(g.O.iloc[idx], entry)

    observed_end = min(ent + hold_days, available_end)
    observed = g.iloc[ent:observed_end + 1]
    if not observed.empty:
        row["mfe_pct"] = round((float(observed.H.max()) / entry - 1) * 100, 3)
        row["mae_pct"] = round((float(observed.L.min()) / entry - 1) * 100, 3)

    exit_idx = ent + hold_days
    if row.get("exit_date") or exit_idx > available_end:
        return

    px = float(g.O.iloc[exit_idx])
    row.update({
        "exit_date": g.Date.iloc[exit_idx].strftime("%Y-%m-%d"),
        "exit_price": round(px, 3),
        "exit_reason": exit_reason,
        "exit_pnl_pct": _pct(px, entry),
        "hold_days": hold_days,
    })


def _append_gc_rows(rows, keyed, gc_results, signal_date, code_to_name):
    for r in gc_results or []:
        code = str(r.get("code"))
        day = pd.Timestamp(signal_date).strftime("%Y-%m-%d")
        key = (code, day, GC_RULE_VERSION)
        if key in keyed:
            continue
        rows.append({
            "strategy": "gc_strong_breakout",
            "rule_version": GC_RULE_VERSION,
            "code": code,
            "name": r.get("name") or code_to_name.get(code, ""),
            "signal_date": day,
            "signal_close": r.get("close"),
            "gc_sequence": r.get("gc_sequence"),
            "dd60_pct": r.get("dd60_pct"),
            "ma25_slope5_pct": r.get("ma25_slope5_pct"),
            "gc_gap_pct": r.get("gc_gap_pct"),
            "volume_ratio20": r.get("volume_ratio20"),
            "bull_candle_pct": r.get("bull_candle_pct"),
            "rule_selected": True,
            "actual_bought": None,
            "entry_date": None, "entry_open": None, "next_open_gap_pct": None, "gap_pass": True,
            "return_5d_pct": None, "return_10d_pct": None, "return_15d_pct": None,
            "return_20d_pct": None, "return_41d_pct": None,
            "mfe_pct": None, "mae_pct": None,
            "exit_date": None, "exit_price": None, "exit_reason": None,
            "exit_pnl_pct": None, "hold_days": None,
        })
        keyed.add(key)


def _append_pinch_rows(rows, keyed, pinch_sensor, code_to_name):
    if not pinch_sensor or not pinch_sensor.get("active"):
        return
    day = pd.Timestamp(pinch_sensor.get("signal_date")).strftime("%Y-%m-%d")
    for r in pinch_sensor.get("main_candidates", []) or []:
        code = str(r.get("code"))
        key = (code, day, PINCH_RULE_VERSION)
        if key in keyed:
            continue
        rows.append({
            "strategy": "pinch_to_chance",
            "rule_version": PINCH_RULE_VERSION,
            "code": code,
            "name": r.get("name") or code_to_name.get(code, ""),
            "signal_date": day,
            "signal_close": r.get("close"),
            "dd20_rank": r.get("dd20_rank"),
            "ret5_pct": r.get("ret5_pct"),
            "dd20_pct": r.get("dd20_pct"),
            "bull_candle_pct": r.get("bull_candle_pct"),
            "volume_ratio20": r.get("volume_ratio20"),
            "market_reversal_rate_pct": pinch_sensor.get("reversal_rate_pct"),
            "topix_return_pct": pinch_sensor.get("topix_return_pct"),
            "topix_low_up": pinch_sensor.get("topix_low_up"),
            "planned_entry_date": pinch_sensor.get("planned_entry_date"),
            "planned_exit_date": pinch_sensor.get("planned_exit_date"),
            "rule_selected": True,
            "actual_bought": None,
            "entry_date": None, "entry_open": None, "next_open_gap_pct": None, "gap_pass": True,
            "return_5d_pct": None, "return_10d_pct": None, "return_15d_pct": None,
            "return_20d_pct": None, "return_41d_pct": None,
            "mfe_pct": None, "mae_pct": None,
            "exit_date": None, "exit_price": None, "exit_reason": None,
            "exit_pnl_pct": None, "hold_days": None,
        })
        keyed.add(key)


def _append_selling_rows(rows, keyed, selling_sensor, code_to_name):
    if not selling_sensor or not selling_sensor.get("active"):
        return
    day = pd.Timestamp(selling_sensor.get("signal_date")).strftime("%Y-%m-%d")
    for r in selling_sensor.get("main_candidates", []) or []:
        code = str(r.get("code"))
        key = (code, day, SELLING_RULE_VERSION)
        if key in keyed:
            continue
        rows.append({
            "strategy": "selling_climax",
            "rule_version": SELLING_RULE_VERSION,
            "code": code,
            "name": r.get("name") or code_to_name.get(code, ""),
            "signal_date": day,
            "signal_close": r.get("close"),
            "dd20_rank": r.get("dd20_rank"),
            "ret5_pct": r.get("ret5_pct"),
            "dd20_pct": r.get("dd20_pct"),
            "bull_candle_pct": r.get("bull_candle_pct"),
            "volume_ratio20": r.get("volume_ratio20"),
            "topix_return_pct": selling_sensor.get("topix_return_pct"),
            "topix_dd20_pct": selling_sensor.get("topix_dd20_pct"),
            "anchor_share_pct": selling_sensor.get("anchor_share_pct"),
            "anchor_count": selling_sensor.get("anchor_count"),
            "coarse_reversal_count": selling_sensor.get("coarse_reversal_count"),
            "single_stock_cap_pct": selling_sensor.get("single_stock_cap_pct", 80),
            "lot_size": selling_sensor.get("lot_size", 100),
            "planned_entry_date": selling_sensor.get("planned_entry_date"),
            "planned_exit_date": selling_sensor.get("planned_exit_date"),
            "rule_selected": True,
            "actual_bought": None,
            "entry_date": None, "entry_open": None, "next_open_gap_pct": None, "gap_pass": True,
            "return_5d_pct": None, "return_10d_pct": None, "return_15d_pct": None,
            "return_20d_pct": None, "return_41d_pct": None,
            "mfe_pct": None, "mae_pct": None,
            "exit_date": None, "exit_price": None, "exit_reason": None,
            "exit_pnl_pct": None, "hold_days": None,
        })
        keyed.add(key)


def _row_passes_entry_filters(row):
    if not row.get("volume_pass") or not row.get("gap_pass"):
        return False
    if row.get("rule_version") == RULE_VERSION:
        return bool(row.get("liquidity_pass"))
    return True


def _mark_rule_selection(rows):
    by_date = {}
    for row in rows:
        if row.get("strategy") not in (None, "", "kuitto_pullback_auto"):
            continue
        if _row_passes_entry_filters(row):
            by_date.setdefault(row["signal_date"], []).append(row)

    for candidates in by_date.values():
        candidates.sort(key=lambda x: abs(x.get("ma5_vs_ma25_pct", 999)))
        for rank, row in enumerate(candidates, start=1):
            row["ma25_rank"] = rank
            row["rule_selected"] = rank <= MAX_DAILY_BUYS


def _mark_actual_buys(rows):
    holdings = _load_json(HOLDINGS_PATH, [])
    known = set()
    for h in holdings if isinstance(holdings, list) else []:
        code = str(h.get("code", ""))
        buy_date = str(h.get("buy_date", ""))
        if code and buy_date:
            known.add((code, buy_date))
    for row in rows:
        if row.get("actual_bought") is True:
            continue
        ent_date = row.get("entry_date")
        if ent_date and (str(row.get("code")), ent_date) in known:
            row["actual_bought"] = True


def update_forward_test(price_df, code_to_name=None, gc_results=None, pinch_sensor=None, selling_sensor=None):
    code_to_name = code_to_name or {}
    rules = _load_json(RULES_PATH, {"versions": []})
    changed_rules = False
    if not any(v.get("version") == RULE_VERSION for v in rules.get("versions", [])):
        rules.setdefault("versions", []).append(RULE_SNAPSHOT)
        changed_rules = True
    if not any(v.get("version") == GC_RULE_VERSION for v in rules.get("versions", [])):
        rules.setdefault("versions", []).append(GC_RULE_SNAPSHOT)
        changed_rules = True
    if not any(v.get("version") == PINCH_RULE_VERSION for v in rules.get("versions", [])):
        rules.setdefault("versions", []).append(PINCH_RULE_SNAPSHOT)
        changed_rules = True
    if not any(v.get("version") == SELLING_RULE_VERSION for v in rules.get("versions", [])):
        rules.setdefault("versions", []).append(SELLING_RULE_SNAPSHOT)
        changed_rules = True
    if changed_rules:
        _save_json(RULES_PATH, rules)

    payload = _load_json(LOG_PATH, {"started_at": "2026-09-09", "items": []})
    rows = payload.setdefault("items", [])
    keyed = {(str(r.get("code")), str(r.get("signal_date")), str(r.get("rule_version"))) for r in rows}

    groups = {}
    for code, group in price_df.groupby("Code"):
        groups[str(code)] = _prepare(group)

    # New rule version begins 2026-09-16; older rows are preserved unchanged.
    for code, g in groups.items():
        for i in range(25, len(g)):
            signal_date = g.Date.iloc[i]
            if signal_date < RULE_EFFECTIVE_FROM:
                continue
            f = _base_features(g, i)
            if not f:
                continue
            key = (code, signal_date.strftime("%Y-%m-%d"), RULE_VERSION)
            if key in keyed:
                continue
            volume_pass = f["volume_ratio"] is not None and f["volume_ratio"] >= VOLUME_RATIO_MIN
            liquidity_pass = f["avg_turnover_20"] is not None and f["avg_turnover_20"] >= AVG_TURNOVER_20_MIN
            rows.append({
                "strategy": "kuitto_pullback_auto",
                "rule_version": RULE_VERSION,
                "code": code,
                "name": code_to_name.get(code, ""),
                "signal_date": signal_date.strftime("%Y-%m-%d"),
                "signal_close": round(float(g.C.iloc[i]), 3),
                **f,
                "volume_pass": bool(volume_pass),
                "liquidity_pass": bool(liquidity_pass),
                "gap_pass": None,
                "ma25_rank": None,
                "rule_selected": None,
                "actual_bought": None,
                "entry_date": None,
                "entry_open": None,
                "next_open_gap_pct": None,
                "return_5d_pct": None,
                "return_10d_pct": None,
                "return_15d_pct": None,
                "mfe_pct": None,
                "mae_pct": None,
                "exit_date": None,
                "exit_price": None,
                "exit_reason": None,
                "exit_pnl_pct": None,
                "hold_days": None,
            })
            keyed.add(key)

    # Append today's sparse production signals.
    latest_signal_date = pd.to_datetime(price_df["Date"]).max() if not price_df.empty else None
    if latest_signal_date is not None:
        _append_gc_rows(rows, keyed, gc_results, latest_signal_date, code_to_name)
    _append_pinch_rows(rows, keyed, pinch_sensor, code_to_name)
    _append_selling_rows(rows, keyed, selling_sensor, code_to_name)

    # Fill future facts for all historical rule versions so old forward-test rows keep progressing.
    for row in rows:
        g = groups.get(str(row.get("code")))
        if g is None or g.empty:
            continue
        matches = g.index[g.Date.dt.strftime("%Y-%m-%d") == row.get("signal_date")].tolist()
        if not matches:
            continue
        strategy = row.get("strategy") or "kuitto_pullback_auto"
        if strategy == "gc_strong_breakout":
            _update_fixed_exit_future(row, g, matches[0], GC_HOLD_DAYS, "fixed10_next_open")
        elif strategy == "pinch_to_chance":
            _update_fixed_exit_future(row, g, matches[0], PINCH_HOLD_DAYS, "fixed41_next_open")
        elif strategy == "selling_climax":
            _update_fixed_exit_future(row, g, matches[0], SELLING_HOLD_DAYS, "fixed41_next_open")
        else:
            _update_future(row, g, matches[0])

    _mark_rule_selection(rows)
    _mark_actual_buys(rows)
    rows.sort(key=lambda r: (r.get("signal_date", ""), r.get("code", "")))
    payload["updated_through"] = (
        pd.to_datetime(price_df["Date"]).max().strftime("%Y-%m-%d") if not price_df.empty else None
    )
    _save_json(LOG_PATH, payload)
    logger.info(f"Forward test log updated: {len(rows)} rows")
