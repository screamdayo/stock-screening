"""Forward-test journal for the production Kuitto strategy.

Only signals on/after FORWARD_TEST_START are recorded. Existing rows are never
re-selected with hindsight; later daily runs only fill in information that became
available after the signal (next-open gap, returns, MFE/MAE, exit result, etc.).
"""

import json
from pathlib import Path

import pandas as pd

from logger import get_logger

logger = get_logger(__name__)

FORWARD_TEST_START = pd.Timestamp("2026-09-09")
RULE_VERSION = "kuitto_v1_2026-09-09"
LOG_PATH = Path("docs/forward_test_log.json")
RULES_PATH = Path("docs/forward_test_rules.json")
HOLDINGS_PATH = Path("holdings.json")

VOLUME_RATIO_MIN = 1.25
MAX_ENTRY_GAP_PCT = 0.5
MAX_DAILY_BUYS = 5
STOP_LOSS_PCT = 5.0
MAX_HOLD_DAYS = 15

RULE_SNAPSHOT = {
    "version": RULE_VERSION,
    "effective_from": "2026-09-09",
    "entry": {
        "signal": "kuitto_pullback_auto base shape",
        "volume_ratio_min": VOLUME_RATIO_MIN,
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
    return {
        "bull_candle_pct": round(float(bull), 3),
        "ma5_prior5d_decline_pct": round(float(decline), 3),
        "ma5_vs_ma25_pct": round(float(ma_gap), 3),
        "close_vs_ma5_pct": round(float(close_ma5), 3),
        "volume_ratio": round(float(volume_ratio), 3) if volume_ratio is not None else None,
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


def _mark_rule_selection(rows):
    by_date = {}
    for row in rows:
        if row.get("volume_pass") and row.get("gap_pass"):
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


def update_forward_test(price_df, code_to_name=None):
    code_to_name = code_to_name or {}
    rules = _load_json(RULES_PATH, {"versions": []})
    if not any(v.get("version") == RULE_VERSION for v in rules.get("versions", [])):
        rules.setdefault("versions", []).append(RULE_SNAPSHOT)
        _save_json(RULES_PATH, rules)

    payload = _load_json(LOG_PATH, {"started_at": "2026-09-09", "items": []})
    rows = payload.setdefault("items", [])
    keyed = {(str(r.get("code")), str(r.get("signal_date")), str(r.get("rule_version"))) for r in rows}

    groups = {}
    for code, group in price_df.groupby("Code"):
        groups[str(code)] = _prepare(group)

    # Add newly observed base-shape signals only from the forward-test start date.
    for code, g in groups.items():
        for i in range(25, len(g)):
            signal_date = g.Date.iloc[i]
            if signal_date < FORWARD_TEST_START:
                continue
            f = _base_features(g, i)
            if not f:
                continue
            key = (code, signal_date.strftime("%Y-%m-%d"), RULE_VERSION)
            if key in keyed:
                continue
            volume_pass = f["volume_ratio"] is not None and f["volume_ratio"] >= VOLUME_RATIO_MIN
            rows.append({
                "rule_version": RULE_VERSION,
                "code": code,
                "name": code_to_name.get(code, ""),
                "signal_date": signal_date.strftime("%Y-%m-%d"),
                "signal_close": round(float(g.C.iloc[i]), 3),
                **f,
                "volume_pass": bool(volume_pass),
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

    # Fill future facts as they become observable.
    for row in rows:
        if row.get("rule_version") != RULE_VERSION:
            continue
        g = groups.get(str(row.get("code")))
        if g is None or g.empty:
            continue
        matches = g.index[g.Date.dt.strftime("%Y-%m-%d") == row.get("signal_date")].tolist()
        if not matches:
            continue
        _update_future(row, g, matches[0])

    _mark_rule_selection(rows)
    _mark_actual_buys(rows)
    rows.sort(key=lambda r: (r.get("signal_date", ""), r.get("code", "")))
    payload["updated_through"] = (
        pd.to_datetime(price_df["Date"]).max().strftime("%Y-%m-%d") if not price_df.empty else None
    )
    _save_json(LOG_PATH, payload)
    logger.info(f"Forward test log updated: {len(rows)} rows")
