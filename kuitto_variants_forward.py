"""Forward journal for Kuitto 204 and Kuitto 59 shadow strategies.

These are intentionally separate from the production Kuitto strategy.
Only signals generated prospectively from the frozen 2026-09-23 definitions are appended.
Existing rows are only updated with future market data as it becomes available.
"""

import json
from pathlib import Path

import pandas as pd

LOG_PATH = Path("docs/kuitto_variants_forward.json")
RULES_PATH = Path("docs/kuitto_variants_rules.json")

HOLD_DAYS = 16

RULES = {
    "kuitto_refined_204": {
        "version": "kuitto_refined_204_v1_2026-09-23",
        "label": "くいっと204",
        "entry": "next trading-day open",
        "exit": "open after 16 trading sessions from entry",
        "stop_loss": None,
    },
    "kuitto_elite_59": {
        "version": "kuitto_elite_59_v1_2026-09-23",
        "label": "くいっと59",
        "entry": "next trading-day open",
        "exit": "open after 16 trading sessions from entry",
        "stop_loss": None,
    },
}


def _load(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _pct(px, entry):
    return round((float(px) / float(entry) - 1) * 100, 3)


def _append(rows, keyed, strategy, results, signal_date, code_to_name):
    rule = RULES[strategy]
    day = pd.Timestamp(signal_date).strftime("%Y-%m-%d")
    for r in results or []:
        code = str(r.get("code"))
        key = (strategy, code, day, rule["version"])
        if key in keyed:
            continue
        rows.append({
            "strategy": strategy,
            "rule_version": rule["version"],
            "label": rule["label"],
            "code": code,
            "name": r.get("name") or code_to_name.get(code, ""),
            "signal_date": day,
            "signal_close": r.get("close"),
            "bull_candle_pct": r.get("bull_candle_pct"),
            "ma5_prior5d_decline_pct": r.get("ma5_prior5d_decline_pct"),
            "ma5_vs_ma25_pct": r.get("ma5_vs_ma25_pct"),
            "close_vs_ma5_pct": r.get("close_vs_ma5_pct"),
            "volume_ratio": r.get("volume_ratio"),
            "atr14_pct": r.get("atr14_pct"),
            "dd20_pct": r.get("dd20_pct"),
            "ma25_slope5_pct": r.get("ma25_slope5_pct"),
            "entry_date": None,
            "entry_open": None,
            "return_5d_pct": None,
            "return_10d_pct": None,
            "return_16d_pct": None,
            "mfe_16d_pct": None,
            "mae_16d_pct": None,
            "exit_date": None,
            "exit_price": None,
            "exit_pnl_pct": None,
        })
        keyed.add(key)


def _update_row(row, g):
    signal_date = pd.Timestamp(row["signal_date"])
    matches = g.index[g["Date"] == signal_date].tolist()
    if not matches:
        return

    signal_i = matches[-1]
    entry_i = signal_i + 1
    if entry_i >= len(g):
        return

    entry = float(g["O"].iloc[entry_i])
    if not entry > 0:
        return

    row["entry_date"] = pd.Timestamp(g["Date"].iloc[entry_i]).strftime("%Y-%m-%d")
    row["entry_open"] = round(entry, 3)

    for days in (5, 10, 16):
        exit_i = entry_i + days
        if exit_i < len(g):
            row[f"return_{days}d_pct"] = _pct(g["O"].iloc[exit_i], entry)

    observed_end = min(entry_i + HOLD_DAYS, len(g) - 1)
    observed = g.iloc[entry_i:observed_end + 1]
    if not observed.empty:
        row["mfe_16d_pct"] = round((float(observed["H"].max()) / entry - 1) * 100, 3)
        row["mae_16d_pct"] = round((float(observed["L"].min()) / entry - 1) * 100, 3)

    exit_i = entry_i + HOLD_DAYS
    if exit_i < len(g):
        px = float(g["O"].iloc[exit_i])
        row["exit_date"] = pd.Timestamp(g["Date"].iloc[exit_i]).strftime("%Y-%m-%d")
        row["exit_price"] = round(px, 3)
        row["exit_pnl_pct"] = _pct(px, entry)


def update(price_df, code_to_name, refined_results, elite_results):
    latest_signal_date = pd.Timestamp(price_df["Date"].max())

    rules_payload = _load(RULES_PATH, {"frozen_at": "2026-09-23", "strategies": {}})
    for name, rule in RULES.items():
        rules_payload.setdefault("strategies", {})[name] = rule
    _save(RULES_PATH, rules_payload)

    payload = _load(LOG_PATH, {"started_at": "2026-09-23", "items": []})
    rows = payload.setdefault("items", [])
    keyed = {
        (r.get("strategy"), str(r.get("code")), r.get("signal_date"), r.get("rule_version"))
        for r in rows
    }

    _append(rows, keyed, "kuitto_refined_204", refined_results, latest_signal_date, code_to_name)
    _append(rows, keyed, "kuitto_elite_59", elite_results, latest_signal_date, code_to_name)

    prepared = {}
    for code, group in price_df.groupby("Code"):
        g = group.sort_values("Date").reset_index(drop=True).copy()
        g["Date"] = pd.to_datetime(g["Date"])
        for c in ["O", "H", "L", "C"]:
            g[c] = pd.to_numeric(g[c], errors="coerce")
        prepared[str(code)] = g

    for row in rows:
        g = prepared.get(str(row.get("code")))
        if g is not None:
            _update_row(row, g)

    rows.sort(key=lambda r: (r.get("signal_date", ""), r.get("strategy", ""), str(r.get("code", ""))))
    payload["updated_at"] = latest_signal_date.strftime("%Y-%m-%d")
    _save(LOG_PATH, payload)
