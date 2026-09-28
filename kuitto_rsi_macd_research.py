"""Prospective research-only journal for relaxed Kuitto + RSI35 + deep MACD GC.

No trading or Discord notification. Frozen research rule:
- relaxed Kuitto profile 1
- RSI14 crosses above 35 on signal day
- MACD(12,26,9) bullish cross within the last 3 sessions
- MACD / close <= -1.5% on signal day

New rows are appended only when the signal occurs prospectively. Existing rows are
updated later with next-open entry, 5/10/15-day returns, and MFE/MAE.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from logger import get_logger

logger = get_logger(__name__)

LOG_PATH = Path("docs/kuitto_rsi_macd_research_log.json")
RULES_PATH = Path("docs/kuitto_rsi_macd_research_rules.json")
RULE_VERSION = "relaxed1_rsi35_macddeep_v1_2026-09-29"

RULE_SNAPSHOT = {
    "version": RULE_VERSION,
    "effective_from": "2026-09-29",
    "purpose": "research_only",
    "entry_reference": "next trading-day open",
    "kuitto_relaxed1": {
        "ma5_prior5d_decline_pct": [-6.5, -1.5],
        "ma5_vs_ma25_pct": [-6.0, 0.5],
        "close_vs_ma5_max_pct": 4.5,
        "bull_candle_pct": [1.0, 4.0],
        "volume_ratio_min": 1.10,
        "ma5_turn": "previous 2 sessions non-rising, signal day rising",
    },
    "rsi": {"period": 14, "condition": "cross above 35 on signal day"},
    "macd": {
        "fast": 12, "slow": 26, "signal": 9,
        "condition": "bullish cross within last 3 sessions and MACD/close <= -1.5%",
    },
    "forward_metrics": ["next_open_gap_pct", "return_5d_pct", "return_10d_pct", "return_15d_pct", "mfe_15d_pct", "mae_15d_pct"],
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

def _prepare(group):
    g = group.sort_values("Date").reset_index(drop=True).copy()
    g["Date"] = pd.to_datetime(g["Date"])
    for c in ["O", "H", "L", "C", "Vo"]:
        g[c] = pd.to_numeric(g[c], errors="coerce")

    g["MA5"] = g["C"].rolling(5).mean()
    g["MA25"] = g["C"].rolling(25).mean()
    g["DECL5"] = (g["MA5"].shift(1) / g["MA5"].shift(6) - 1) * 100
    g["MA_GAP"] = (g["MA5"] / g["MA25"] - 1) * 100
    g["CLOSE_MA5"] = (g["C"] / g["MA5"] - 1) * 100
    g["BULL"] = (g["C"] / g["O"] - 1) * 100
    g["VOLR"] = g["Vo"] / g["Vo"].shift(1)
    g["TURN"] = (
        (g["MA5"] > g["MA5"].shift(1))
        & (g["MA5"].shift(1) <= g["MA5"].shift(2))
        & (g["MA5"].shift(2) <= g["MA5"].shift(3))
    )

    delta = g["C"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    g["RSI14"] = 100 - (100 / (1 + rs))
    g["RSI_CROSS35"] = (g["RSI14"] > 35) & (g["RSI14"].shift(1) <= 35)

    ema12 = g["C"].ewm(span=12, adjust=False).mean()
    ema26 = g["C"].ewm(span=26, adjust=False).mean()
    g["MACD"] = ema12 - ema26
    g["MACD_SIGNAL"] = g["MACD"].ewm(span=9, adjust=False).mean()
    g["MACD_CROSS"] = (g["MACD"] > g["MACD_SIGNAL"]) & (g["MACD"].shift(1) <= g["MACD_SIGNAL"].shift(1))
    g["MACD_CROSS3"] = g["MACD_CROSS"] | g["MACD_CROSS"].shift(1).fillna(False) | g["MACD_CROSS"].shift(2).fillna(False)
    g["MACD_PCT"] = g["MACD"] / g["C"] * 100
    return g

def _is_signal(g, i):
    if i < 30:
        return False
    r = g.iloc[i]
    needed = ["MA5","MA25","DECL5","MA_GAP","CLOSE_MA5","BULL","VOLR","RSI14","MACD_PCT"]
    if any(pd.isna(r[x]) for x in needed):
        return False
    return bool(
        r["TURN"]
        and -6.5 <= r["DECL5"] <= -1.5
        and -6.0 <= r["MA_GAP"] <= 0.5
        and r["CLOSE_MA5"] <= 4.5
        and 1.0 <= r["BULL"] <= 4.0
        and r["VOLR"] >= 1.10
        and r["RSI_CROSS35"]
        and r["MACD_CROSS3"]
        and r["MACD_PCT"] <= -1.5
    )

def _update_future(row, g):
    signal_date = pd.Timestamp(row["signal_date"])
    matches = g.index[g["Date"] == signal_date].tolist()
    if not matches:
        return
    i = matches[-1]
    ent = i + 1
    if ent >= len(g):
        return
    entry = float(g["O"].iloc[ent])
    close = float(g["C"].iloc[i])
    if not entry > 0 or not close > 0:
        return

    row["entry_date"] = g["Date"].iloc[ent].strftime("%Y-%m-%d")
    row["entry_open"] = round(entry, 3)
    row["next_open_gap_pct"] = round((entry / close - 1) * 100, 3)

    for days in (5, 10, 15):
        idx = ent + days - 1
        if idx < len(g):
            row[f"return_{days}d_pct"] = _pct(g["C"].iloc[idx], entry)

    observed_end = min(ent + 14, len(g) - 1)
    observed = g.iloc[ent:observed_end + 1]
    if not observed.empty:
        row["mfe_15d_pct"] = round((float(observed["H"].max()) / entry - 1) * 100, 3)
        row["mae_15d_pct"] = round((float(observed["L"].min()) / entry - 1) * 100, 3)

def update(price_df, code_to_name=None):
    code_to_name = code_to_name or {}
    latest = pd.Timestamp(price_df["Date"].max()).normalize()

    rules = _load(RULES_PATH, {"versions": []})
    if not any(v.get("version") == RULE_VERSION for v in rules.get("versions", [])):
        rules.setdefault("versions", []).append(RULE_SNAPSHOT)
        _save(RULES_PATH, rules)

    payload = _load(LOG_PATH, {"started_at": "2026-09-29", "items": []})
    rows = payload.setdefault("items", [])
    keyed = {(str(r.get("code")), r.get("signal_date"), r.get("rule_version")) for r in rows}

    prepared = {}
    added = 0
    for code, group in price_df.groupby("Code"):
        g = _prepare(group)
        code = str(code)
        prepared[code] = g
        if g.empty or pd.Timestamp(g["Date"].iloc[-1]).normalize() != latest:
            continue
        i = len(g) - 1
        if not _is_signal(g, i):
            continue
        day = latest.strftime("%Y-%m-%d")
        key = (code, day, RULE_VERSION)
        if key in keyed:
            continue
        r = g.iloc[i]
        rows.append({
            "strategy": "kuitto_relaxed_rsi_macd_research",
            "rule_version": RULE_VERSION,
            "code": code,
            "name": code_to_name.get(code, ""),
            "signal_date": day,
            "signal_close": round(float(r["C"]), 3),
            "ma5_prior5d_decline_pct": round(float(r["DECL5"]), 3),
            "ma5_vs_ma25_pct": round(float(r["MA_GAP"]), 3),
            "close_vs_ma5_pct": round(float(r["CLOSE_MA5"]), 3),
            "bull_candle_pct": round(float(r["BULL"]), 3),
            "volume_ratio": round(float(r["VOLR"]), 3),
            "rsi14": round(float(r["RSI14"]), 3),
            "macd": round(float(r["MACD"]), 6),
            "macd_signal": round(float(r["MACD_SIGNAL"]), 6),
            "macd_pct": round(float(r["MACD_PCT"]), 4),
            "macd_cross_within3": True,
            "entry_date": None,
            "entry_open": None,
            "next_open_gap_pct": None,
            "return_5d_pct": None,
            "return_10d_pct": None,
            "return_15d_pct": None,
            "mfe_15d_pct": None,
            "mae_15d_pct": None,
        })
        keyed.add(key)
        added += 1

    for row in rows:
        g = prepared.get(str(row.get("code")))
        if g is not None:
            _update_future(row, g)

    rows.sort(key=lambda r: (r.get("signal_date", ""), str(r.get("code", ""))))
    payload["updated_through"] = latest.strftime("%Y-%m-%d")
    _save(LOG_PATH, payload)
    logger.info("RSI×MACD research log updated: %d rows (+%d today)", len(rows), added)
