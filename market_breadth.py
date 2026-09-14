"""Market breadth journal for the production Prime-market universe.

The breadth numbers are derived from the same J-Quants daily bars used by the
production screener, so they can be reproduced later without relying on a web
page that may change.  Only dates on/after BREADTH_START are stored.
"""

import json
from pathlib import Path

import pandas as pd

from logger import get_logger

logger = get_logger(__name__)

BREADTH_START = pd.Timestamp("2026-09-09")
LOG_PATH = Path("docs/market_breadth_log.json")


def _load_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def update_market_breadth(price_df):
    """Store daily Prime advance/decline breadth derived from daily closes.

    `price_df` is expected to already be filtered to the production target
    universe (Prime common stocks after the normal excluded-code filtering).
    A stock is comparable when both today's and its previous available close
    are present and positive.
    """
    required = {"Code", "Date", "C"}
    if price_df is None or price_df.empty or not required.issubset(price_df.columns):
        logger.warning("Market breadth skipped: required price data is missing")
        return

    df = price_df[["Code", "Date", "C"]].copy()
    df["Code"] = df["Code"].astype(str)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["C"] = pd.to_numeric(df["C"], errors="coerce")
    df = df.dropna(subset=["Code", "Date", "C"])
    df = df[df["C"] > 0].sort_values(["Code", "Date"])

    if df.empty:
        logger.warning("Market breadth skipped: no valid closes")
        return

    df["prev_close"] = df.groupby("Code")["C"].shift(1)
    df["return_pct"] = (df["C"] / df["prev_close"] - 1) * 100
    df = df[(df["Date"] >= BREADTH_START) & df["prev_close"].notna() & (df["prev_close"] > 0)]

    payload = _load_json(
        LOG_PATH,
        {
            "started_at": BREADTH_START.strftime("%Y-%m-%d"),
            "source": "derived_from_jquants_prime_daily_bars",
            "universe": "production Prime universe after excluded-code filtering",
            "definition": "today close versus previous available close for each comparable stock",
            "days": [],
        },
    )

    existing = {str(r.get("date")): r for r in payload.get("days", []) if r.get("date")}

    for date, day in df.groupby("Date"):
        r = day["return_pct"].dropna().astype(float)
        if r.empty:
            continue

        advances = int((r > 0).sum())
        declines = int((r < 0).sum())
        unchanged = int((r == 0).sum())
        comparable = int(len(r))
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")

        existing[date_str] = {
            "date": date_str,
            "advances": advances,
            "declines": declines,
            "unchanged": unchanged,
            "comparable": comparable,
            "advance_pct": round(advances / comparable * 100, 2),
            "decline_pct": round(declines / comparable * 100, 2),
            "advance_decline_ratio": round(advances / declines, 3) if declines else None,
            "equal_weight_mean_return_pct": round(float(r.mean()), 3),
            "equal_weight_median_return_pct": round(float(r.median()), 3),
        }

    payload["days"] = [existing[k] for k in sorted(existing)]
    payload["updated_through"] = payload["days"][-1]["date"] if payload["days"] else None
    _save_json(LOG_PATH, payload)
    logger.info(
        "Market breadth log updated: %s days (through %s)",
        len(payload["days"]),
        payload.get("updated_through"),
    )
