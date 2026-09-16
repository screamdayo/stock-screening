"""Market breadth + screening frequency journal for the production Prime universe.

The analysis is derived from the same J-Quants daily bars used by the production
screener.  We keep enough history to test whether days with many kuitto candidates
are simply broad up-market days.
"""

import json
from pathlib import Path

import pandas as pd

from logger import get_logger
from strategies import kuitto_pullback_auto

logger = get_logger(__name__)

# 200 business days are fetched by main.py, so June gives us a comfortably long
# comparison window while leaving plenty of MA/turnover warm-up data before it.
BREADTH_START = pd.Timestamp("2026-06-01")
LOG_PATH = Path("docs/market_breadth_log.json")


def _load_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _candidate_counts(price_df):
    """Return candidate counts by signal date under two definitions.

    production_rule_count reproduces the strategy's date-effective rule: the
    liquidity filter starts on 2026-09-16.

    current_rule_count applies today's full rule, including 20-day average
    turnover >= 500m yen, to *all* historical dates.  This is the apples-to-apples
    series used for the breadth correlation analysis.
    """
    target_codes = set(price_df["Code"].astype(str).unique())
    signals, _ = kuitto_pullback_auto.find_signals(price_df, target_codes)
    production = {}
    current = {}

    for s in signals:
        date_str = pd.Timestamp(s["signal_date"]).strftime("%Y-%m-%d")
        production[date_str] = production.get(date_str, 0) + 1
        turnover = s.get("avg_turnover_20")
        if turnover is not None and float(turnover) >= kuitto_pullback_auto.AVG_TURNOVER_20_MIN:
            current[date_str] = current.get(date_str, 0) + 1

    return production, current


def _correlation_summary(days):
    rows = [
        r for r in days
        if r.get("advance_pct") is not None and r.get("current_rule_count") is not None
    ]
    if len(rows) < 3:
        return {"n_days": len(rows), "pearson_candidate_vs_advance_pct": None}

    x = pd.Series([float(r["current_rule_count"]) for r in rows])
    y = pd.Series([float(r["advance_pct"]) for r in rows])
    corr = x.corr(y)

    high = [r for r in rows if int(r["current_rule_count"]) >= 5]
    low = [r for r in rows if int(r["current_rule_count"]) < 5]

    def avg(group, key):
        if not group:
            return None
        return round(sum(float(r[key]) for r in group) / len(group), 2)

    return {
        "n_days": len(rows),
        "pearson_candidate_vs_advance_pct": round(float(corr), 3) if pd.notna(corr) else None,
        "days_with_5plus_candidates": len(high),
        "avg_advance_pct_on_5plus_days": avg(high, "advance_pct"),
        "avg_advance_pct_on_under5_days": avg(low, "advance_pct"),
        "avg_candidates_when_advance_pct_60plus": round(
            sum(int(r["current_rule_count"]) for r in rows if float(r["advance_pct"]) >= 60)
            / max(1, sum(1 for r in rows if float(r["advance_pct"]) >= 60)), 2
        ),
        "avg_candidates_when_advance_pct_under40": round(
            sum(int(r["current_rule_count"]) for r in rows if float(r["advance_pct"]) < 40)
            / max(1, sum(1 for r in rows if float(r["advance_pct"]) < 40)), 2
        ),
    }


def update_market_breadth(price_df):
    """Backfill breadth and candidate frequency from BREADTH_START onward."""
    required = {"Code", "Date", "C"}
    if price_df is None or price_df.empty or not required.issubset(price_df.columns):
        logger.warning("Market breadth skipped: required price data is missing")
        return

    source = price_df.copy()
    source["Code"] = source["Code"].astype(str)
    source["Date"] = pd.to_datetime(source["Date"], errors="coerce")

    production_counts, current_counts = _candidate_counts(source)

    df = source[["Code", "Date", "C"]].copy()
    df["C"] = pd.to_numeric(df["C"], errors="coerce")
    df = df.dropna(subset=["Code", "Date", "C"])
    df = df[df["C"] > 0].sort_values(["Code", "Date"])

    if df.empty:
        logger.warning("Market breadth skipped: no valid closes")
        return

    df["prev_close"] = df.groupby("Code")["C"].shift(1)
    df["return_pct"] = (df["C"] / df["prev_close"] - 1) * 100
    df = df[(df["Date"] >= BREADTH_START) & df["prev_close"].notna() & (df["prev_close"] > 0)]

    days = []
    for date, day in df.groupby("Date"):
        r = day["return_pct"].dropna().astype(float)
        if r.empty:
            continue

        advances = int((r > 0).sum())
        declines = int((r < 0).sum())
        unchanged = int((r == 0).sum())
        comparable = int(len(r))
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")

        days.append({
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
            "production_rule_count": int(production_counts.get(date_str, 0)),
            "current_rule_count": int(current_counts.get(date_str, 0)),
        })

    payload = {
        "started_at": BREADTH_START.strftime("%Y-%m-%d"),
        "source": "derived_from_jquants_prime_daily_bars",
        "universe": "production Prime universe after excluded-code filtering",
        "definition": "today close versus previous available close for each comparable stock",
        "candidate_count_definition": (
            "current_rule_count applies the 2026-09-16 full kuitto rule including "
            "20-day average turnover >= 500m yen to every historical date; "
            "production_rule_count preserves the date-effective production rule"
        ),
        "days": days,
        "analysis": _correlation_summary(days),
        "updated_through": days[-1]["date"] if days else None,
    }
    _save_json(LOG_PATH, payload)
    logger.info(
        "Market breadth/candidate log updated: %s days (through %s), corr=%s",
        len(days),
        payload.get("updated_through"),
        payload["analysis"].get("pearson_candidate_vs_advance_pct"),
    )
