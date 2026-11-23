"""Monthly Momentum6M candidate report (moved from stock-screening-data on 2026-11-23).

Runs in the public repository's workflow with the private J-Quants archive mounted under data/
(batches, daily_updates, margin_interest). The report is written to .momentum6m/latest.json, which is NOT
committed and only read by scripts/notify_momentum6m.py: the public repository and its Actions logs never
contain the candidate names or codes (only counts are printed).

Entry changed on 2026-11-23: buy limit = signal close x 1.10 (was the signal close). Source:
results/momentum6m_entry.json -- skipped gap-up names went on to large gains; +10% behaves like a market order
while still refusing an extreme opening gap.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from price_adjust import stale_basis_multiplier

ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = ROOT / "data" / "batches"
UPDATE_DIR = ROOT / "data" / "daily_updates"
WORK_DIR = ROOT / ".momentum6m"  # not committed (see .gitignore)
MASTER_PATH = WORK_DIR / "master_latest.parquet"
LATEST_OUT = WORK_DIR / "latest.json"
CAPITAL_PER_SLOT = 200_000
MAX_POSITIONS = 5
LOT_SIZE = 100
LOOKBACK_SESSIONS = 126
BUY_LIMIT_MULT = 1.10           # limit = signal close x 1.10 (results/momentum6m_entry.json, 2026-11-23)

# Exclusion filters adopted from results/new_condition_scan_summary.json (2026-11-07)
MARGIN_DIR = ROOT / "data" / "margin_interest"
LIQ_MIN_YEN = 100_000_000      # 20-session median trading value; the scan's universe
VOLSURGE_MAX = 2.0             # exclude if 20d avg trading value >= 2x the 120d average
SHORT_DAYS_MAX_PCT = 0.80      # exclude the top 20% by short interest / 20d avg volume
MARGIN_LAG_SESSIONS = 4        # weekly as-of Friday, published the following Tuesday
MARGIN_MAX_AGE_DAYS = 21       # older margin data -> short filter reported as unavailable


# Standard TSE tick sizes (non-TOPIX500). Their multiples are also valid ticks for TOPIX500 issues.
TICKS = [(3_000, 1), (5_000, 5), (30_000, 10), (50_000, 50), (300_000, 100), (500_000, 500),
         (3_000_000, 1_000), (5_000_000, 5_000), (30_000_000, 10_000), (50_000_000, 50_000)]


def floor_to_tick(price):
    """Round a limit price down to a valid tick so the order never exceeds the intended +10%."""
    tick = next((t for upper, t in TICKS if price <= upper), 100_000)
    return float(int(price // tick) * tick)


def previous_month(now):
    first = now.replace(day=1)
    return (first - pd.Timedelta(days=1)).strftime("%Y-%m")


def read_current_prime_codes():
    import os
    import requests

    api_key = os.environ.get("JQUANTS_API_KEY")
    if not api_key:
        raise RuntimeError("JQUANTS_API_KEY is required to refresh the current listing master")
    response = requests.get(
        "https://api.jquants.com/v2/equities/master",
        headers={"x-api-key": api_key},
        timeout=60,
    )
    response.raise_for_status()
    master = pd.DataFrame(response.json().get("data", []))
    if not {"Code", "MktNm"}.issubset(master.columns):
        raise RuntimeError(f"Unexpected master columns: {list(master.columns)}")
    master = master[master["MktNm"].isin(["プライム", "スタンダード", "グロース"])].copy()
    master = master[~master["Code"].astype(str).str.endswith(("5", "6"))].copy()
    master["Code"] = master["Code"].astype(str).map(
        lambda x: x[:-1] if len(x) == 5 and x.endswith("0") else x
    )
    master = master.drop_duplicates("Code", keep="last")
    MASTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    master.to_parquet(MASTER_PATH, index=False, compression="zstd")
    prime = master.loc[master["MktNm"].eq("プライム"), "Code"].astype(str)
    print(f"refreshed listing master: {len(master):,} listings; {len(prime):,} Prime", flush=True)
    return set(prime)


def load_prices(prime_codes):
    paths = sorted(BATCH_DIR.glob("batch_*.parquet")) + sorted(UPDATE_DIR.glob("*.parquet"))
    if not paths:
        raise RuntimeError("No archived or updated daily bars found")
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        if not {"Code", "Date"}.issubset(frame.columns):
            continue
        frame["Code"] = frame["Code"].astype(str)
        frame["Code"] = frame["Code"].map(
            lambda x: x[:-1] if len(x) == 5 and x.endswith("0") else x
        )
        frame = frame[frame["Code"].isin(prime_codes)]
        if frame.empty:
            continue
        close_col = "AdjC" if "AdjC" in frame.columns else "C"
        if close_col not in frame.columns:
            raise RuntimeError(f"No adjusted close in {path}")
        keep = ["Code", "Date", close_col] + [c for c in ("Va", "Vo") if c in frame.columns]
        part = frame[keep].rename(columns={close_col: "close", "Va": "value_yen", "Vo": "volume_raw"})
        # raw close + AdjFactor let load_prices() detect archived rows that predate a later split
        part["close_raw"] = frame["C"] if "C" in frame.columns else part["close"]
        part["AdjFactor"] = frame["AdjFactor"] if "AdjFactor" in frame.columns else 1.0
        frames.append(part)
        print(f"loaded {path.name}: {len(part):,} Prime rows", flush=True)
    if not frames:
        raise RuntimeError("No bars matched current Prime master")
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"])
    for c in ("close", "value_yen", "volume_raw"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df = df.dropna(subset=["Date", "close"])
    df = df[df["close"] > 0].drop_duplicates(["Code", "Date"], keep="last")
    df = df.sort_values(["Code", "Date"]).reset_index(drop=True)
    # The archive's adjusted close predates splits that went ex after it was fetched; without this a
    # split looks like a crash in mom126 for the next 126 sessions (see price_adjust.py).
    stale = stale_basis_multiplier(df, raw_col="close_raw", adj_col="close")
    fixed = df.loc[stale.ne(1.0), "Code"].nunique()
    if fixed:
        print(f"re-based stale adjusted closes for {fixed} codes", flush=True)
    df["close"] = df["close"] * stale
    return df.drop(columns=["close_raw", "AdjFactor"])


def load_short_interest(codes, trading_dates, signal_date):
    """Latest margin short balance usable at signal_date (as-of date + MARGIN_LAG_SESSIONS sessions)."""
    paths = sorted(MARGIN_DIR.glob("*.parquet"))[-2:]
    if not paths:
        return pd.Series(dtype=float), None
    m = pd.concat([pd.read_parquet(p, columns=["Date", "Code", "ShrtVol"]) for p in paths], ignore_index=True)
    m["Code"] = m["Code"].astype(str).map(lambda x: x[:-1] if len(x) == 5 and x.endswith("0") else x)
    m["Date"] = pd.to_datetime(m["Date"], errors="coerce")
    m["ShrtVol"] = pd.to_numeric(m["ShrtVol"], errors="coerce")
    m = m.dropna(subset=["Date"])
    m = m[m["Code"].isin(codes)]
    tdates = pd.DatetimeIndex(sorted(trading_dates))
    pos = tdates.searchsorted(m["Date"].values) + MARGIN_LAG_SESSIONS
    # Rows whose availability date lies beyond the last loaded session are not usable yet (NaT).
    m["avail"] = pd.to_datetime([tdates[p] if p < len(tdates) else pd.NaT for p in pos])
    m = m[m["avail"].notna() & (m["avail"] <= signal_date)]
    if m.empty:
        return pd.Series(dtype=float), None
    latest = m.sort_values("Date").drop_duplicates("Code", keep="last")
    return latest.set_index("Code")["ShrtVol"], latest["Date"].max()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal-month", help="Target completed month in YYYY-MM; defaults to previous Japan calendar month")
    args = parser.parse_args()

    now = datetime.now(ZoneInfo("Asia/Tokyo"))
    month = args.signal_month or previous_month(now)
    try:
        target_period = pd.Period(month, freq="M")
    except Exception as exc:
        raise RuntimeError("--signal-month must be YYYY-MM") from exc

    prime_codes = read_current_prime_codes()
    df = load_prices(prime_codes)
    group = df.groupby("Code", sort=False)
    df["mom126"] = group["close"].transform(lambda s: s / s.shift(LOOKBACK_SESSIONS) - 1)
    df["liq20_median_yen"] = group["value_yen"].transform(lambda s: s.rolling(20).median())
    df["volsurge"] = group["value_yen"].transform(
        lambda s: s.rolling(20).mean() / s.rolling(120, min_periods=100).mean()
    )
    df["adv20_shares"] = group["volume_raw"].transform(lambda s: s.rolling(20, min_periods=15).mean())

    month_rows = df[df["Date"].dt.to_period("M").eq(target_period)]
    if month_rows.empty:
        raise RuntimeError(f"No market data found for target month {month}")
    signal_date = month_rows["Date"].max()
    snap = df[df["Date"].eq(signal_date)].dropna(subset=["mom126", "close"]).copy()
    snap = snap[snap["liq20_median_yen"] >= LIQ_MIN_YEN].copy()

    # Short interest relative to volume, ranked within the liquid Prime universe
    short_vol, margin_asof = load_short_interest(set(snap["Code"]), df["Date"].drop_duplicates(), signal_date)
    margin_age = (signal_date - margin_asof).days if margin_asof is not None else None
    short_filter_on = margin_age is not None and margin_age <= MARGIN_MAX_AGE_DAYS
    snap["short_days"] = snap["Code"].map(short_vol) / snap["adv20_shares"]
    snap["short_days_pct"] = snap["short_days"].rank(pct=True)

    snap = snap.sort_values(["mom126", "Code"], ascending=[False, True]).reset_index(drop=True)
    snap["raw_rank"] = range(1, len(snap) + 1)
    reasons = []
    for _, row in snap.iterrows():
        why = []
        if pd.isna(row["volsurge"]) or row["volsurge"] >= VOLSURGE_MAX:
            why.append("volume_surge" if pd.notna(row["volsurge"]) else "volume_history_short")
        if short_filter_on and pd.notna(row["short_days_pct"]) and row["short_days_pct"] > SHORT_DAYS_MAX_PCT:
            why.append("heavy_short_interest")
        reasons.append(why)
    snap["excluded_by"] = reasons
    excluded = snap[snap["excluded_by"].map(bool)].head(20)
    snap = snap[~snap["excluded_by"].map(bool)].reset_index(drop=True)
    snap["rank"] = range(1, len(snap) + 1)

    next_dates = sorted(pd.DatetimeIndex(df["Date"].drop_duplicates()).to_list())
    later = [d for d in next_dates if d > signal_date]
    next_observed_session = later[0].date().isoformat() if later else None

    master = pd.read_parquet(MASTER_PATH)
    master["Code"] = master["Code"].astype(str).map(
        lambda x: x[:-1] if len(x) == 5 and x.endswith("0") else x
    )
    name_col = next((c for c in ("CoName", "CompanyName", "Name") if c in master.columns), None)
    names = dict(zip(master["Code"], master[name_col])) if name_col else {}

    candidates = []
    for _, row in snap.head(20).iterrows():
        price = float(row["close"])
        limit = floor_to_tick(price * BUY_LIMIT_MULT)
        shares = int(CAPITAL_PER_SLOT // (price * LOT_SIZE)) * LOT_SIZE  # sized on the close; median opening gap is ~0%
        candidates.append({
            "rank": int(row["rank"]),
            "raw_rank": int(row["raw_rank"]),
            "code": str(row["Code"]),
            "name": str(names.get(str(row["Code"]), "")),
            "momentum_126_sessions_pct": round(float(row["mom126"]) * 100, 3),
            "volume_surge_ratio": round(float(row["volsurge"]), 3),
            "short_days": None if pd.isna(row["short_days"]) else round(float(row["short_days"]), 3),
            "short_days_percentile": None if pd.isna(row["short_days_pct"]) else round(float(row["short_days_pct"]), 3),
            "signal_close_yen": round(price, 2),
            "buy_limit_yen": limit,
            "shares_at_200k_cap": shares,
            "estimated_order_value_yen": round(shares * price, 2),
            "max_order_value_at_limit_yen": round(shares * limit, 2),
            "priority_slot": int(row["rank"]) <= MAX_POSITIONS,
        })

    result = {
        "strategy": "Momentum6M monthly candidate report",
        "status": "candidate list only; no Discord notification and no order placement",
        "generated_at_jst": now.isoformat(timespec="seconds"),
        "signal_month": str(target_period),
        "signal_date": signal_date.date().isoformat(),
        "next_observed_market_session": next_observed_session,
        "universe": "current Prime listings from master_latest.parquet with 20-session median trading value >= 100M yen",
        "rule": {
            "rank": "126-session adjusted-close return descending, after the exclusion filters",
            "exclusions": {
                "volume_surge": f"20-session avg trading value >= {VOLSURGE_MAX}x the 120-session avg",
                "heavy_short_interest": f"margin short balance / 20-session avg volume above the {int(SHORT_DAYS_MAX_PCT * 100)}th percentile of the liquid Prime universe; margin data lagged {MARGIN_LAG_SESSIONS} sessions",
                "source": "results/new_condition_scan_summary.json (2026-11-07 research; t about 2.5, discount for selection)",
            },
            "entry": f"next session buy limit at signal-date close x {BUY_LIMIT_MULT:.2f}; if it does not fill, leave the slot vacant",
            "exit": "first session open strictly after the next calendar month-end",
            "capital_yen": 1_000_000,
            "max_positions": MAX_POSITIONS,
            "max_yen_per_position": CAPITAL_PER_SLOT,
            "lot_size": LOT_SIZE,
            "indicators_added": ["liquidity_floor", "volume_surge_exclusion", "short_interest_exclusion"],
        },
        "short_interest_filter": {
            "applied": short_filter_on,
            "margin_asof_date": margin_asof.date().isoformat() if margin_asof is not None else None,
            "margin_age_days": margin_age,
            "note": None if short_filter_on else f"margin data missing or older than {MARGIN_MAX_AGE_DAYS} days; only the volume-surge filter was applied",
        },
        "notes": [
            "Top 5 rows are the priority list; rows 6-20 are reference ranks, not extra buy orders.",
            "If a priority order is unfilled, this live report does not automatically replace it with a lower rank.",
            "Limit is +10% above the signal close (changed 2026-11-23): gap-ups are bought unless the open is extreme.",
            "Point-in-time re-check incl. delisted issues (results/momentum6m_pit.json): edge vs the equal-weight universe is small and not significant; expect large swings.",
            "The historical backtest used next-rank replacement after a gap-up; this report avoids assuming an executable same-open replacement.",
            "No stop loss was added; the tested exit timing is preserved.",
            "Current Prime universe may omit delisted or formerly Prime listings."
        ],
        "candidate_count": len(snap),
        "displayed_count": len(candidates),
        "candidates": candidates,
        "excluded_high_momentum": [
            {
                "raw_rank": int(r["raw_rank"]),
                "code": str(r["Code"]),
                "name": str(names.get(str(r["Code"]), "")),
                "momentum_126_sessions_pct": round(float(r["mom126"]) * 100, 3),
                "excluded_by": r["excluded_by"],
                "volume_surge_ratio": None if pd.isna(r["volsurge"]) else round(float(r["volsurge"]), 3),
                "short_days_percentile": None if pd.isna(r["short_days_pct"]) else round(float(r["short_days_pct"]), 3),
            }
            for _, r in excluded.iterrows()
        ],
    }
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Public Actions logs: counts only, never names or codes.
    print(json.dumps({
        "signal_month": str(target_period),
        "signal_date": signal_date.date().isoformat(),
        "candidates": len(snap),
        "displayed": len(candidates),
        "short_interest_filter_applied": short_filter_on,
        "excluded_in_raw_top20": int((excluded["raw_rank"] <= 20).sum()) if "raw_rank" in excluded else 0,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
