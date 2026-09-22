"""Persist the production relationship between Selling Climax and Pinch sensors."""

import json
from pathlib import Path
import pandas as pd

from logger import get_logger

logger = get_logger(__name__)

STATE_PATH = Path("docs/crash_cycle_state.json")
MAX_WAIT_BUSINESS_DAYS = 30


def _load():
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "status": "idle",
            "selling_climax_date": None,
            "last_selling_signal_date": None,
            "pinch_date": None,
            "business_days_since_selling": None,
            "confirmed_after_business_days": None,
        }


def _save(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _business_days_between(price_df, start_date, end_date):
    if not start_date or not end_date or price_df is None or price_df.empty:
        return None
    days = (
        pd.to_datetime(price_df["Date"], errors="coerce")
        .dropna()
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
    )
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    # signal day -> later signal day: count trading sessions after start through end.
    return int(((days > start) & (days <= end)).sum())


def update_state(price_df, selling_sensor, pinch_sensor):
    state = _load()
    latest_date = pd.to_datetime(price_df["Date"], errors="coerce").max().normalize()
    latest = latest_date.strftime("%Y-%m-%d")

    selling_active = bool((selling_sensor or {}).get("active"))
    pinch_active = bool((pinch_sensor or {}).get("active"))
    selling_date = (
        pd.Timestamp(selling_sensor.get("signal_date")).strftime("%Y-%m-%d")
        if selling_active else None
    )
    pinch_date = (
        pd.Timestamp(pinch_sensor.get("signal_date")).strftime("%Y-%m-%d")
        if pinch_active else None
    )

    # A new selling-climax starts a cycle only when we are not already waiting.
    # Repeated selling-climax signals inside the same crash do not reset the clock.
    if selling_active:
        if state.get("status") != "waiting_for_pinch":
            state.update({
                "status": "waiting_for_pinch",
                "selling_climax_date": selling_date,
                "pinch_date": None,
                "confirmed_after_business_days": None,
            })
        state["last_selling_signal_date"] = selling_date

    if state.get("status") == "waiting_for_pinch" and state.get("selling_climax_date"):
        elapsed = _business_days_between(
            price_df, state.get("selling_climax_date"), latest
        )
        state["business_days_since_selling"] = elapsed

        if pinch_active:
            confirmed = _business_days_between(
                price_df, state.get("selling_climax_date"), pinch_date
            )
            state.update({
                "status": "pinch_confirmed",
                "pinch_date": pinch_date,
                "business_days_since_selling": confirmed,
                "confirmed_after_business_days": confirmed,
            })
        elif elapsed is not None and elapsed > MAX_WAIT_BUSINESS_DAYS:
            state["status"] = "expired"

    elif pinch_active:
        # A Pinch can occur without a preceding Selling Climax (2022-type case).
        state.update({
            "status": "pinch_without_selling",
            "selling_climax_date": None,
            "last_selling_signal_date": None,
            "pinch_date": pinch_date,
            "business_days_since_selling": None,
            "confirmed_after_business_days": None,
        })

    state["updated_through"] = latest
    state["max_wait_business_days"] = MAX_WAIT_BUSINESS_DAYS
    state["label"] = {
        "idle": "待機中",
        "waiting_for_pinch": "セリクラ発動 → ピンチ確認待ち",
        "pinch_confirmed": "セリクラ → ピンチ確認済み",
        "pinch_without_selling": "ピンチ単独発動",
        "expired": "セリクラ後30営業日以内にピンチ未確認",
    }.get(state.get("status"), state.get("status"))

    _save(state)
    logger.info(
        "暴落サイクル状態: %s / セリクラ=%s / ピンチ=%s / 経過=%s営業日",
        state.get("status"),
        state.get("selling_climax_date"),
        state.get("pinch_date"),
        state.get("business_days_since_selling"),
    )
    return state
