"""Shadow journal for strict-gakutto exits on broad market selloff days.

Production exit behavior is unchanged. This module only records cases where:
- strict_gakutto is confirmed for a currently held stock,
- TOPIX daily return <= -1%, and
- production-Prime decline ratio >= 70%.

Later daily runs fill the normal next-open exit and +1/+3/+5/+10-session
counterfactual open prices so the rescue idea can be evaluated out of sample.
"""

import json
from datetime import timedelta
from pathlib import Path

import pandas as pd

import download
from logger import get_logger

logger = get_logger(__name__)

LOG_PATH = Path("docs/gakutto_market_rescue_shadow_log.json")
TOPIX_MAX_PCT = -1.0
DECLINE_MIN_PCT = 70.0
DELAYS = (1, 3, 5, 10)


def _load_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _latest_breadth(price_df, latest_date):
    d = price_df[["Code", "Date", "C"]].copy()
    d["Code"] = d["Code"].astype(str)
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d["C"] = pd.to_numeric(d["C"], errors="coerce")
    d = d.dropna(subset=["Code", "Date", "C"])
    d = d[d["C"] > 0].sort_values(["Code", "Date"])
    d["prev_close"] = d.groupby("Code")["C"].shift(1)
    day = d[d["Date"].dt.normalize() == pd.Timestamp(latest_date).normalize()].copy()
    day = day[day["prev_close"].notna() & (day["prev_close"] > 0)]
    if day.empty:
        return None
    day["ret"] = day["C"] / day["prev_close"] - 1
    comparable = int(len(day))
    declines = int((day["ret"] < 0).sum())
    return {
        "declines": declines,
        "comparable": comparable,
        "decline_pct": round(declines / comparable * 100, 3),
    }


def _fetch_topix_return(latest_date):
    start = (pd.Timestamp(latest_date) - timedelta(days=14)).strftime("%Y-%m-%d")
    end = pd.Timestamp(latest_date).strftime("%Y-%m-%d")
    res = download._request_with_retry(
        f"{download.BASE_URL}/indices/bars/daily/topix",
        params={"from": start, "to": end},
    )
    res.raise_for_status()
    rows = res.json().get("data", [])
    d = pd.DataFrame(rows)
    if d.empty:
        raise RuntimeError("TOPIX日足が取得できませんでした")
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d["C"] = pd.to_numeric(d["C"], errors="coerce")
    d = d.dropna(subset=["Date", "C"]).sort_values("Date").reset_index(drop=True)
    d["RET1"] = d["C"].pct_change() * 100
    latest = d.iloc[-1]
    if pd.Timestamp(latest["Date"]).normalize() != pd.Timestamp(latest_date).normalize():
        raise RuntimeError(
            f"TOPIX最新日が株価最新日と不一致 ({latest['Date'].date()} != {pd.Timestamp(latest_date).date()})"
        )
    if pd.isna(latest["RET1"]):
        raise RuntimeError("TOPIX前日データが不足しています")
    return round(float(latest["RET1"]), 3)


def _prepare_groups(price_df):
    groups = {}
    d = price_df.copy()
    d["Code"] = d["Code"].astype(str)
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d["O"] = pd.to_numeric(d["O"], errors="coerce")
    d["C"] = pd.to_numeric(d["C"], errors="coerce")
    for code, g in d.groupby("Code"):
        groups[str(code)] = (
            g.dropna(subset=["Date", "O"])
            .sort_values("Date")
            .drop_duplicates(subset=["Date"], keep="last")
            .reset_index(drop=True)
        )
    return groups


def _pct(px, base):
    if base is None:
        return None
    try:
        px = float(px)
        base = float(base)
    except (TypeError, ValueError):
        return None
    if not base > 0:
        return None
    return round((px / base - 1) * 100, 3)


def _fill_future_observations(item, g):
    matches = g.index[
        g["Date"].dt.strftime("%Y-%m-%d") == str(item.get("signal_date"))
    ].tolist()
    if not matches:
        return False

    changed = False
    j = matches[-1]
    normal_idx = j + 1
    if normal_idx >= len(g):
        return False

    normal_open = float(g["O"].iloc[normal_idx])
    normal_date = g["Date"].iloc[normal_idx].strftime("%Y-%m-%d")
    if item.get("normal_exit_date") != normal_date:
        item["normal_exit_date"] = normal_date
        changed = True
    if item.get("normal_exit_open") != round(normal_open, 3):
        item["normal_exit_open"] = round(normal_open, 3)
        changed = True

    normal_trade_ret = _pct(normal_open, item.get("buy_price"))
    if item.get("normal_exit_trade_return_pct") != normal_trade_ret:
        item["normal_exit_trade_return_pct"] = normal_trade_ret
        changed = True

    observations = item.setdefault("shadow_delays", {})
    for delay in DELAYS:
        idx = normal_idx + delay
        if idx >= len(g):
            continue
        px = float(g["O"].iloc[idx])
        date_str = g["Date"].iloc[idx].strftime("%Y-%m-%d")
        value = {
            "date": date_str,
            "open": round(px, 3),
            "edge_vs_normal_exit_pct": _pct(px, normal_open),
            "trade_return_pct": _pct(px, item.get("buy_price")),
        }
        key = str(delay)
        if observations.get(key) != value:
            observations[key] = value
            changed = True
    return changed


def update(price_df, sell_alerts, code_to_name=None):
    """Append qualifying events and progress prior shadow observations.

    No production exit decision is changed by this function.
    """
    code_to_name = code_to_name or {}
    if price_df is None or price_df.empty:
        return

    payload = _load_json(
        LOG_PATH,
        {
            "started_at": "2026-09-29",
            "mode": "shadow_only_no_trade_rule_change",
            "condition": {
                "topix_return_pct_max": TOPIX_MAX_PCT,
                "decline_pct_min": DECLINE_MIN_PCT,
            },
            "delay_definition": (
                "delay N = N extra trading sessions after the normal strict-gakutto next-open exit"
            ),
            "delays": list(DELAYS),
            "items": [],
        },
    )
    rows = payload.setdefault("items", [])
    groups = _prepare_groups(price_df)
    changed = False

    # Progress already-recorded events even after the real holding has been sold.
    for item in rows:
        g = groups.get(str(item.get("code")))
        if g is not None and not g.empty:
            changed = _fill_future_observations(item, g) or changed

    strict_alerts = [a for a in (sell_alerts or []) if a.get("reason") == "strict_gakutto"]
    if strict_alerts:
        latest_date = pd.to_datetime(price_df["Date"], errors="coerce").max()
        if pd.notna(latest_date):
            breadth = _latest_breadth(price_df, latest_date)
            try:
                topix_ret = _fetch_topix_return(latest_date)
            except Exception as e:
                logger.warning("がくっと全面安ログ: TOPIX取得失敗: %s", e)
                topix_ret = None

            decline_pct = breadth.get("decline_pct") if breadth else None
            market_hit = bool(
                topix_ret is not None
                and decline_pct is not None
                and topix_ret <= TOPIX_MAX_PCT
                and decline_pct >= DECLINE_MIN_PCT
            )

            if market_hit:
                known = {
                    (
                        str(r.get("code")),
                        str(r.get("signal_date")),
                        str(r.get("buy_date")),
                    )
                    for r in rows
                }
                for alert in strict_alerts:
                    signal_date = str(alert.get("latest_date") or "")
                    key = (
                        str(alert.get("code")),
                        signal_date,
                        str(alert.get("buy_date")),
                    )
                    if key in known:
                        continue
                    item = {
                        "code": str(alert.get("code")),
                        "name": alert.get("name") or code_to_name.get(str(alert.get("code")), ""),
                        "buy_date": alert.get("buy_date"),
                        "buy_price": alert.get("buy_price"),
                        "signal_date": signal_date,
                        "signal_close": alert.get("latest_close"),
                        "hold_days_at_signal": alert.get("hold_days"),
                        "reason": "strict_gakutto",
                        "production_action": "翌営業日始値で売却",
                        "topix_return_pct": topix_ret,
                        "decline_pct": decline_pct,
                        "declines": breadth.get("declines") if breadth else None,
                        "comparable": breadth.get("comparable") if breadth else None,
                        "condition_hit": True,
                        "normal_exit_date": None,
                        "normal_exit_open": None,
                        "normal_exit_trade_return_pct": None,
                        "shadow_delays": {},
                    }
                    rows.append(item)
                    known.add(key)
                    g = groups.get(str(alert.get("code")))
                    if g is not None and not g.empty:
                        _fill_future_observations(item, g)
                    changed = True
                    logger.info(
                        "がくっと全面安ログ追加: %s %s / TOPIX %+0.3f%% / 値下がり %.2f%%",
                        item["code"],
                        signal_date,
                        topix_ret,
                        decline_pct,
                    )
            else:
                logger.info(
                    "がくっと全面安ログ対象外: strict=%d / TOPIX=%s%% / 値下がり=%s%%",
                    len(strict_alerts),
                    topix_ret,
                    decline_pct,
                )

    rows.sort(key=lambda r: (str(r.get("signal_date", "")), str(r.get("code", ""))))
    payload["updated_through"] = (
        pd.to_datetime(price_df["Date"], errors="coerce").max().strftime("%Y-%m-%d")
    )
    if changed or not LOG_PATH.exists():
        _save_json(LOG_PATH, payload)
        logger.info("がくっと全面安・影ログ更新: %d件", len(rows))
