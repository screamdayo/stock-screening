"""TOPIXくいっと発生日の本番運用ポリシー。

10年検証で、TOPIX自身がくいっとした日は個別くいっとが大量発生する一方、
非本命（ATR14<3.0 または DD20>-5.5）は成績が弱く、
本命と204系は成績が保たれたため、候補を削除せず表示・行動フラグだけ付与する。

- TOPIXくいっと + 非本命: 警戒 / 原則見送り
- TOPIXくいっと + 本命: 通常候補
- TOPIXくいっと + 204: 最優先
"""

from datetime import timedelta

import pandas as pd

import download
from logger import get_logger

logger = get_logger(__name__)


def _fetch_topix_daily(latest_date):
    latest = pd.Timestamp(latest_date).normalize()
    start = (latest - timedelta(days=60)).strftime("%Y-%m-%d")
    end = latest.strftime("%Y-%m-%d")
    res = download._request_with_retry(
        f"{download.BASE_URL}/indices/bars/daily/topix",
        params={"from": start, "to": end},
    )
    res.raise_for_status()
    d = pd.DataFrame(res.json().get("data", []))
    if d.empty:
        raise RuntimeError("TOPIX日足が取得できませんでした")

    for c in ["O", "C"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["Date"] = pd.to_datetime(d["Date"]).dt.normalize()
    d = (
        d.dropna(subset=["Date", "O", "C"])
        .sort_values("Date")
        .drop_duplicates("Date")
        .reset_index(drop=True)
    )
    d["MA5"] = d["C"].rolling(5).mean()
    return d


def _latest_kuitto_state(latest_date):
    latest = pd.Timestamp(latest_date).normalize()
    d = _fetch_topix_daily(latest)

    same = d.index[d["Date"] == latest].tolist()
    if not same:
        return {
            "available": False,
            "active": False,
            "requested_date": latest.strftime("%Y-%m-%d"),
            "latest_topix_date": d["Date"].iloc[-1].strftime("%Y-%m-%d"),
            "reason": "TOPIX同日データ未取得",
        }

    i = same[-1]
    if i < 3 or pd.isna(d["MA5"].iloc[i]):
        return {
            "available": False,
            "active": False,
            "requested_date": latest.strftime("%Y-%m-%d"),
            "latest_topix_date": d["Date"].iloc[i].strftime("%Y-%m-%d"),
            "reason": "TOPIX MA5判定データ不足",
        }

    need = [i - 3, i - 2, i - 1, i]
    if any(pd.isna(d["MA5"].iloc[k]) for k in need):
        active = False
    else:
        active = bool(
            d["MA5"].iloc[i - 1] <= d["MA5"].iloc[i - 2]
            and d["MA5"].iloc[i - 2] <= d["MA5"].iloc[i - 3]
            and d["MA5"].iloc[i] > d["MA5"].iloc[i - 1]
            and d["C"].iloc[i] > d["O"].iloc[i]
        )

    return {
        "available": True,
        "active": active,
        "date": d["Date"].iloc[i].strftime("%Y-%m-%d"),
        "open": round(float(d["O"].iloc[i]), 3),
        "close": round(float(d["C"].iloc[i]), 3),
        "ma5": round(float(d["MA5"].iloc[i]), 3),
        "ma5_prev": round(float(d["MA5"].iloc[i - 1]), 3),
        "reason": "TOPIXくいっと発動" if active else "TOPIXくいっと非発動",
    }


def apply(results, price_df, kuitto_204_results=None):
    """本番くいっと候補へTOPIXくいっと運用フラグを付与する。"""
    if not results:
        return {"available": False, "active": False, "reason": "個別候補なし"}

    latest = pd.Timestamp(price_df["Date"].max()).normalize()
    try:
        state = _latest_kuitto_state(latest)
    except Exception as e:
        logger.warning("TOPIXくいっと運用判定をスキップ: %s", e)
        state = {
            "available": False,
            "active": False,
            "requested_date": latest.strftime("%Y-%m-%d"),
            "reason": str(e),
        }

    codes204 = {
        str(r.get("code"))
        for r in (kuitto_204_results or [])
        if r.get("code") is not None
    }

    for r in results:
        code = str(r.get("code"))
        is_204 = code in codes204
        winner = bool(r.get("winner_filter"))

        r["topix_kuitto_available"] = bool(state.get("available"))
        r["topix_kuitto_active"] = bool(state.get("active"))
        r["topix_kuitto_date"] = state.get("date") or state.get("requested_date")
        r["topix_kuitto_204"] = bool(is_204)
        r["topix_kuitto_priority"] = bool(state.get("active") and is_204)
        r["topix_kuitto_caution"] = bool(state.get("active") and not winner)

        if r["topix_kuitto_priority"]:
            r["topix_kuitto_policy"] = "priority_204"
            r["topix_kuitto_action"] = "最優先"
        elif r["topix_kuitto_caution"]:
            r["topix_kuitto_policy"] = "caution_skip"
            r["topix_kuitto_action"] = "警戒・原則見送り"
        elif state.get("active") and winner:
            r["topix_kuitto_policy"] = "winner_keep"
            r["topix_kuitto_action"] = "本命・通常候補"
        else:
            r["topix_kuitto_policy"] = "normal"
            r["topix_kuitto_action"] = "通常"

    logger.info(
        "TOPIXくいっと運用: %s / 最優先204=%d / 本命継続=%d / 警戒見送り=%d",
        "発動" if state.get("active") else ("非発動" if state.get("available") else "判定不可"),
        sum(bool(r.get("topix_kuitto_priority")) for r in results),
        sum(bool(r.get("topix_kuitto_active") and r.get("winner_filter") and not r.get("topix_kuitto_priority")) for r in results),
        sum(bool(r.get("topix_kuitto_caution")) for r in results),
    )
    return state
