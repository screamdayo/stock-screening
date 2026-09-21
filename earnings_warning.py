"""日次スクリーニング候補に直近決算予定の警告を付ける。"""

from datetime import date

import numpy as np
import pandas as pd

import download
from logger import get_logger

logger = get_logger(__name__)

EARNINGS_WARNING_BUSINESS_DAYS = 10


def _api_code(code):
    code = str(code).strip()
    return code if len(code) == 5 else code + "0"


def _fetch_schedule(code):
    rows = []
    pagination_key = None
    while True:
        params = {"code": _api_code(code)}
        if pagination_key:
            params["pagination_key"] = pagination_key
        res = download._request_with_retry(
            f"{download.BASE_URL}/fins/earnings-date",
            params=params,
        )
        res.raise_for_status()
        body = res.json()
        rows.extend(body.get("data") or [])
        pagination_key = body.get("pagination_key")
        if not pagination_key:
            break
    return rows


def _next_known_earnings(records, signal_date):
    sig = pd.Timestamp(signal_date).normalize()
    known = []
    for row in records:
        pub = row.get("PubDate")
        sch = row.get("SchDate")
        if not pub or not sch:
            continue
        try:
            pubd = pd.Timestamp(pub).normalize()
            schd = pd.Timestamp(sch).normalize()
        except Exception:
            continue
        if pubd > sig:
            continue
        known.append({
            "pub": pubd,
            "sch": schd,
            "fy": str(row.get("FYE") or ""),
            "fq": str(row.get("FQName") or ""),
        })

    if not known:
        return None

    df = pd.DataFrame(known).sort_values("pub")
    latest = df.groupby(["fy", "fq"], dropna=False, as_index=False).tail(1)
    future = latest[latest["sch"] > sig].sort_values("sch")
    if future.empty:
        return None

    sch = pd.Timestamp(future.iloc[0]["sch"]).normalize()
    # 「翌営業日から数えて何営業日先か」。J-Quantsの予定日が土日に
    # なることは通常ないため、平日ベースで日次警告用の距離を数える。
    days = int(np.busday_count(sig.date(), sch.date()))
    if 1 <= days <= EARNINGS_WARNING_BUSINESS_DAYS:
        return sch.date().isoformat(), days
    return None


def add_earnings_warnings(results, signal_date):
    """候補を除外せず、10営業日以内の決算予定だけメタ情報として付与する。"""
    if not results:
        return results

    warned = 0
    for item in results:
        item["earnings_within_10bd"] = False
        item["earnings_date"] = None
        item["earnings_business_days"] = None
        try:
            hit = _next_known_earnings(_fetch_schedule(item.get("code")), signal_date)
            if hit:
                item["earnings_within_10bd"] = True
                item["earnings_date"], item["earnings_business_days"] = hit
                warned += 1
        except Exception as exc:
            # 決算予定APIの一時障害で日次スクリーニング全体を止めない。
            logger.warning("決算予定の取得失敗 code=%s: %s", item.get("code"), exc)

    logger.info(
        "決算警告付与: %s/%s件（%s営業日以内）",
        warned,
        len(results),
        EARNINGS_WARNING_BUSINESS_DAYS,
    )
    return results
