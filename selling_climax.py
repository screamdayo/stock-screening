"""
selling_climax.py
暴落当日の「売り尽くし（セリングクライマックス）」先行センサー。

凍結候補ルール:
- TOPIX 当日騰落率 <= -2%
- TOPIX DD20 <= -15%
- 強反転銘柄比率 >= 10%
  強反転銘柄 = ret5<=-12%, DD20<=-20%, 陽線>=+2%, 出来高20日比>=1.5
  比率の分母 = 粗い反転銘柄数
- 個別候補は強反転銘柄を DD20 が深い順
- 本命 TOP3
- 資金配分は1銘柄最大80%、100株単位で #1→#2→#3、余りは現金
- 翌営業日寄りで入り、41営業日目寄りを出口ガイド
"""

from datetime import timedelta
import pandas as pd

import download
import pinch_to_chance
from logger import get_logger

logger = get_logger(__name__)

SENSOR_NAME = "selling_climax"
SIGNAL_LABEL = "売り尽くしセンサー"

TOPIX_RET1_MAX_PCT = -2.0
TOPIX_DD20_MAX_PCT = -15.0
ANCHOR_SHARE_MIN_PCT = 10.0

COARSE_RET5_MAX = -8.0
COARSE_DD20_MAX = -10.0
COARSE_CANDLE_MIN = 1.0

CANDIDATE_RET5_MAX = -12.0
CANDIDATE_DD20_MAX = -20.0
CANDIDATE_CANDLE_MIN = 2.0
CANDIDATE_VOL_RATIO20_MIN = 1.5

MAIN_PICK_COUNT = 3
REFERENCE_PICK_COUNT = 2
SINGLE_STOCK_CAP_PCT = 80
LOT_SIZE = 100
EXIT_GUIDE_BUSINESS_DAYS = 41


def _prepare(group):
    g = group.sort_values("Date").copy()
    for c in ["O", "H", "L", "C", "Vo"]:
        if c in g.columns:
            g[c] = pd.to_numeric(g[c], errors="coerce")
        else:
            g[c] = pd.NA
    g["RET5"] = (g["C"] / g["C"].shift(5) - 1) * 100
    g["HIGH20"] = g["C"].rolling(20).max()
    g["DD20"] = (g["C"] / g["HIGH20"] - 1) * 100
    g["CANDLE"] = (g["C"] / g["O"] - 1) * 100
    g["VOL20"] = g["Vo"].rolling(20).mean()
    g["VOLR20"] = g["Vo"] / g["VOL20"]
    return g


def _fetch_topix(latest_date):
    start = (pd.Timestamp(latest_date) - timedelta(days=70)).strftime("%Y-%m-%d")
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
    d["Date"] = pd.to_datetime(d["Date"])
    for c in ["O", "H", "L", "C"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["C"]).sort_values("Date").reset_index(drop=True)
    d["RET1"] = d["C"].pct_change() * 100
    d["HIGH20"] = d["C"].rolling(20).max()
    d["DD20"] = (d["C"] / d["HIGH20"] - 1) * 100
    return d


def evaluate(price_df, target_codes):
    df = price_df[price_df["Code"].isin(target_codes)].copy()
    if df.empty:
        return {"active": False, "reason": "株価データなし", "candidates": []}

    df["Date"] = pd.to_datetime(df["Date"])
    latest_date = pd.Timestamp(df["Date"].max()).normalize()

    coarse_count = 0
    candidates = []
    for code, group in df.groupby("Code"):
        g = _prepare(group)
        if g.empty or pd.Timestamp(g["Date"].iloc[-1]).normalize() != latest_date:
            continue
        r = g.iloc[-1]
        if any(pd.isna(r[c]) for c in ["RET5", "DD20", "CANDLE"]):
            continue

        ret5 = float(r["RET5"])
        dd20 = float(r["DD20"])
        candle = float(r["CANDLE"])
        volr = float(r["VOLR20"]) if pd.notna(r["VOLR20"]) else None

        if ret5 <= COARSE_RET5_MAX and dd20 <= COARSE_DD20_MAX and candle >= COARSE_CANDLE_MIN:
            coarse_count += 1

        if (
            ret5 <= CANDIDATE_RET5_MAX
            and dd20 <= CANDIDATE_DD20_MAX
            and candle >= CANDIDATE_CANDLE_MIN
            and volr is not None
            and volr >= CANDIDATE_VOL_RATIO20_MIN
        ):
            candidates.append({
                "code": str(code),
                "close": round(float(r["C"]), 1),
                "open": round(float(r["O"]), 1),
                "ret5_pct": round(ret5, 3),
                "dd20_pct": round(dd20, 3),
                "bull_candle_pct": round(candle, 3),
                "volume_ratio20": round(volr, 3),
                "signal_type": SENSOR_NAME,
                "signal_label": SIGNAL_LABEL,
            })

    candidates.sort(key=lambda x: (x["dd20_pct"], x["code"]))
    for i, item in enumerate(candidates, 1):
        item["dd20_rank"] = i
        item["selection_tier"] = "main" if i <= 3 else ("reference" if i <= 5 else "other")

    anchor_n = len(candidates)
    anchor_share = (anchor_n / coarse_count * 100) if coarse_count else 0.0

    topix = _fetch_topix(latest_date)
    latest_ix = topix.iloc[-1]
    if pd.Timestamp(latest_ix["Date"]).normalize() != latest_date:
        return {
            "active": False,
            "reason": f"TOPIX最新日が株価最新日と不一致 ({latest_ix['Date'].date()} != {latest_date.date()})",
            "signal_date": latest_date,
            "coarse_reversal_count": coarse_count,
            "anchor_count": anchor_n,
            "anchor_share_pct": round(anchor_share, 3),
            "candidates": [],
        }

    ret1 = float(latest_ix["RET1"])
    dd20_ix = float(latest_ix["DD20"])
    ret_ok = ret1 <= TOPIX_RET1_MAX_PCT
    dd_ok = dd20_ix <= TOPIX_DD20_MAX_PCT
    share_ok = anchor_share >= ANCHOR_SHARE_MIN_PCT
    active = bool(ret_ok and dd_ok and share_ok)

    trade_plan = pinch_to_chance._fetch_trade_plan(latest_date, EXIT_GUIDE_BUSINESS_DAYS) if active else {
        "planned_entry_date": None,
        "planned_exit_date": None,
    }

    result = {
        "active": active,
        "signal_date": latest_date,
        "topix_return_pct": round(ret1, 3),
        "topix_return_threshold_pct": TOPIX_RET1_MAX_PCT,
        "topix_dd20_pct": round(dd20_ix, 3),
        "topix_dd20_threshold_pct": TOPIX_DD20_MAX_PCT,
        "coarse_reversal_count": coarse_count,
        "anchor_count": anchor_n,
        "anchor_share_pct": round(anchor_share, 3),
        "anchor_share_threshold_pct": ANCHOR_SHARE_MIN_PCT,
        "market_conditions": {
            "topix_return_ok": ret_ok,
            "topix_dd20_ok": dd_ok,
            "anchor_share_ok": share_ok,
        },
        "entry_rule": "翌営業日寄り",
        "exit_rule": "41営業日目の寄り",
        "exit_guide_business_days": EXIT_GUIDE_BUSINESS_DAYS,
        "planned_entry_date": trade_plan["planned_entry_date"],
        "planned_exit_date": trade_plan["planned_exit_date"],
        "ranking_rule": "DD20が深い順",
        "allocation_rule": "DD20順位順に1銘柄最大80%、100株単位、TOP3まで。余りは現金",
        "single_stock_cap_pct": SINGLE_STOCK_CAP_PCT,
        "lot_size": LOT_SIZE,
        "allocation_candidate_pool": "TOP3",
        "main_pick_count": MAIN_PICK_COUNT,
        "reference_pick_count": REFERENCE_PICK_COUNT,
        "candidate_count": len(candidates) if active else 0,
        "main_candidates": [x for x in candidates if x["selection_tier"] == "main"] if active else [],
        "reference_candidates": [x for x in candidates if x["selection_tier"] == "reference"] if active else [],
        "candidates": candidates if active else [],
        "relationship_note": "先行センサー。発動後はピンチをチャンスにセンサーの底打ち確認を待つ。",
    }
    if not active:
        result["reason"] = "売り尽くし条件未成立"

    logger.info(
        "売り尽くしセンサー: %s / TOPIX %+0.3f%% / DD20 %+0.3f%% / 強反転比率 %.3f%% (%d/%d) / 候補=%d",
        "発動" if active else "待機", ret1, dd20_ix, anchor_share, anchor_n, coarse_count,
        len(candidates) if active else 0,
    )
    return result
