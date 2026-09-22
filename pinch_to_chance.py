"""
ピンチをチャンスにセンサー（日次運用）。

10年弱の保存データで固定した暴落後の「底打ち確認型」シグナル。
市場条件と個別候補条件を分離し、当日引け後にだけ判定する。

市場センサー:
- 粗い急落反転銘柄率 >= 1.0%
  （5日騰落<=-8%, DD20<=-10%, 当日陽線>=+1%）
- TOPIX当日騰落 >= +3.0%
- TOPIX当日安値 > 前日安値

個別候補:
- 5日騰落 <= -12%
- DD20 <= -20%
- 当日陽線 >= +2%
- 出来高 / 20日平均 >= 1.5倍

本番出口は41営業日目の寄り。ここでは売買執行せず、通知・表示だけ行う。
"""
from datetime import timedelta

import pandas as pd

import download
from logger import get_logger

logger = get_logger(__name__)

SENSOR_NAME = "pinch_to_chance"
SIGNAL_LABEL = "ピンチをチャンスに"

REVERSAL_RATE_MIN_PCT = 1.0
TOPIX_RETURN_MIN_PCT = 3.0

COARSE_RET5_MAX = -8.0
COARSE_DD20_MAX = -10.0
COARSE_CANDLE_MIN = 1.0

CANDIDATE_RET5_MAX = -12.0
CANDIDATE_DD20_MAX = -20.0
CANDIDATE_CANDLE_MIN = 2.0
CANDIDATE_VOL_RATIO20_MIN = 1.5

EXIT_GUIDE_BUSINESS_DAYS = 41
MAIN_PICK_COUNT = 3
REFERENCE_PICK_COUNT = 2


def _prepare(group):
    g = group.dropna(subset=["O", "H", "L", "C"]).sort_values("Date").reset_index(drop=True).copy()
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


def _fetch_trade_plan(signal_date, hold_days=EXIT_GUIDE_BUSINESS_DAYS):
    """J-Quants取引カレンダーから翌営業日寄りと固定出口日を求める。

    バックテスト定義に合わせ、entry=シグナル翌営業日の寄り、
    exit=entryからhold_days営業日後の寄り（entry日は0日目）とする。
    カレンダー取得に失敗してもセンサー判定自体は止めない。
    """
    start = (pd.Timestamp(signal_date) + timedelta(days=1)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(signal_date) + timedelta(days=120)).strftime("%Y-%m-%d")
    try:
        res = download._request_with_retry(
            f"{download.BASE_URL}/markets/calendar",
            params={"from": start, "to": end},
        )
        res.raise_for_status()
        rows = res.json().get("data", [])
        d = pd.DataFrame(rows)
        if d.empty or "Date" not in d.columns:
            raise RuntimeError("取引カレンダーが空です")
        hol_col = "HolDiv" if "HolDiv" in d.columns else ("HolidayDivision" if "HolidayDivision" in d.columns else None)
        if hol_col:
            d = d[d[hol_col].astype(str).isin(["1", "2"])].copy()
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        days = d["Date"].dropna().sort_values().drop_duplicates().tolist()
        if len(days) <= hold_days:
            raise RuntimeError(f"取引カレンダー営業日不足: {len(days)}日")
        return {
            "planned_entry_date": pd.Timestamp(days[0]).strftime("%Y-%m-%d"),
            "planned_exit_date": pd.Timestamp(days[hold_days]).strftime("%Y-%m-%d"),
        }
    except Exception as e:
        logger.warning("ピンチセンサー売買予定日の算出に失敗: %s", e)
        return {"planned_entry_date": None, "planned_exit_date": None}


def _fetch_topix(latest_date):
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
    d["Date"] = pd.to_datetime(d["Date"])
    for c in ["O", "H", "L", "C"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["L", "C"]).sort_values("Date").reset_index(drop=True)
    d["RET1"] = d["C"].pct_change() * 100
    return d


def evaluate(price_df, target_codes):
    df = price_df[price_df["Code"].isin(target_codes)].copy()
    if df.empty:
        return {"active": False, "reason": "株価データなし", "candidates": []}

    df["Date"] = pd.to_datetime(df["Date"])
    latest_date = pd.Timestamp(df["Date"].max()).normalize()

    latest_rows = df[df["Date"].dt.normalize() == latest_date].dropna(subset=["O", "H", "L", "C"])
    universe_n = int(latest_rows["Code"].nunique())

    coarse_count = 0
    candidates = []
    for code, group in df.groupby("Code"):
        g = _prepare(group)
        if g.empty or pd.Timestamp(g["Date"].iloc[-1]).normalize() != latest_date:
            continue
        r = g.iloc[-1]
        required = ["RET5", "DD20", "CANDLE"]
        if any(pd.isna(r[c]) for c in required):
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

    # 実運用の選別はDD20が深い順。過去検証では上位3が資金効率の山だった。
    candidates.sort(key=lambda x: (x["dd20_pct"], x["code"]))
    for i, item in enumerate(candidates, start=1):
        item["dd20_rank"] = i
        if i <= MAIN_PICK_COUNT:
            item["selection_tier"] = "main"
        elif i <= MAIN_PICK_COUNT + REFERENCE_PICK_COUNT:
            item["selection_tier"] = "reference"
        else:
            item["selection_tier"] = "other"

    reversal_rate = (coarse_count / universe_n * 100) if universe_n else 0.0

    topix = _fetch_topix(latest_date)
    latest_ix = topix.iloc[-1]
    if pd.Timestamp(latest_ix["Date"]).normalize() != latest_date:
        return {
            "active": False,
            "reason": f"TOPIX最新日が株価最新日と不一致 ({latest_ix['Date'].date()} != {latest_date.date()})",
            "signal_date": latest_date,
            "universe_n": universe_n,
            "coarse_reversal_count": coarse_count,
            "reversal_rate_pct": round(reversal_rate, 3),
            "candidates": [],
        }
    if len(topix) < 2:
        raise RuntimeError("TOPIX前日データが不足しています")

    prev_ix = topix.iloc[-2]
    topix_ret1 = float(latest_ix["RET1"])
    low_up = bool(float(latest_ix["L"]) > float(prev_ix["L"]))
    rate_ok = reversal_rate >= REVERSAL_RATE_MIN_PCT
    return_ok = topix_ret1 >= TOPIX_RETURN_MIN_PCT
    active = bool(rate_ok and return_ok and low_up)

    trade_plan = _fetch_trade_plan(latest_date) if active else {
        "planned_entry_date": None,
        "planned_exit_date": None,
    }

    result = {
        "active": active,
        "signal_date": latest_date,
        "universe_n": universe_n,
        "coarse_reversal_count": coarse_count,
        "reversal_rate_pct": round(reversal_rate, 3),
        "reversal_rate_threshold_pct": REVERSAL_RATE_MIN_PCT,
        "topix_return_pct": round(topix_ret1, 3),
        "topix_return_threshold_pct": TOPIX_RETURN_MIN_PCT,
        "topix_low": round(float(latest_ix["L"]), 3),
        "topix_prev_low": round(float(prev_ix["L"]), 3),
        "topix_low_up": low_up,
        "market_conditions": {
            "reversal_rate_ok": rate_ok,
            "topix_return_ok": return_ok,
            "topix_low_up_ok": low_up,
        },
        "entry_rule": "翌営業日寄り",
        "exit_rule": "41営業日目の寄り",
        "exit_guide_business_days": EXIT_GUIDE_BUSINESS_DAYS,
        "planned_entry_date": trade_plan["planned_entry_date"],
        "planned_exit_date": trade_plan["planned_exit_date"],
        "ranking_rule": "DD20が深い順",
        "main_pick_count": MAIN_PICK_COUNT,
        "reference_pick_count": REFERENCE_PICK_COUNT,
        "candidate_count": len(candidates) if active else 0,
        "main_candidates": [x for x in candidates if x["selection_tier"] == "main"] if active else [],
        "reference_candidates": [x for x in candidates if x["selection_tier"] == "reference"] if active else [],
        "candidates": candidates if active else [],
    }
    if not active:
        result["reason"] = "市場センサー条件未成立"

    logger.info(
        "ピンチをチャンスにセンサー: %s / 反転率 %.3f%% (%d/%d) / TOPIX %+0.3f%% / 安値切上げ=%s / 候補=%d",
        "発動" if active else "待機",
        reversal_rate,
        coarse_count,
        universe_n,
        topix_ret1,
        low_up,
        len(candidates) if active else 0,
    )
    return result
