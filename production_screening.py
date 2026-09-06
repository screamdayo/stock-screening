"""日次の厳格ma5_breakout結果から、本番用A/B正規化混合候補を作る。

バックテストで固定したルールのみを使う。未来情報・市場環境フィルターは使わない。

A:
- MA25乖離 -10%以上 -5%未満
- 出来高前日比 < 1.0
- MA25乖離が浅い（-5%側）ほど上
- 同日A候補の下位20%を除外

B:
- MA25乖離 -10%以上 -5%未満
- RSI14 40以上50未満
- MA25乖離が深い（-10%側）ほど上
- 同日B候補の下位50%を除外

残ったA/Bの順位を各群0〜1へ正規化して混合し、同一銘柄は先に来た方だけ採用する。
"""

import math

import pandas as pd

A_CUT_PCT = 20
B_CUT_PCT = 50
RSI_PERIOD = 14


def _rsi_series(close, period=RSI_PERIOD):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(100).where(avg_gain.notna())


def _features_for_code(price_df, code):
    g = price_df[price_df["Code"] == code].dropna(subset=["C", "O"]).sort_values("Date").copy()
    if len(g) < 25:
        return None

    g["MA25"] = g["C"].rolling(25).mean()
    rsi = _rsi_series(g["C"])
    latest = g.iloc[-1]
    prev = g.iloc[-2] if len(g) >= 2 else None
    ma25 = g["MA25"].iloc[-1]
    if pd.isna(ma25) or ma25 == 0:
        return None

    volume_ratio = None
    if prev is not None and "Vo" in g.columns:
        pv = prev.get("Vo")
        tv = latest.get("Vo")
        if pd.notna(pv) and pd.notna(tv) and pv > 0:
            volume_ratio = float(tv / pv)

    rsi14 = rsi.iloc[-1]
    return {
        "ma25_dev_pct": float((latest["C"] / ma25 - 1) * 100),
        "volume_ratio": volume_ratio,
        "rsi14": None if pd.isna(rsi14) else float(rsi14),
    }


def _cut_ranked(items, kind, cut_pct):
    if kind == "A":
        ranked = sorted(items, key=lambda x: (-x["ma25_dev_pct"], str(x["code"])))
    else:
        ranked = sorted(items, key=lambda x: (x["ma25_dev_pct"], str(x["code"])))

    if not ranked:
        return []
    keep_n = max(1, math.ceil(len(ranked) * (1 - cut_pct / 100)))
    kept = ranked[:keep_n]
    denom = max(1, len(kept) - 1)
    out = []
    for i, item in enumerate(kept):
        x = dict(item)
        x["production_strategy"] = kind
        x["within_strategy_rank"] = i + 1
        x["within_strategy_count"] = len(kept)
        x["normalized_rank"] = i / denom if len(kept) > 1 else 0.0
        out.append(x)
    return out


def build_production_shortlist(price_df, strict_results):
    """厳格シグナルからA/B正規化混合の本命候補を順位付きで返す。"""
    enriched = []
    for r in strict_results:
        f = _features_for_code(price_df, r["code"])
        if not f:
            continue
        x = dict(r)
        x.update(f)
        enriched.append(x)

    a = [
        x for x in enriched
        if -10 <= x["ma25_dev_pct"] < -5
        and x["volume_ratio"] is not None
        and x["volume_ratio"] < 1.0
    ]
    b = [
        x for x in enriched
        if -10 <= x["ma25_dev_pct"] < -5
        and x["rsi14"] is not None
        and 40 <= x["rsi14"] < 50
    ]

    a_kept = _cut_ranked(a, "A", A_CUT_PCT)
    b_kept = _cut_ranked(b, "B", B_CUT_PCT)
    combined = sorted(
        a_kept + b_kept,
        key=lambda x: (
            x["normalized_rank"],
            0 if x["production_strategy"] == "A" else 1,
            str(x["code"]),
        ),
    )

    result = []
    seen = set()
    for x in combined:
        code = str(x["code"])
        if code in seen:
            continue
        seen.add(code)
        y = dict(x)
        y["production_rank"] = len(result) + 1
        y["screening_bucket"] = "primary"
        base_label = y.get("signal_label", "")
        y["base_signal_label"] = base_label
        y["signal_label"] = f"本命 #{y['production_rank']} {y['production_strategy']} / {base_label}"
        result.append(y)

    meta = {
        "strict_count": len(strict_results),
        "a_before_cut": len(a),
        "a_after_cut": len(a_kept),
        "b_before_cut": len(b),
        "b_after_cut": len(b_kept),
        "primary_count": len(result),
    }
    return result, meta
