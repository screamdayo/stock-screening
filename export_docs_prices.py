"""
export_docs_prices.py
GitHub Pages用に対象市場全銘柄の直近日足を docs/prices/ に出力し、
同じ株価データを使って最新のスクリーニング結果を docs/screening.json に保存する。
さらに、保有管理画面の銘柄名自動補完用に docs/stock_names.json を出力する。
"""

import os
import json
import time
from datetime import datetime

import config
import download
from strategies import registry
from logger import get_logger

logger = get_logger(__name__)


def _json_safe(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return round(value, 4)
    return value


def export_docs(price_df, target_codes, code_to_name, screening_results, gc_results=None, pinch_sensor=None, selling_sensor=None):
    start = time.time()

    df = price_df[price_df["Code"].isin(target_codes)]
    if df.empty:
        logger.warning("株価データが空でした。docs/へのエクスポートをスキップします。")
        return

    os.makedirs(config.DOCS_PRICES_DIR, exist_ok=True)

    exported = 0
    for code, group in df.groupby("Code"):
        group = group.sort_values("Date")
        bars = []
        for _, row in group.iterrows():
            bar = {
                "t": row["Date"].strftime("%Y-%m-%d"),
                "o": round(float(row["O"]), 2),
                "h": round(float(row["H"]), 2),
                "l": round(float(row["L"]), 2),
                "c": round(float(row["C"]), 2),
            }
            if "Vo" in group.columns:
                try:
                    if row["Vo"] == row["Vo"]:
                        bar["v"] = int(row["Vo"])
                except Exception:
                    pass
            bars.append(bar)

        path = os.path.join(config.DOCS_PRICES_DIR, f"{code}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(bars, f, separators=(",", ":"))
        exported += 1

    meta_path = os.path.join(config.DOCS_PRICES_DIR, "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "codes": exported,
            "business_days": config.CHART_BUSINESS_DAYS,
        }, f, ensure_ascii=False, indent=2)

    gc_results = gc_results or []

    screening_items = []
    for r in screening_results:
        item = {k: _json_safe(v) for k, v in r.items()}
        if "name" not in item or not item["name"]:
            item["name"] = code_to_name.get(r["code"], "")
        screening_items.append(item)

    screening_path = os.path.join("docs", "screening.json")
    with open(screening_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "strategy": config.ACTIVE_STRATEGY,
            "count": len(screening_items),
            "items": screening_items,
        }, f, ensure_ascii=False, indent=2)

    gc_items = []
    for r in gc_results:
        item = {k: _json_safe(v) for k, v in r.items()}
        if "name" not in item or not item["name"]:
            item["name"] = code_to_name.get(r["code"], "")
        gc_items.append(item)

    gc_path = os.path.join("docs", "gc_screening.json")
    with open(gc_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "strategy": "gc_strong_breakout",
            "count": len(gc_items),
            "items": gc_items,
        }, f, ensure_ascii=False, indent=2)

    pinch_path = os.path.join("docs", "pinch_sensor.json")
    pinch_payload = _json_safe(pinch_sensor or {"active": False, "reason": "未判定", "candidates": []})
    pinch_payload["generated_at"] = datetime.now().isoformat(timespec="seconds")
    with open(pinch_path, "w", encoding="utf-8") as f:
        json.dump(pinch_payload, f, ensure_ascii=False, indent=2)

    selling_path = os.path.join("docs", "selling_climax_sensor.json")
    selling_payload = _json_safe(selling_sensor or {"active": False, "reason": "未判定", "candidates": []})
    selling_payload["generated_at"] = datetime.now().isoformat(timespec="seconds")
    with open(selling_path, "w", encoding="utf-8") as f:
        json.dump(selling_payload, f, ensure_ascii=False, indent=2)

    # Pages上で銘柄コードを入力した瞬間に会社名を補完するための軽量マスター。
    names_path = os.path.join("docs", "stock_names.json")
    names = {
        str(code): str(code_to_name.get(str(code), ""))
        for code in sorted(target_codes)
        if code_to_name.get(str(code))
    }
    with open(names_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "items": names,
        }, f, ensure_ascii=False, separators=(",", ":"))

    logger.info(
        f"docs/エクスポート完了: 株価{exported}銘柄 / "
        f"スクリーニング{len(screening_items)}件 / GC強ブレイク{len(gc_items)}件 / "
        f"ピンチセンサー{'発動' if (pinch_sensor or {}).get('active') else '待機'} / "
        f"売り尽くしセンサー{'発動' if (selling_sensor or {}).get('active') else '待機'} / 銘柄名{len(names)}件 "
        f"（{time.time() - start:.1f}秒）"
    )


def run():
    logger.info("=== GitHub Pages用データエクスポート開始（単独実行） ===")

    target_codes, code_to_name = download.get_target_codes_and_names()

    logger.info(
        f"株価データ取得中（直近{config.CHART_BUSINESS_DAYS}営業日、"
        f"対象{len(target_codes)}銘柄）..."
    )
    price_df = download.get_price_history(
        target_business_days=config.CHART_BUSINESS_DAYS,
        max_lookback_days=config.CHART_MAX_LOOKBACK_DAYS,
    )

    if price_df.empty:
        logger.warning("株価データが空でした。エクスポートをスキップします。")
        return

    screener_fn = registry.get_latest_screener(config.ACTIVE_STRATEGY)
    gc_screener_fn = registry.get_latest_screener("gc_strong_breakout")
    results = screener_fn(price_df, target_codes)
    gc_results = gc_screener_fn(price_df, target_codes)

    # 単独エクスポート時もセンサー状態を生成する。
    import pinch_to_chance
    import selling_climax
    pinch_sensor = pinch_to_chance.evaluate(price_df, target_codes)
    selling_sensor = selling_climax.evaluate(price_df, target_codes)
    for r in pinch_sensor.get("candidates", []):
        r["name"] = code_to_name.get(r["code"], "")
    for r in selling_sensor.get("candidates", []):
        r["name"] = code_to_name.get(r["code"], "")

    export_docs(price_df, target_codes, code_to_name, results, gc_results, pinch_sensor, selling_sensor)


if __name__ == "__main__":
    run()
