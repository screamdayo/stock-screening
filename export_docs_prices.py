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
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return round(value, 4)
    return value


def export_docs(price_df, target_codes, code_to_name, screening_results):
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
        f"スクリーニング{len(screening_items)}件 / 銘柄名{len(names)}件 "
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
    results = screener_fn(price_df, target_codes)

    export_docs(price_df, target_codes, code_to_name, results)


if __name__ == "__main__":
    run()
