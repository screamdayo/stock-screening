"""
export_docs_prices.py
GitHub Pages（docs/index.html）で使う「保有ポジションの利確チャート」用に、
対象市場全銘柄の直近株価データ（ローソク足用）を docs/prices/ 以下にJSONで出力する。

重要: ここで出力するのは銘柄コード→株価という「誰が見ても問題ない公開情報」のみ。
実際の保有ポジション（どの銘柄をいくらで何株持っているか）はSBI証券の約定履歴CSVから
ブラウザ側（docs/index.html）でその場で計算する設計になっており、
このリポジトリには一切含まれない・送信されない。

出力先: docs/prices/<銘柄コード>.json
形式: [{"t": "2026-06-01", "o": 2500.0, "h": 2520.0, "l": 2490.0, "c": 2510.0}, ...]
（キー名はファイルサイズ削減のため短縮している）

GitHub Actionsから平日の夜間などに自動実行される想定
（.github/workflows/export_prices.yml）。

使い方（ローカルで試す場合）:
    python export_docs_prices.py
"""

import os
import json
import time
from datetime import datetime

import config
import download
from logger import get_logger

logger = get_logger(__name__)


def run():
    logger.info("=== docs/prices/ 株価データエクスポート開始 ===")
    start = time.time()

    target_codes = download.get_target_codes()

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

    # 対象市場の銘柄のみに絞る（他市場の銘柄が混ざらないようにする）
    price_df = price_df[price_df["Code"].isin(target_codes)]

    os.makedirs(config.DOCS_PRICES_DIR, exist_ok=True)

    exported = 0
    for code, group in price_df.groupby("Code"):
        group = group.sort_values("Date")
        bars = [
            {
                "t": row["Date"].strftime("%Y-%m-%d"),
                "o": round(float(row["O"]), 2),
                "h": round(float(row["H"]), 2),
                "l": round(float(row["L"]), 2),
                "c": round(float(row["C"]), 2),
            }
            for _, row in group.iterrows()
        ]

        path = os.path.join(config.DOCS_PRICES_DIR, f"{code}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(bars, f, separators=(",", ":"))
        exported += 1

    # 生成日時・対象銘柄数のメタ情報。
    # docs/index.html側で「データがどれだけ新しいか」を表示するのに使う。
    meta_path = os.path.join(config.DOCS_PRICES_DIR, "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "codes": exported,
            "business_days": config.CHART_BUSINESS_DAYS,
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"エクスポート完了: {exported}銘柄 → {config.DOCS_PRICES_DIR} "
                f"（{time.time() - start:.1f}秒）")


if __name__ == "__main__":
    run()
