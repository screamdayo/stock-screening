"""月次モメンタム6Mの結果をDiscordへ通知する（結果のみ・発注なし）。

レポート本体は非公開データリポジトリ（stock-screening-data）の月次ワークフローが
results/momentum6m_monthly_latest.json に保存する。ここではそれを読んで通知するだけ。

    python scripts/notify_momentum6m.py                 # 前月分のレポートなら通知
    python scripts/notify_momentum6m.py --allow-stale   # 対象月が違っても通知
    python scripts/notify_momentum6m.py --dry-run       # 送信せずに表示
"""
import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import notifier  # noqa: E402

DEFAULT_REPORT = ROOT / ".private-data/results/momentum6m_monthly_latest.json"


def previous_month_jst():
    first = datetime.now(ZoneInfo("Asia/Tokyo")).replace(day=1)
    return (first - timedelta(days=1)).strftime("%Y-%m")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--allow-stale", action="store_true", help="対象月が前月でなくても通知する")
    parser.add_argument("--dry-run", action="store_true", help="送信せずメッセージを表示する")
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    if args.dry_run:
        notifier._post_long = print
        notifier._post = print

    expected = previous_month_jst()
    if report.get("signal_month") != expected and not args.allow_stale:
        notifier._post(
            f"⚠️ **月次モメンタム6M**：{expected} 分のレポートがまだありません"
            f"（最新は {report.get('signal_month')} 分）。データ側の月次実行を確認してください。"
        )
        sys.exit(1)

    notifier.notify_momentum6m(report)
    # 公開リポジトリのログは誰でも読めるので、銘柄はログに出さない
    print(f"notified momentum6m report for {report.get('signal_month')}", flush=True)


if __name__ == "__main__":
    main()
