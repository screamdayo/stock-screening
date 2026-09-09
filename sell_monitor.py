"""保有銘柄の機械的な売りシグナル監視。

ルール:
- 損切り -3% は証券会社側の逆指値/注文で管理するため、ここでは判定しない。
- 厳しめがくっと: 直前2区間のMA5が上向き/横ばい、当日MA5が下向き、当日陰線。
- 厳しめがくっとは引け後に確定し、翌営業日始値で売る運用。
- 最大保有は15営業日。15営業日到達時も翌営業日売却を通知する。
"""

import json
from pathlib import Path

import pandas as pd

HOLDINGS_PATH = Path("holdings.json")
MAX_HOLD_DAYS = 15


def load_holdings():
    if not HOLDINGS_PATH.exists():
        return []
    try:
        rows = json.loads(HOLDINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [r for r in rows if isinstance(r, dict) and r.get("code") and r.get("buy_date")]


def _strict_gakutto(g):
    if len(g) < 8:
        return False
    g = g.copy()
    g["C"] = pd.to_numeric(g["C"], errors="coerce")
    g["O"] = pd.to_numeric(g["O"], errors="coerce")
    g["MA5"] = g["C"].rolling(5).mean()
    j = len(g) - 1
    if pd.isna(g.MA5.iloc[j]) or pd.isna(g.MA5.iloc[j-1]) or pd.isna(g.MA5.iloc[j-2]) or pd.isna(g.MA5.iloc[j-3]):
        return False
    if not (g.MA5.iloc[j] < g.MA5.iloc[j-1]):
        return False
    if g.MA5.iloc[j-1] < g.MA5.iloc[j-2]:
        return False
    if g.MA5.iloc[j-2] < g.MA5.iloc[j-3]:
        return False
    return float(g.C.iloc[j]) < float(g.O.iloc[j])


def check_sell_signals(price_df, code_to_name=None):
    code_to_name = code_to_name or {}
    holdings = load_holdings()
    if not holdings or price_df is None or price_df.empty:
        return []

    df = price_df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    alerts = []

    for h in holdings:
        code = str(h["code"])
        buy_date = pd.to_datetime(h["buy_date"], errors="coerce")
        if pd.isna(buy_date):
            continue

        g = df[df["Code"].astype(str) == code].sort_values("Date").reset_index(drop=True)
        if g.empty:
            continue

        latest_date = g["Date"].iloc[-1]
        held_dates = g.loc[(g["Date"] >= buy_date) & (g["Date"] <= latest_date), "Date"].dropna().drop_duplicates()
        hold_days = int(len(held_dates))
        name = h.get("name") or code_to_name.get(code) or code
        buy_price = h.get("buy_price")

        reason = None
        if _strict_gakutto(g):
            reason = "strict_gakutto"
        elif hold_days >= MAX_HOLD_DAYS:
            reason = "max_hold"

        if reason:
            latest_close = pd.to_numeric(g["C"].iloc[-1], errors="coerce")
            alerts.append({
                "code": code,
                "name": name,
                "buy_date": str(h["buy_date"]),
                "buy_price": buy_price,
                "latest_date": latest_date.strftime("%Y-%m-%d") if pd.notna(latest_date) else "",
                "latest_close": None if pd.isna(latest_close) else float(latest_close),
                "hold_days": hold_days,
                "reason": reason,
                "action": "翌営業日始値で売却",
            })

    return alerts
