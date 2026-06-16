import os, requests, yfinance as yf, time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

DISCORD_WEBHOOK  = os.environ["DISCORD_WEBHOOK_URL"]
JQUANTS_API_KEY  = os.environ["JQUANTS_API_KEY"]

# ===== プライム銘柄リスト取得 =====
def get_prime_codes():
    res = requests.get(
        "https://api.jquants.com/v2/equities/master",
        headers={"x-api-key": JQUANTS_API_KEY}
    )
    data = res.json()
    df = pd.DataFrame(data["data"])
    prime = df[df["MktNm"] == "プライム"]["Code"].tolist()
    print(f"プライム銘柄数: {len(prime)}件")
    return prime

# ===== スクリーニング（1件ずつ順番に処理）=====
def screen(codes):
    results = []
    for i, code in enumerate(codes):
        if i % 100 == 0:
            print(f"  {i}/{len(codes)}件処理中...")
        try:
            time.sleep(0.5)
            ticker = yf.Ticker(f"{code}.T")
            df = ticker.history(period="20d")
            if df.empty or len(df) < 6:
                continue

            latest = df.iloc[-1]
            open_  = latest["Open"]
            close  = latest["Close"]
            if close <= open_:
                continue

            ma5_today = df["Close"].iloc[-5:].mean()
            ma5_prev  = df["Close"].iloc[-6:-1].mean()

            if ma5_today > ma5_prev:
                results.append({
                    "code":      code,
                    "close":     round(close, 1),
                    "open":      round(open_, 1),
                    "ma5_today": round(ma5_today, 1),
                    "ma5_prev":  round(ma5_prev, 1),
                })
        except Exception as e:
            print(f"{code} error: {e}")
    return results

# ===== Discord通知 =====
def notify(results):
    today = datetime.now().strftime("%Y/%m/%d")

    if not results:
        msg = f"📊 **株スクリーニング結果 {today}**\n該当銘柄なし"
        requests.post(DISCORD_WEBHOOK, json={"content": msg})
        return

    lines = [
        f"・**{r['code']}**｜終値 {r['close']}円｜5MA {r['ma5_prev']}→{r['ma5_today']}↑"
        for r in results[:30]
    ]
    header = (f"📊 **株スクリーニング結果 {today}**\n"
              f"✅ 5日MA上向き & 陽線（{len(results)}件）\n")
    footer = f"\n...他{len(results)-30}件" if len(results) > 30 else ""
    msg = header + "\n".join(lines) + footer

    for i in range(0, len(msg), 1900):
        requests.post(DISCORD_WEBHOOK, json={"content": msg[i:i+1900]})

# ===== 実行 =====
if __name__ == "__main__":
    print("プライム銘柄リスト取得中...")
    codes = get_prime_codes()
    print(f"対象銘柄数: {len(codes)}件")

    print("スクリーニング中...")
    results = screen(codes)
    print(f"該当: {len(results)}件")

    notify(results)
    print("完了！")
