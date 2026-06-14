import os, requests, yfinance as yf
from datetime import datetime
import pandas as pd

DISCORD_WEBHOOK  = os.environ["DISCORD_WEBHOOK_URL"]
JQUANTS_EMAIL    = os.environ["JQUANTS_EMAIL"]
JQUANTS_PASSWORD = os.environ["JQUANTS_PASSWORD"]

# ===== J-Quants 認証 =====
def get_token():
    # Step1: リフレッシュトークン取得
    res = requests.post(
        "https://api.jquants.com/v2/token/auth_user",
        json={"mailaddress": JQUANTS_EMAIL, "password": JQUANTS_PASSWORD}
    )
    print("認証レスポンス:", res.status_code, res.text)  # デバッグ用
    
    body = res.json()
    refresh_token = body.get("refreshToken") or body.get("refresh_token")
    if not refresh_token:
        raise Exception(f"リフレッシュトークン取得失敗: {body}")

    # Step2: IDトークン取得
    res2 = requests.post(
        "https://api.jquants.com/v1/token/auth_refresh",
        params={"refreshtoken": refresh_token}
    )
    print("トークンレスポンス:", res2.status_code, res2.text)  # デバッグ用

    body2 = res2.json()
    id_token = body2.get("idToken") or body2.get("id_token")
    if not id_token:
        raise Exception(f"IDトークン取得失敗: {body2}")
    
    return id_token

# ===== プライム銘柄リストだけ取得 =====
def get_prime_codes(token):
    res = requests.get("https://api.jquants.com/v1/listed/info",
        headers={"Authorization": f"Bearer {token}"})
    df = pd.DataFrame(res.json()["info"])
    prime = df[df["MarketCodeName"] == "プライム"]["Code"].tolist()
    return prime

# ===== yfinance で株価取得 =====
def get_prices_yf(code):
    ticker = yf.Ticker(f"{code}.T")
    df = ticker.history(period="20d")
    if df.empty or len(df) < 6:
        return None
    return df

# ===== スクリーニング =====
def screen(codes):
    results = []

    for i, code in enumerate(codes):
        if i % 100 == 0:
            print(f"  {i}/{len(codes)}件処理中...")
        try:
            df = get_prices_yf(code)
            if df is None:
                continue

            latest = df.iloc[-1]
            prev5  = df.iloc[-6:-1]  # 前日までの5日分

            open_  = latest["Open"]
            close  = latest["Close"]

            # 陽線チェック
            if close <= open_:
                continue

            # 5日MA上向きチェック
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
    if len(results) > 30:
        footer = f"\n...他{len(results)-30}件"
    else:
        footer = ""

    msg = header + "\n".join(lines) + footer

    # 2000文字制限で分割
    for i in range(0, len(msg), 1900):
        requests.post(DISCORD_WEBHOOK, json={"content": msg[i:i+1900]})

# ===== 実行 =====
if __name__ == "__main__":
    print("トークン取得中...")
    token = get_token()

    print("プライム銘柄リスト取得中...")
    codes = get_prime_codes(token)
    print(f"対象銘柄数: {len(codes)}件")

    print("スクリーニング中...")
    results = screen(codes)
    print(f"該当: {len(results)}件")

    notify(results)
    print("完了！")
