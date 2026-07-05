import os, requests
from datetime import datetime, timedelta
import pandas as pd

DISCORD_WEBHOOK  = os.environ["DISCORD_WEBHOOK_URL"]
JQUANTS_API_KEY  = os.environ["JQUANTS_API_KEY"]

HEADERS = {"x-api-key": JQUANTS_API_KEY}

# ===== プライム銘柄リスト取得 =====
def get_prime_codes():
    res = requests.get(
        "https://api.jquants.com/v2/equities/master",
        headers=HEADERS
    )
    res.raise_for_status()
    data = res.json()
    df = pd.DataFrame(data["data"])
    prime = df[df["MktNm"] == "プライム"]["Code"].tolist()

    # 優先株式（末尾5,6）を除外
    prime = [code for code in prime if not code.endswith(('5', '6'))]

    print(f"プライム銘柄数: {len(prime)}件")
    return set(prime)


# ===== 指定日の全銘柄分を1リクエストで取得（ページング対応） =====
def get_bars_for_date(date_str):
    all_rows = []
    pagination_key = None

    while True:
        params = {"date": date_str}
        if pagination_key:
            params["pagination_key"] = pagination_key

        res = requests.get(
            "https://api.jquants.com/v2/equities/bars/daily",
            headers=HEADERS,
            params=params
        )
        if res.status_code != 200:
            print(f"  APIエラー response: {res.status_code} {res.text}")
        res.raise_for_status()
        body = res.json()

        rows = body.get("data", [])
        all_rows.extend(rows)

        pagination_key = body.get("pagination_key")
        if not pagination_key:
            break

    return all_rows


# ===== 直近営業日を遡って必要日数分の全銘柄データを収集 =====
def get_bars(target_business_days=8, max_lookback_days=25):
    all_rows = []
    collected_days = 0
    day = datetime.now()
    lookback = 0

    while collected_days < target_business_days and lookback < max_lookback_days:
        date_str = day.strftime("%Y%m%d")  # YYYYMMDD形式（ハイフンなし）
        rows = get_bars_for_date(date_str)
        if rows:
            all_rows.extend(rows)
            collected_days += 1
            print(f"  {date_str}: {len(rows)}件取得")
        else:
            print(f"  {date_str}: データなし（休日等）")
        day -= timedelta(days=1)
        lookback += 1

    print(f"取得した株価レコード数（合計）: {len(all_rows)}件")
    return pd.DataFrame(all_rows)


# ===== スクリーニング =====
def screen(df, prime_codes):
    results = []

    cnt_no_data = 0
    cnt_not_bullish = 0
    cnt_ma_down = 0
    cnt_pass = 0

    df = df[df["Code"].isin(prime_codes)].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Code", "Date"])

    latest_date = df["Date"].max()
    print(f"データ最新日付: {latest_date.date()}")

    for code, g in df.groupby("Code"):
        g = g.dropna(subset=["C", "O"])
        if len(g) < 6:
            cnt_no_data += 1
            continue

        latest = g.iloc[-1]
        open_ = latest["O"]
        close = latest["C"]

        if pd.isna(close) or pd.isna(open_):
            cnt_no_data += 1
            continue

        is_bullish = close > open_
        ma5_today = g["C"].iloc[-5:].mean()
        ma5_prev  = g["C"].iloc[-6:-1].mean()
        is_ma_up = ma5_today > ma5_prev

        if not is_bullish:
            cnt_not_bullish += 1
        if not is_ma_up:
            cnt_ma_down += 1

        if is_bullish and is_ma_up:
            cnt_pass += 1
            results.append({
                "code":      code,
                "close":     round(close, 1),
                "open":      round(open_, 1),
                "ma5_today": round(ma5_today, 1),
                "ma5_prev":  round(ma5_prev, 1),
            })

    print("===== 診断結果 =====")
    print(f"データ不足で除外: {cnt_no_data}件")
    print(f"陽線でない: {cnt_not_bullish}件")
    print(f"MA上向きでない: {cnt_ma_down}件")
    print(f"両条件通過: {cnt_pass}件")
    print("=====================")

    return results


# ===== Discord通知 =====
def notify(results):
    today = datetime.now().strftime("%Y/%m/%d")

    if not results:
        msg = f"📊 **株スクリーニング結果 {today}**\n該当銘柄なし"
        requests.post(DISCORD_WEBHOOK, json={"content": msg})
        return

    lines = [
        f"・**{r['code']}**｜終値 {r['close']}円｜5MA {r['ma5_prev']}→{r['ma5_today']}↑\n  https://kabutan.jp/stock/chart?code={r['code']}"
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
    prime_codes = get_prime_codes()
    print(f"対象銘柄数: {len(prime_codes)}件")

    print("株価データ取得中（直近8営業日分を日付ごとに取得）...")
    bars_df = get_bars(target_business_days=8, max_lookback_days=25)

    print("スクリーニング中...")
    results = screen(bars_df, prime_codes)
    print(f"該当: {len(results)}件")

    notify(results)
    print("完了！")
