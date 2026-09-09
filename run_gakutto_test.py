"""がくっと検証専用。

現行 ma5_breakout（くいっと）の左右反転版を、既存本番コードを変更せず検証する。
条件:
- 直近5営業日でMA5が0.5%以上上昇していた
- シグナル当日 MA5 >= MA25
- 直近2日間のMA5傾きが上向き/横ばい
- 当日初めてMA5が下向き
- 当日が陰線かつ始値→終値 -1.5%以上
- 翌営業日始値で空売り
- 株価 -5%で利確 / +3%で損切り / 最大10営業日
"""
import json
import os
import pandas as pd

import config
import download

LOOKBACK_UP_DAYS = 5
MIN_MA5_RISE_PCT = 0.5
TURN_LOOKBACK_DAYS = 2
MIN_BEARISH_PCT = 1.5
TP_PCT = 5.0
SL_PCT = 3.0
HOLD_DAYS = 10


def detect_signals(price_df, target_codes):
    signals = []
    by_code = {}
    df = price_df[price_df["Code"].isin(target_codes)].copy()
    for code, group in df.groupby("Code"):
        g = group.dropna(subset=["O", "H", "L", "C"]).sort_values("Date").reset_index(drop=True)
        if len(g) < 30:
            continue
        g["MA5"] = g["C"].rolling(5).mean()
        g["MA25"] = g["C"].rolling(25).mean()
        by_code[code] = g
        start = max(25, LOOKBACK_UP_DAYS + 1, TURN_LOOKBACK_DAYS + 1)
        for i in range(start, len(g) - 1):
            row = g.iloc[i]
            if pd.isna(row["MA5"]) or pd.isna(row["MA25"]):
                continue
            # 強い陰線
            if row["O"] <= 0 or (row["C"] / row["O"] - 1) * 100 > -MIN_BEARISH_PCT:
                continue
            # MA5がMA25以上
            if row["MA5"] < row["MA25"]:
                continue
            # 転換前にMA5が実際に上昇していた
            base_i = i - 1
            past_i = base_i - LOOKBACK_UP_DAYS
            if past_i < 0 or pd.isna(g["MA5"].iloc[past_i]) or g["MA5"].iloc[past_i] == 0:
                continue
            rise_pct = (g["MA5"].iloc[base_i] / g["MA5"].iloc[past_i] - 1) * 100
            if rise_pct < MIN_MA5_RISE_PCT:
                continue
            # 直近2日上向き/横ばい → 今日初めて下向き
            ok = True
            for j in range(i - TURN_LOOKBACK_DAYS, i):
                if g["MA5"].iloc[j] < g["MA5"].iloc[j - 1]:
                    ok = False
                    break
            if not ok or not (g["MA5"].iloc[i] < g["MA5"].iloc[i - 1]):
                continue
            signals.append((code, i))
    return signals, by_code


def simulate(signals, by_code):
    trades = []
    for code, sig_i in signals:
        g = by_code[code]
        ent_i = sig_i + 1
        if ent_i >= len(g):
            continue
        entry = float(g.iloc[ent_i]["O"])
        if entry <= 0:
            continue
        tp = entry * (1 - TP_PCT / 100)
        sl = entry * (1 + SL_PCT / 100)
        end_i = min(ent_i + HOLD_DAYS - 1, len(g) - 1)
        exit_price = None
        reason = None
        exit_date = None
        days = 0
        for i in range(ent_i, end_i + 1):
            r = g.iloc[i]
            days = i - ent_i + 1
            hit_tp = float(r["L"]) <= tp
            hit_sl = float(r["H"]) >= sl
            # 同日両方なら保守的に損切り優先
            if hit_sl:
                exit_price, reason, exit_date = sl, "stop_loss", r["Date"]
                break
            if hit_tp:
                exit_price, reason, exit_date = tp, "take_profit", r["Date"]
                break
        if exit_price is None:
            r = g.iloc[end_i]
            exit_price, reason, exit_date = float(r["C"]), "time_exit", r["Date"]
            days = end_i - ent_i + 1
        # 空売り損益率
        profit_pct = (entry - exit_price) / entry * 100
        trades.append({
            "code": code,
            "signal_date": str(pd.Timestamp(g.iloc[sig_i]["Date"]).date()),
            "entry_date": str(pd.Timestamp(g.iloc[ent_i]["Date"]).date()),
            "entry_price": round(entry, 2),
            "exit_date": str(pd.Timestamp(exit_date).date()),
            "exit_price": round(exit_price, 2),
            "exit_reason": reason,
            "holding_days": days,
            "profit_pct": round(profit_pct, 3),
        })
    return trades


def summarize(trades):
    if not trades:
        return {"total_trades": 0}
    s = pd.Series([t["profit_pct"] for t in trades], dtype=float)
    wins = s[s > 0]
    losses = s[s <= 0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else None
    return {
        "total_trades": int(len(s)),
        "win_count": int((s > 0).sum()),
        "loss_count": int((s <= 0).sum()),
        "win_rate": round(float((s > 0).mean() * 100), 2),
        "avg_profit_pct_all": round(float(s.mean()), 3),
        "median_profit_pct": round(float(s.median()), 3),
        "profit_factor": round(float(pf), 3) if pf is not None else None,
        "take_profit_count": sum(t["exit_reason"] == "take_profit" for t in trades),
        "stop_loss_count": sum(t["exit_reason"] == "stop_loss" for t in trades),
        "time_exit_count": sum(t["exit_reason"] == "time_exit" for t in trades),
    }


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    cache = f"backtest_prices_{config.TARGET_MARKET}_{config.BACKTEST_YEARS}y.csv"
    target_codes = download.get_target_codes()
    prices = download.get_price_history_incremental(cache_filename=cache, years=config.BACKTEST_YEARS)
    signals, by_code = detect_signals(prices, target_codes)
    trades = simulate(signals, by_code)
    summary = summarize(trades)
    pd.DataFrame(trades).to_csv(os.path.join(config.OUTPUT_DIR, "gakutto_trades.csv"), index=False)
    with open(os.path.join(config.OUTPUT_DIR, "gakutto_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("GAKUTTO_SUMMARY=" + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
