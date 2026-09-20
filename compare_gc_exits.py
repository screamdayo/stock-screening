import argparse
import os
from pathlib import Path

import pandas as pd

import download
from compare_gc import YEARS, DEEP_DROP_START_PCT, DEEP_DROP_RESET_PCT, prepare, summarize, fmt


EXITS = ["fixed10", "dead_cross", "gakutto"]
MAX_HOLD = 60


def stats(rows):
    rets = [r["ret_pct"] for r in rows]
    base = summarize(rets)
    if not rows:
        return {
            **base,
            "avg_hold": None,
            "median_hold": None,
            "avg_mfe": None,
            "avg_giveback": None,
            "forced": 0,
            "open": 0,
        }
    s_hold = pd.Series([r["hold_days"] for r in rows], dtype=float)
    s_mfe = pd.Series([r["mfe_pct"] for r in rows], dtype=float)
    s_give = pd.Series([r["giveback_pct"] for r in rows], dtype=float)
    return {
        **base,
        "avg_hold": float(s_hold.mean()),
        "median_hold": float(s_hold.median()),
        "avg_mfe": float(s_mfe.mean()),
        "avg_giveback": float(s_give.mean()),
        "forced": sum(1 for r in rows if r["exit_reason"] == "forced_60d"),
        "open": 0,
    }


def get_price(g, idx, col):
    v = pd.to_numeric(g.loc[idx, col], errors="coerce")
    return None if pd.isna(v) else float(v)


def simulate_exit(g, entry_idx, mode):
    entry = get_price(g, entry_idx, "O")
    if entry is None or entry <= 0:
        return None, False

    n = len(g)

    if mode == "fixed10":
        exit_idx = entry_idx + 9
        if exit_idx >= n:
            return None, True
        exit_px = get_price(g, exit_idx, "C")
        if exit_px is None:
            return None, True
        reason = "fixed_10d"
    else:
        signal_idx = None
        last_signal_idx = min(entry_idx + MAX_HOLD - 1, n - 2)

        for i in range(entry_idx, last_signal_idx + 1):
            if mode == "dead_cross":
                if i < 1:
                    continue
                vals = [g.loc[i - 1, "MA5"], g.loc[i - 1, "MA25"], g.loc[i, "MA5"], g.loc[i, "MA25"]]
                if any(pd.isna(x) for x in vals):
                    continue
                if vals[0] >= vals[1] and vals[2] < vals[3]:
                    signal_idx = i
                    break

            elif mode == "gakutto":
                if i < 2:
                    continue
                ma5_2 = g.loc[i - 2, "MA5"]
                ma5_1 = g.loc[i - 1, "MA5"]
                ma5_0 = g.loc[i, "MA5"]
                o = get_price(g, i, "O")
                c = get_price(g, i, "C")
                if any(pd.isna(x) for x in [ma5_2, ma5_1, ma5_0]) or o is None or c is None:
                    continue
                # がくっと = MA5が上向きから下向きへ転換し、当日が陰線
                if ma5_1 > ma5_2 and ma5_0 < ma5_1 and c < o:
                    signal_idx = i
                    break

        if signal_idx is not None:
            exit_idx = signal_idx + 1
            if exit_idx >= n:
                return None, True
            exit_px = get_price(g, exit_idx, "O")
            if exit_px is None:
                return None, True
            reason = mode
        else:
            forced_idx = entry_idx + MAX_HOLD - 1
            if forced_idx >= n:
                return None, True
            exit_idx = forced_idx
            exit_px = get_price(g, exit_idx, "C")
            if exit_px is None:
                return None, True
            reason = "forced_60d"

    # MFE is based on closes from entry day through exit day.
    closes = pd.to_numeric(g.loc[entry_idx:exit_idx, "C"], errors="coerce").dropna()
    if closes.empty:
        return None, False
    max_close = float(closes.max())
    mfe = (max_close / entry - 1) * 100
    ret = (exit_px / entry - 1) * 100
    giveback = mfe - ret

    return {
        "ret_pct": ret,
        "hold_days": exit_idx - entry_idx + 1,
        "mfe_pct": mfe,
        "giveback_pct": giveback,
        "exit_reason": reason,
    }, False


def collect_for_mode(price_df, target_codes, mode):
    targets = set(map(str, target_codes))
    trades = []
    unresolved = 0

    for code, group in price_df.groupby("Code"):
        code = str(code)
        if code not in targets:
            continue

        g = prepare(group)
        deep_episode = False
        gc_count = 0

        for i in range(25, len(g) - 1):
            dd = g.loc[i, "DRAWDOWN60_PCT"]

            if not pd.isna(dd):
                if not deep_episode and float(dd) <= DEEP_DROP_START_PCT:
                    deep_episode = True
                    gc_count = 0
                elif deep_episode and float(dd) > DEEP_DROP_RESET_PCT:
                    deep_episode = False
                    gc_count = 0

            ma5_prev, ma25_prev = g.loc[i - 1, ["MA5", "MA25"]]
            ma5_now, ma25_now = g.loc[i, ["MA5", "MA25"]]
            if any(pd.isna(x) for x in [ma5_prev, ma25_prev, ma5_now, ma25_now]):
                continue
            if not (ma5_prev <= ma25_prev and ma5_now > ma25_now):
                continue

            if not deep_episode:
                continue
            gc_count += 1
            if gc_count != 3:
                continue

            slope = g.loc[i, "MA25_5D_SLOPE_PCT"]
            if pd.isna(slope) or float(slope) < 1.0:
                continue

            below_days = 0
            j = i - 1
            while j >= 0:
                m5 = g.loc[j, "MA5"]
                m25 = g.loc[j, "MA25"]
                if pd.isna(m5) or pd.isna(m25) or float(m5) > float(m25):
                    break
                below_days += 1
                j -= 1
            if not 6 <= below_days <= 10:
                continue

            entry_idx = i + 1
            result, is_unresolved = simulate_exit(g, entry_idx, mode)
            if is_unresolved:
                unresolved += 1
                continue
            if result is None:
                continue

            result.update({
                "code": code,
                "signal_date": str(pd.Timestamp(g.loc[i, "Date"]).date()),
                "entry_date": str(pd.Timestamp(g.loc[entry_idx, "Date"]).date()),
            })
            trades.append(result)

    return trades, unresolved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exit", choices=EXITS, required=True)
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)
    target_codes = download.get_target_codes()
    price_df = download.get_price_history_incremental(
        cache_filename="backtest_prices_prime_5y.csv", years=YEARS
    )
    price_df["Date"] = pd.to_datetime(price_df["Date"])

    rows, unresolved = collect_for_mode(price_df, target_codes, args.exit)
    s = stats(rows)

    labels = {
        "fixed10": "10 trading days fixed; exit at 10th holding-day close",
        "dead_cross": "MA5 crosses from >= MA25 to < MA25; exit next open",
        "gakutto": "MA5 turns from rising to falling AND signal day is bearish candle; exit next open",
    }

    lines = [
        f"GC exit comparison: {args.exit} / Prime / {YEARS} years through {price_df['Date'].max().date()}",
        "Entry condition = exactly 3rd GC in deep-drop episode + MA25 5d slope >= +1.0% + MA5<=MA25 for 6-10 prior trading days",
        "Entry = next open after GC",
        f"Exit = {labels[args.exit]}",
        f"Signal exits capped at {MAX_HOLD} trading days; if no signal by then, exit at day-{MAX_HOLD} close",
        "Unresolved near dataset end are excluded and reported separately.",
        "",
        "[Overall]",
        f"n={s['n']} unresolved={unresolved} win={fmt(s['win_rate'])}% avg={fmt(s['avg'])}% median={fmt(s['median'])}%",
        f"avg_hold={fmt(s['avg_hold'])}d median_hold={fmt(s['median_hold'])}d",
        f"avg_MFE={fmt(s['avg_mfe'])}% avg_giveback_from_best_close={fmt(s['avg_giveback'])}%",
        f"forced_60d={s['forced']}",
    ]

    text = "\n".join(lines)
    out = Path(f"output/gc_exit_{args.exit}.txt")
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
