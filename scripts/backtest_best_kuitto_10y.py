import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
BATCH_DIR = DATA_DIR / "batches"
RESULT_DIR = Path("results")

# Frozen "best kuitto" rule from the 5-year exploration (2026-09-09):
# Entry signal = kuitto_pullback_auto WITHOUT the later 2026-09-16 liquidity filter.
# Exit = -3% stop, otherwise strict gakutto confirmed at close -> next open,
#        otherwise after 15 full trading sessions -> next open.
MA_SHORT = 5
MA_LONG = 25
TURN_LOOKBACK = 2
DECLINE_LOOKBACK = 5
DECLINE_MIN_PCT = -5.5
DECLINE_MAX_PCT = -2.0
MA5_VS_MA25_MIN_PCT = -5.0
MA5_VS_MA25_MAX_PCT = 0.0
CLOSE_VS_MA5_MAX_PCT = 4.0
BULL_MIN_PCT = 1.5
BULL_MAX_PCT = 3.5
VOLUME_RATIO_MIN = 1.25
STOP_PCT = -3.0
MAX_HOLD_SESSIONS = 15


def load_archive():
    paths = sorted(BATCH_DIR.glob("batch_*.parquet"))
    if not paths:
        raise RuntimeError("No archive batches found.")

    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        cols = [c for c in ["Code", "Date", "O", "H", "L", "C", "Vo", "AdjO", "AdjH", "AdjL", "AdjC", "AdjVo", "ArchiveMarket"] if c in df.columns]
        frames.append(df[cols].copy())

    df = pd.concat(frames, ignore_index=True)

    # Match the original 5-year tests: Prime only.
    if "ArchiveMarket" in df.columns:
        df = df[df["ArchiveMarket"] == "プライム"].copy()

    # Prefer adjusted OHLCV when available, same idea as production downloader.
    for raw, adj in [("O","AdjO"),("H","AdjH"),("L","AdjL"),("C","AdjC"),("Vo","AdjVo")]:
        if adj in df.columns:
            if raw in df.columns:
                df[raw] = df[adj].where(df[adj].notna(), df[raw])
            else:
                df[raw] = df[adj]

    for c in ["O","H","L","C","Vo"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Code"] = df["Code"].astype(str)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.drop_duplicates(["Code","Date"], keep="last")
    df = df.sort_values(["Code","Date"]).reset_index(drop=True)
    return df


def prepare(g):
    g = g.dropna(subset=["O","H","L","C"]).sort_values("Date").reset_index(drop=True).copy()
    g["MA5"] = g["C"].rolling(MA_SHORT).mean()
    g["MA25"] = g["C"].rolling(MA_LONG).mean()
    g["MA25_SLOPE5_PCT"] = (g["MA25"] / g["MA25"].shift(5) - 1) * 100
    g["HIGH20"] = g["C"].rolling(20).max()
    g["DD20_PCT"] = (g["C"] / g["HIGH20"] - 1) * 100
    tr1 = g["H"] - g["L"]
    tr2 = (g["H"] - g["C"].shift(1)).abs()
    tr3 = (g["L"] - g["C"].shift(1)).abs()
    g["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    g["ATR14_PCT"] = g["TR"].rolling(14).mean() / g["C"] * 100
    return g


def is_entry_signal(g, i):
    if i < MA_LONG or i < DECLINE_LOOKBACK + 1:
        return False

    r = g.iloc[i]
    if not (r["O"] > 0 and r["MA5"] > 0 and r["MA25"] > 0):
        return False

    bull = (r["C"] / r["O"] - 1) * 100
    if not (BULL_MIN_PCT <= bull <= BULL_MAX_PCT):
        return False

    ma_gap = (r["MA5"] / r["MA25"] - 1) * 100
    if not (MA5_VS_MA25_MIN_PCT <= ma_gap <= MA5_VS_MA25_MAX_PCT):
        return False

    base = i - 1
    past = base - DECLINE_LOOKBACK
    if past < 0 or not g["MA5"].iloc[past] > 0:
        return False

    decline = (g["MA5"].iloc[base] / g["MA5"].iloc[past] - 1) * 100
    if not (DECLINE_MIN_PCT <= decline <= DECLINE_MAX_PCT):
        return False

    # Previous 2 MA5 slopes must be down/flat; today is first upturn.
    for j in range(i - TURN_LOOKBACK, i):
        if g["MA5"].iloc[j] > g["MA5"].iloc[j - 1]:
            return False
    if not g["MA5"].iloc[i] > g["MA5"].iloc[i - 1]:
        return False

    close_gap = (r["C"] / r["MA5"] - 1) * 100
    if close_gap > CLOSE_VS_MA5_MAX_PCT:
        return False

    prev_v = g["Vo"].iloc[i - 1]
    today_v = r["Vo"]
    if pd.isna(prev_v) or pd.isna(today_v) or not prev_v > 0:
        return False
    if today_v / prev_v < VOLUME_RATIO_MIN:
        return False

    return True



def passes_composite_refined_rule(g, i):
    r = g.iloc[i]
    atr = r["ATR14_PCT"]
    dd20 = r["DD20_PCT"]
    slope = r["MA25_SLOPE5_PCT"]
    if pd.isna(atr) or pd.isna(dd20) or pd.isna(slope):
        return False
    if atr < 3.4:
        return False
    if -7.0 < dd20 <= -5.5:
        return True
    if -9.0 < dd20 <= -7.0 and slope >= -1.5:
        return True
    return False


def is_strict_gakutto(g, i):
    # Two prior MA5 intervals rising/flat, then today's MA5 turns down, with bearish candle.
    if i < 3:
        return False
    if any(pd.isna(g["MA5"].iloc[k]) for k in [i-3, i-2, i-1, i]):
        return False

    prior1 = g["MA5"].iloc[i-2] >= g["MA5"].iloc[i-3]
    prior2 = g["MA5"].iloc[i-1] >= g["MA5"].iloc[i-2]
    turn_down = g["MA5"].iloc[i] < g["MA5"].iloc[i-1]
    bearish = g["C"].iloc[i] < g["O"].iloc[i]
    return prior1 and prior2 and turn_down and bearish


def simulate_trade(g, signal_i):
    entry_i = signal_i + 1
    if entry_i >= len(g):
        return None

    entry = float(g["O"].iloc[entry_i])
    if not np.isfinite(entry) or entry <= 0:
        return None

    stop_price = entry * (1 + STOP_PCT / 100.0)

    # Hold up to 15 full sessions beginning with entry day.
    last_hold_i = min(entry_i + MAX_HOLD_SESSIONS - 1, len(g) - 1)

    for i in range(entry_i, last_hold_i + 1):
        o = float(g["O"].iloc[i])
        l = float(g["L"].iloc[i])

        # If the market gaps through the stop, use the opening price.
        if o <= stop_price:
            exit_price = o
            return make_trade(g, signal_i, entry_i, i, exit_price, "stop_gap", entry)
        if l <= stop_price:
            exit_price = stop_price
            return make_trade(g, signal_i, entry_i, i, exit_price, "stop_-3%", entry)

        if is_strict_gakutto(g, i):
            exit_i = i + 1
            if exit_i < len(g):
                exit_price = float(g["O"].iloc[exit_i])
                return make_trade(g, signal_i, entry_i, exit_i, exit_price, "strict_gakutto_next_open", entry)
            return None

    # Max hold reached: exit next open after the 15th held session.
    exit_i = last_hold_i + 1
    if exit_i < len(g):
        exit_price = float(g["O"].iloc[exit_i])
        return make_trade(g, signal_i, entry_i, exit_i, exit_price, "max15_next_open", entry)

    return None


def make_trade(g, signal_i, entry_i, exit_i, exit_price, reason, entry_price):
    ret = (exit_price / entry_price - 1) * 100
    return {
        "code": str(g["Code"].iloc[entry_i]),
        "signal_date": str(pd.Timestamp(g["Date"].iloc[signal_i]).date()),
        "entry_date": str(pd.Timestamp(g["Date"].iloc[entry_i]).date()),
        "entry_price": entry_price,
        "exit_date": str(pd.Timestamp(g["Date"].iloc[exit_i]).date()),
        "exit_price": exit_price,
        "return_pct": ret,
        "exit_reason": reason,
        "hold_sessions_to_exit": int(exit_i - entry_i),
    }


def fixed10_metrics(df):
    r = df["fixed10_return_pct"].dropna()
    if r.empty:
        return {"n": 0}
    wins = r[r > 0]
    losses = r[r < 0]
    gross_profit = wins.sum()
    gross_loss = -losses.sum()
    pf = gross_profit / gross_loss if gross_loss > 0 else None
    return {
        "n": int(len(r)),
        "win_rate_pct": round(float((r > 0).mean() * 100), 2),
        "avg_return_pct": round(float(r.mean()), 3),
        "median_return_pct": round(float(r.median()), 3),
        "profit_factor": round(float(pf), 3) if pf is not None else None,
    }


def metrics(df):
    if df.empty:
        return {"n": 0}

    r = df["return_pct"]
    wins = r[r > 0]
    losses = r[r < 0]
    gross_profit = wins.sum()
    gross_loss = -losses.sum()
    pf = gross_profit / gross_loss if gross_loss > 0 else None

    return {
        "n": int(len(df)),
        "win_rate_pct": round(float((r > 0).mean() * 100), 2),
        "avg_return_pct": round(float(r.mean()), 3),
        "median_return_pct": round(float(r.median()), 3),
        "profit_factor": round(float(pf), 3) if pf is not None else None,
        "avg_hold_sessions_to_exit": round(float(df["hold_sessions_to_exit"].mean()), 2),
        "gross_profit_pct_points": round(float(gross_profit), 2),
        "gross_loss_pct_points": round(float(gross_loss), 2),
    }


def simulate_trade_variant(g, signal_i, stop_pct, max_hold_sessions, use_gakutto):
    entry_i = signal_i + 1
    if entry_i >= len(g):
        return None
    entry = float(g["O"].iloc[entry_i])
    if not np.isfinite(entry) or entry <= 0:
        return None
    stop_price = entry * (1 + stop_pct / 100.0)
    last_hold_i = min(entry_i + max_hold_sessions - 1, len(g) - 1)
    for i in range(entry_i, last_hold_i + 1):
        o = float(g["O"].iloc[i]); l = float(g["L"].iloc[i])
        if o <= stop_price:
            return make_trade(g, signal_i, entry_i, i, o, "stop_gap", entry)
        if l <= stop_price:
            return make_trade(g, signal_i, entry_i, i, stop_price, f"stop_{stop_pct}%", entry)
        if use_gakutto and is_strict_gakutto(g, i):
            exit_i = i + 1
            if exit_i < len(g):
                return make_trade(g, signal_i, entry_i, exit_i, float(g["O"].iloc[exit_i]), "strict_gakutto_next_open", entry)
            return None
    exit_i = last_hold_i + 1
    if exit_i < len(g):
        return make_trade(g, signal_i, entry_i, exit_i, float(g["O"].iloc[exit_i]), f"max{max_hold_sessions}_next_open", entry)
    return None


def main():
    df = load_archive()
    trades = []
    signal_keys = []
    cache_rows = []

    # Single full-universe scan: identify composite signals once and cache 30 future sessions.
    for code, group in df.groupby("Code", sort=True):
        g = prepare(group)
        if len(g) <= MA_LONG + 1:
            continue
        for i in range(MA_LONG, len(g) - 1):
            if not (is_entry_signal(g, i) and passes_composite_refined_rule(g, i)):
                continue

            signal_keys.append((g, i))
            entry_i = i + 1

            # Cache entry day through +30 sessions for lightweight exit research.
            for offset in range(0, 31):
                k = entry_i + offset
                if k >= len(g):
                    break
                cache_rows.append({
                    "code": str(g["Code"].iloc[k]),
                    "signal_date": str(pd.Timestamp(g["Date"].iloc[i]).date()),
                    "entry_date": str(pd.Timestamp(g["Date"].iloc[entry_i]).date()),
                    "offset": offset,
                    "date": str(pd.Timestamp(g["Date"].iloc[k]).date()),
                    "O": float(g["O"].iloc[k]),
                    "H": float(g["H"].iloc[k]),
                    "L": float(g["L"].iloc[k]),
                    "C": float(g["C"].iloc[k]),
                    "MA5": float(g["MA5"].iloc[k]) if pd.notna(g["MA5"].iloc[k]) else np.nan,
                })

            t = simulate_trade(g, i)
            if t:
                fixed_exit_i = entry_i + 10
                if fixed_exit_i < len(g):
                    fixed_exit = float(g["O"].iloc[fixed_exit_i])
                    if np.isfinite(fixed_exit) and fixed_exit > 0:
                        t["fixed10_return_pct"] = (fixed_exit / t["entry_price"] - 1) * 100
                    else:
                        t["fixed10_return_pct"] = np.nan
                else:
                    t["fixed10_return_pct"] = np.nan
                trades.append(t)

    # Exit grid from the already-found signal list; no extra universe scan.
    exit_grid = []
    for stop_pct in [-3.0, -4.0, -5.0, -6.0, -7.0]:
        for max_hold in [10, 15]:
            for use_gakutto in [True, False]:
                rows = []
                for g, i in signal_keys:
                    tr = simulate_trade_variant(g, i, stop_pct, max_hold, use_gakutto)
                    if tr:
                        rows.append(tr)
                part = pd.DataFrame(rows)
                exit_grid.append({
                    "stop_pct": stop_pct,
                    "max_hold_sessions": max_hold,
                    "use_gakutto": use_gakutto,
                    "metrics": metrics(part) if not part.empty else {"n": 0},
                    "exit_reasons": {str(k): int(v) for k, v in part["exit_reason"].value_counts().to_dict().items()} if not part.empty else {},
                })

    fixed_hold_grid = []
    for hold in [10, 12, 14, 16, 18, 20, 25, 30]:
        vals = []
        for g, i in signal_keys:
            entry_i = i + 1
            exit_i = entry_i + hold
            if exit_i >= len(g):
                continue
            entry = float(g["O"].iloc[entry_i])
            exitp = float(g["O"].iloc[exit_i])
            if np.isfinite(entry) and np.isfinite(exitp) and entry > 0:
                vals.append({
                    "return_pct": (exitp / entry - 1) * 100,
                    "hold_sessions_to_exit": hold,
                })
        part = pd.DataFrame(vals)
        fixed_hold_grid.append({
            "hold_sessions": hold,
            "metrics": metrics(part) if not part.empty else {"n": 0},
        })

    fixed_hold_yearly_compare = {}
    for hold in [14, 16]:
        rows = []
        for g, i in signal_keys:
            entry_i = i + 1
            exit_i = entry_i + hold
            if exit_i >= len(g):
                continue
            entry = float(g["O"].iloc[entry_i])
            exitp = float(g["O"].iloc[exit_i])
            if np.isfinite(entry) and np.isfinite(exitp) and entry > 0:
                rows.append({
                    "entry_date": str(pd.Timestamp(g["Date"].iloc[entry_i]).date()),
                    "return_pct": (exitp / entry - 1) * 100,
                    "hold_sessions_to_exit": hold,
                })
        part = pd.DataFrame(rows)
        if not part.empty:
            part["entry_date_dt"] = pd.to_datetime(part["entry_date"])
            fixed_hold_yearly_compare[str(hold)] = {
                str(int(y)): metrics(gp)
                for y, gp in part.groupby(part["entry_date_dt"].dt.year)
            }
        else:
            fixed_hold_yearly_compare[str(hold)] = {}

    tdf = pd.DataFrame(trades)
    if tdf.empty:
        raise RuntimeError("No trades found.")

    tdf["entry_date_dt"] = pd.to_datetime(tdf["entry_date"])
    archive_start = pd.Timestamp(df["Date"].min()).normalize()
    split = archive_start + pd.DateOffset(years=5)

    older = tdf[(tdf["entry_date_dt"] >= archive_start) & (tdf["entry_date_dt"] < split)]
    recent = tdf[tdf["entry_date_dt"] >= split]

    yearly = {}
    for year, part in tdf.groupby(tdf["entry_date_dt"].dt.year):
        yearly[str(int(year))] = metrics(part)

    exit_reasons = {
        str(k): int(v) for k, v in tdf["exit_reason"].value_counts().to_dict().items()
    }

    summary = {
        "strategy": "refined_kuitto_composite_three_tier",
        "universe": "current Prime listings in saved archive",
        "archive_start": str(pd.Timestamp(df["Date"].min()).date()),
        "archive_end": str(pd.Timestamp(df["Date"].max()).date()),
        "split_date": str(split.date()),
        "entry": {
            "refined_filter": "ATR14>=3.4; -7<DD20<=-5.5 OR (-9<DD20<=-7 AND MA25_SLOPE5>=-1.5); DD20<=-9 excluded",
            "next_open": True,
            "ma5_prior5d_decline_pct": [-5.5, -2.0],
            "ma5_vs_ma25_pct": [-5.0, 0.0],
            "close_vs_ma5_pct_max": 4.0,
            "bullish_candle_pct": [1.5, 3.5],
            "volume_vs_prev_min": 1.25,
            "ma5_prior_two_slopes": "down_or_flat",
            "ma5_today": "first_upturn",
            "liquidity_filter": "not used (it was added later, 2026-09-16)",
        },
        "exit": {
            "stop_loss_pct": -3.0,
            "strict_gakutto": "previous 2 MA5 slopes >=0, today MA5 down, bearish candle; exit next open",
            "max_hold": "15 full trading sessions; exit following open",
            "take_profit": None,
        },
        "overall_10y_dynamic_exit": metrics(tdf),
        "overall_10y_fixed10": fixed10_metrics(tdf),
        "older_5y_dynamic_exit": metrics(older),
        "recent_5y_dynamic_exit": metrics(recent),
        "older_5y_fixed10": fixed10_metrics(older),
        "recent_5y_fixed10": fixed10_metrics(recent),
        "yearly": yearly,
        "exit_reasons": exit_reasons,
        "exit_grid": exit_grid,
        "fixed_hold_grid": fixed_hold_grid,
        "fixed_hold_yearly_compare": fixed_hold_yearly_compare,
        "cache": {
            "path": "results/kuitto_composite_signal_cache.parquet",
            "signals": int(len(signal_keys)),
            "rows": int(len(cache_rows)),
            "future_sessions": 30,
        },
        "note": "Exploratory rule was selected on recent data. Older half is the key robustness check; current-listing universe implies survivorship bias.",
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULT_DIR / "best_kuitto_10y_trades.csv"
    out_json = RESULT_DIR / "best_kuitto_10y_summary.json"
    out_cache = RESULT_DIR / "kuitto_composite_signal_cache.parquet"

    tdf.drop(columns=["entry_date_dt"]).to_csv(out_csv, index=False)
    pd.DataFrame(cache_rows).to_parquet(out_cache, index=False)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
