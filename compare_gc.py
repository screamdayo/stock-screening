import json
import os
from pathlib import Path

import pandas as pd

import download

YEARS = 5
FORWARD_DAYS = [5, 10, 15]
DEEP_DROP_START_PCT = -15.0
DEEP_DROP_RESET_PCT = -5.0


def summarize(vals):
    if not vals:
        return {"n": 0, "win_rate": None, "avg": None, "median": None}
    s = pd.Series(vals, dtype=float)
    return {
        "n": int(len(s)),
        "win_rate": float((s > 0).mean() * 100),
        "avg": float(s.mean()),
        "median": float(s.median()),
    }


def prepare(group):
    g = group.copy().sort_values("Date").reset_index(drop=True)
    g["C"] = pd.to_numeric(g["C"], errors="coerce")
    g["O"] = pd.to_numeric(g["O"], errors="coerce")
    g["MA5"] = g["C"].rolling(5).mean()
    g["MA25"] = g["C"].rolling(25).mean()
    g["MA25_5D_SLOPE_PCT"] = (g["MA25"] / g["MA25"].shift(5) - 1) * 100
    g["HIGH60"] = g["C"].rolling(60, min_periods=20).max()
    g["DRAWDOWN60_PCT"] = (g["C"] / g["HIGH60"] - 1) * 100
    return g


def drawdown_bucket(dd):
    if pd.isna(dd):
        return None
    fall = -float(dd)
    if fall < 5:
        return "0-5%"
    if fall < 10:
        return "5-10%"
    if fall < 15:
        return "10-15%"
    return "15%+"


def slope_bucket(v):
    if pd.isna(v):
        return None
    if v > 0.25:
        return "rising"
    if v < -0.25:
        return "falling"
    return "flat"


def sequence_bucket(n):
    if n == 1:
        return "1st"
    if n == 2:
        return "2nd"
    return "3rd+"


def collect(price_df, target_codes):
    targets = set(map(str, target_codes))
    rows = []

    for code, group in price_df.groupby("Code"):
        code = str(code)
        if code not in targets:
            continue
        g = prepare(group)
        deep_episode = False
        gc_count_in_episode = 0

        for i in range(25, len(g) - max(FORWARD_DAYS) - 1):
            dd = g.loc[i, "DRAWDOWN60_PCT"]

            # A deep-drop episode starts once price is 15%+ below its 60-day high.
            # It stays active until price recovers to within 5% of that high.
            if not pd.isna(dd):
                if not deep_episode and float(dd) <= DEEP_DROP_START_PCT:
                    deep_episode = True
                    gc_count_in_episode = 0
                elif deep_episode and float(dd) > DEEP_DROP_RESET_PCT:
                    deep_episode = False
                    gc_count_in_episode = 0

            ma5_prev, ma25_prev = g.loc[i - 1, ["MA5", "MA25"]]
            ma5_now, ma25_now = g.loc[i, ["MA5", "MA25"]]
            if any(pd.isna(x) for x in [ma5_prev, ma25_prev, ma5_now, ma25_now]):
                continue
            if not (ma5_prev <= ma25_prev and ma5_now > ma25_now):
                continue

            ent = i + 1
            entry = float(g.loc[ent, "O"])
            if not entry > 0:
                continue

            slope = g.loc[i, "MA25_5D_SLOPE_PCT"]
            seq_no = None
            seq_bucket = None
            if deep_episode:
                gc_count_in_episode += 1
                seq_no = gc_count_in_episode
                seq_bucket = sequence_bucket(seq_no)

            rec = {
                "code": code,
                "signal_date": str(pd.Timestamp(g.loc[i, "Date"]).date()),
                "drawdown60_pct": None if pd.isna(dd) else float(dd),
                "drawdown_bucket": drawdown_bucket(dd),
                "ma25_slope5_pct": None if pd.isna(slope) else float(slope),
                "ma25_slope_bucket": slope_bucket(slope),
                "deep_episode_gc_number": seq_no,
                "deep_episode_gc_bucket": seq_bucket,
            }
            for d in FORWARD_DAYS:
                exit_px = float(g.loc[ent + d - 1, "C"])
                rec[f"ret_{d}d"] = (exit_px / entry - 1) * 100
            rows.append(rec)

    return rows


def group_report(rows, key):
    out = {}
    values = sorted({r[key] for r in rows if r.get(key) is not None})
    for v in values:
        subset = [r for r in rows if r.get(key) == v]
        out[v] = {f"{d}d": summarize([r[f"ret_{d}d"] for r in subset]) for d in FORWARD_DAYS}
    return out


def fmt(x, n=2):
    return "-" if x is None else f"{x:.{n}f}"


def append_group(lines, title, rows, key, ordered_values):
    report = group_report(rows, key)
    lines += ["", title]
    for bucket in ordered_values:
        if bucket not in report:
            continue
        parts = []
        for d in FORWARD_DAYS:
            s = report[bucket][f"{d}d"]
            parts.append(
                f"{d}d n={s['n']} win={fmt(s['win_rate'])}% avg={fmt(s['avg'])}% median={fmt(s['median'])}%"
            )
        lines.append(f"{bucket}: " + " | ".join(parts))
    return report


def main():
    os.makedirs("output", exist_ok=True)
    target_codes = download.get_target_codes()
    price_df = download.get_price_history_incremental(
        cache_filename="backtest_prices_prime_5y.csv", years=YEARS
    )
    price_df["Date"] = pd.to_datetime(price_df["Date"])

    rows = collect(price_df, target_codes)
    overall = {f"{d}d": summarize([r[f"ret_{d}d"] for r in rows]) for d in FORWARD_DAYS}
    by_dd = group_report(rows, "drawdown_bucket")
    by_slope = group_report(rows, "ma25_slope_bucket")
    by_sequence = group_report(rows, "deep_episode_gc_bucket")

    deep_falling = [
        r for r in rows
        if r.get("drawdown_bucket") == "15%+" and r.get("ma25_slope_bucket") == "falling"
    ]
    deep_falling_summary = {
        f"{d}d": summarize([r[f"ret_{d}d"] for r in deep_falling]) for d in FORWARD_DAYS
    }

    episode_falling = [
        r for r in rows
        if r.get("deep_episode_gc_bucket") is not None and r.get("ma25_slope_bucket") == "falling"
    ]
    by_sequence_falling_ma25 = group_report(episode_falling, "deep_episode_gc_bucket")

    report = {
        "through": str(price_df["Date"].max().date()),
        "years": YEARS,
        "definition": "MA5 crosses from <= MA25 to > MA25; enter next open",
        "deep_episode_definition": (
            "episode starts when drawdown from 60-day high reaches -15% or worse; "
            "resets after recovery above -5%"
        ),
        "overall": overall,
        "by_drawdown_from_60d_high": by_dd,
        "by_ma25_5d_slope": by_slope,
        "deep_fall_and_falling_ma25": deep_falling_summary,
        "by_gc_sequence_in_deep_episode": by_sequence,
        "by_gc_sequence_in_deep_episode_when_ma25_falling": by_sequence_falling_ma25,
    }
    Path("output/gc_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        f"Golden Cross backtest / Prime / {YEARS} years through {report['through']}",
        "GC = MA5: previous <= MA25, today > MA25; entry = next open",
        "Returns = close after 5/10/15 trading days vs entry",
        "Deep-drop episode = starts at 15%+ below 60d high, ends after recovery to within 5%",
        "",
        "[Overall]",
    ]
    for d in FORWARD_DAYS:
        s = overall[f"{d}d"]
        lines.append(
            f"{d}d n={s['n']} win={fmt(s['win_rate'])}% avg={fmt(s['avg'])}% median={fmt(s['median'])}%"
        )

    lines += ["", "[By drawdown from 60-day high]"]
    for bucket in ["0-5%", "5-10%", "10-15%", "15%+"]:
        if bucket not in by_dd:
            continue
        parts = []
        for d in FORWARD_DAYS:
            s = by_dd[bucket][f"{d}d"]
            parts.append(f"{d}d n={s['n']} win={fmt(s['win_rate'])}% avg={fmt(s['avg'])}%")
        lines.append(f"{bucket}: " + " | ".join(parts))

    lines += ["", "[By MA25 5-day slope]"]
    for bucket in ["falling", "flat", "rising"]:
        if bucket not in by_slope:
            continue
        parts = []
        for d in FORWARD_DAYS:
            s = by_slope[bucket][f"{d}d"]
            parts.append(f"{d}d n={s['n']} win={fmt(s['win_rate'])}% avg={fmt(s['avg'])}%")
        lines.append(f"{bucket}: " + " | ".join(parts))

    lines += ["", "[15%+ drawdown AND MA25 falling]"]
    for d in FORWARD_DAYS:
        s = deep_falling_summary[f"{d}d"]
        lines.append(
            f"{d}d n={s['n']} win={fmt(s['win_rate'])}% avg={fmt(s['avg'])}% median={fmt(s['median'])}%"
        )

    append_group(
        lines,
        "[GC sequence after deep drop: 1st vs 2nd vs 3rd+]",
        rows,
        "deep_episode_gc_bucket",
        ["1st", "2nd", "3rd+"],
    )
    append_group(
        lines,
        "[GC sequence after deep drop, MA25 still falling]",
        episode_falling,
        "deep_episode_gc_bucket",
        ["1st", "2nd", "3rd+"],
    )

    text = "\n".join(lines)
    Path("output/gc_comparison.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
