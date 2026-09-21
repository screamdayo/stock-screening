import json
from pathlib import Path
import pandas as pd

SRC = Path("results/gc_seq2_high_confidence_trades.csv")
OUT = Path("results/gc_seq2_strong_breakout_robustness.json")

VOL_RATIO_MIN = 0.88
GC_GAP_MIN = 0.13
CANDLE_MIN = 2.14
SPLIT = pd.Timestamp("2021-09-20")


def metrics(df):
    r = pd.to_numeric(df["ret10"], errors="coerce").dropna()
    if r.empty:
        return {"n": 0}
    gp = r[r > 0].sum()
    gl = -r[r < 0].sum()
    return {
        "n": int(len(r)),
        "win_rate_pct": round(float((r > 0).mean() * 100), 2),
        "avg_ret10_pct": round(float(r.mean()), 3),
        "median_ret10_pct": round(float(r.median()), 3),
        "p5_plus_pct": round(float((r >= 5).mean() * 100), 2),
        "p10_plus_pct": round(float((r >= 10).mean() * 100), 2),
        "loss5_or_worse_pct": round(float((r <= -5).mean() * 100), 2),
        "profit_factor": round(float(gp / gl), 3) if gl > 0 else None,
        "worst_ret10_pct": round(float(r.min()), 3),
        "best_ret10_pct": round(float(r.max()), 3),
    }


def main():
    df = pd.read_csv(SRC, dtype={"code": str}, parse_dates=["signal_date"])
    selected = df[
        (df["vol_ratio20"] > VOL_RATIO_MIN)
        & (df["gc_gap_pct"] > GC_GAP_MIN)
        & (df["candle_pct"] > CANDLE_MIN)
    ].copy()

    older = selected[selected["signal_date"] < SPLIT]
    recent = selected[selected["signal_date"] >= SPLIT]

    yearly = {
        str(int(y)): metrics(g)
        for y, g in selected.groupby(selected["signal_date"].dt.year)
    }

    # Rolling calendar blocks to see whether the effect is confined to one era.
    blocks = {}
    for start_year in range(2017, 2026, 2):
        start = pd.Timestamp(f"{start_year}-01-01")
        end = pd.Timestamp(f"{start_year + 2}-01-01")
        blocks[f"{start_year}-{start_year+1}"] = metrics(
            selected[(selected["signal_date"] >= start) & (selected["signal_date"] < end)]
        )

    out = {
        "strategy": "GC seq2 strong breakout post-hoc branch",
        "rules": {
            "vol_ratio20_gt": VOL_RATIO_MIN,
            "gc_gap_pct_gt": GC_GAP_MIN,
            "candle_pct_gt": CANDLE_MIN,
        },
        "important_caveat": "These thresholds were discovered after inspecting the later-period decision-tree leaf, so this is NOT an untouched OOS validation. Use only as a robustness/stability check.",
        "all_10y": metrics(selected),
        "older5": metrics(older),
        "recent5": metrics(recent),
        "yearly": yearly,
        "two_year_blocks": blocks,
        "signals_per_year_avg": round(len(selected) / 10.0, 2),
        "selected_codes_dates": [
            {
                "code": str(r["code"]),
                "signal_date": r["signal_date"].date().isoformat(),
                "ret10": round(float(r["ret10"]), 3),
            }
            for _, r in selected.sort_values("signal_date").iterrows()
        ],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

# trigger run 2026-09-22
