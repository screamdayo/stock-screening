import json
from pathlib import Path
import numpy as np
import pandas as pd

RESULT_DIR = Path("results")
CACHE = RESULT_DIR / "kuitto_composite_signal_cache.parquet"


def metrics(df):
    if df.empty:
        return {"n": 0}
    r = df["return_pct"].dropna()
    if r.empty:
        return {"n": 0}
    wins = r[r > 0]
    losses = r[r < 0]
    gp = wins.sum()
    gl = -losses.sum()
    return {
        "n": int(len(r)),
        "win_rate_pct": round(float((r > 0).mean() * 100), 2),
        "avg_return_pct": round(float(r.mean()), 3),
        "median_return_pct": round(float(r.median()), 3),
        "profit_factor": round(float(gp / gl), 3) if gl > 0 else None,
    }


def strict_gakutto(path, idx):
    if idx - 3 < 0:
        return False
    vals = path.iloc[idx-3:idx+1]
    if vals["MA5"].isna().any():
        return False
    m = vals["MA5"].to_numpy()
    prior1 = m[1] >= m[0]
    prior2 = m[2] >= m[1]
    turn_down = m[3] < m[2]
    bearish = float(path.iloc[idx]["C"]) < float(path.iloc[idx]["O"])
    return prior1 and prior2 and turn_down and bearish


def simulate(path, stop_pct=None, max_hold=16, use_gakutto=False):
    entry_row = path[path["offset"] == 0]
    if entry_row.empty:
        return None
    entry = float(entry_row.iloc[0]["O"])
    if not np.isfinite(entry) or entry <= 0:
        return None

    usable = path[(path["offset"] >= 0) & (path["offset"] <= max_hold)].sort_values("offset").reset_index(drop=True)
    if usable.empty:
        return None

    stop_price = entry * (1 + stop_pct / 100.0) if stop_pct is not None else None

    for idx, row in usable.iterrows():
        off = int(row["offset"])
        if off >= max_hold:
            return {
                "return_pct": (float(row["O"]) / entry - 1) * 100,
                "exit_reason": f"fixed_{max_hold}",
                "hold_sessions_to_exit": max_hold,
            }
        if stop_price is not None:
            o = float(row["O"])
            l = float(row["L"])
            if o <= stop_price:
                return {"return_pct": (o / entry - 1) * 100, "exit_reason": "stop_gap", "hold_sessions_to_exit": off}
            if l <= stop_price:
                return {"return_pct": stop_pct, "exit_reason": f"stop_{stop_pct}", "hold_sessions_to_exit": off}

        if use_gakutto:
            full_idx = path.index[path["offset"] == off]
            if len(full_idx):
                pos = path.index.get_loc(full_idx[0])
                if strict_gakutto(path.reset_index(drop=True), pos):
                    nxt = path[path["offset"] == off + 1]
                    if not nxt.empty:
                        px = float(nxt.iloc[0]["O"])
                        return {"return_pct": (px / entry - 1) * 100, "exit_reason": "strict_gakutto_next_open", "hold_sessions_to_exit": off + 1}
    return None


def main():
    if not CACHE.exists():
        raise RuntimeError(f"Missing cache: {CACHE}")
    df = pd.read_parquet(CACHE).sort_values(["code", "signal_date", "offset"]).reset_index(drop=True)

    groups = [g.reset_index(drop=True) for _, g in df.groupby(["code", "signal_date"], sort=False)]

    fixed_holds = {}
    for hold in [10, 12, 14, 15, 16, 18, 20, 25, 30]:
        rows = [simulate(g, stop_pct=None, max_hold=hold, use_gakutto=False) for g in groups]
        part = pd.DataFrame([r for r in rows if r is not None])
        fixed_holds[str(hold)] = metrics(part)

    exit_grid = []
    for stop in [-3.0, -4.0, -5.0, -6.0, -7.0]:
        for hold in [10, 15, 16]:
            for use_gakutto in [True, False]:
                rows = [simulate(g, stop_pct=stop, max_hold=hold, use_gakutto=use_gakutto) for g in groups]
                part = pd.DataFrame([r for r in rows if r is not None])
                exit_grid.append({
                    "stop_pct": stop,
                    "max_hold_sessions": hold,
                    "use_gakutto": use_gakutto,
                    "metrics": metrics(part),
                    "exit_reasons": part["exit_reason"].value_counts().to_dict() if not part.empty else {},
                })

    out = {
        "source_cache": str(CACHE),
        "signal_count": len(groups),
        "fixed_hold_grid": fixed_holds,
        "exit_grid": exit_grid,
    }
    out_path = RESULT_DIR / "kuitto_exit_cache_summary.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
