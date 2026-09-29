import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
BATCH_DIR = DATA_DIR / "batches"
TOPIX_PATH = DATA_DIR / "topix" / "topix_daily.parquet"
SIGNALS_PATH = Path("results/kuitto_runner_features_trades.csv")
OUT_JSON = Path("results/gakutto_market_rescue_summary.json")
OUT_CSV = Path("results/gakutto_market_rescue_variants.csv")

STOP_PCT = -5.0
MAX_HOLD_SESSIONS = 15
LIQUIDITY_MIN = 500_000_000
SPLIT_DATE = pd.Timestamp("2021-09-20")
DELAYS = [1, 3, 5, 10]

# Predefined before looking at results.
CONDITIONS = {
    "topix_le_-1": {"topix_max": -1.0},
    "breadth_ge_70": {"decline_min": 70.0},
    "both_-1_70": {"topix_max": -1.0, "decline_min": 70.0},
    "both_-1_80": {"topix_max": -1.0, "decline_min": 80.0},
    "both_-1.5_70": {"topix_max": -1.5, "decline_min": 70.0},
}
MAIN_CONDITION = "both_-1_70"


def norm_code(v):
    s = str(v).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if len(s) == 5 and s.endswith("0"):
        s = s[:-1]
    return s


def load_archive():
    paths = sorted(BATCH_DIR.glob("batch_*.parquet"))
    if not paths:
        raise RuntimeError("No archive batches found")

    frames = []
    want = [
        "Code", "Date", "O", "H", "L", "C", "Vo",
        "AdjO", "AdjH", "AdjL", "AdjC", "AdjVo", "ArchiveMarket",
    ]
    for p in paths:
        z = pd.read_parquet(p)
        cols = [c for c in want if c in z.columns]
        frames.append(z[cols].copy())

    df = pd.concat(frames, ignore_index=True)
    if "ArchiveMarket" in df.columns:
        df = df[df["ArchiveMarket"] == "プライム"].copy()

    for raw, adj in [("O", "AdjO"), ("H", "AdjH"), ("L", "AdjL"), ("C", "AdjC"), ("Vo", "AdjVo")]:
        if adj in df.columns:
            if raw in df.columns:
                df[raw] = df[adj].where(df[adj].notna(), df[raw])
            else:
                df[raw] = df[adj]

    for c in ["O", "H", "L", "C", "Vo"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Code"] = df["Code"].map(norm_code)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Code", "Date", "C"])
    df = df.drop_duplicates(["Code", "Date"], keep="last")
    return df.sort_values(["Code", "Date"]).reset_index(drop=True)


def build_market_context(df):
    b = df[["Code", "Date", "C"]].copy()
    b = b.dropna(subset=["C"]).sort_values(["Code", "Date"])
    b["prev_close"] = b.groupby("Code")["C"].shift(1)
    b = b[b["prev_close"].notna() & (b["prev_close"] > 0)].copy()
    b["ret1"] = (b["C"] / b["prev_close"] - 1) * 100

    rows = []
    for d, g in b.groupby("Date", sort=True):
        r = pd.to_numeric(g["ret1"], errors="coerce").dropna()
        if r.empty:
            continue
        rows.append({
            "date": pd.Timestamp(d),
            "decline_pct": float((r < 0).mean() * 100),
            "advance_pct": float((r > 0).mean() * 100),
            "equal_weight_mean_pct": float(r.mean()),
            "comparable": int(len(r)),
        })
    breadth = pd.DataFrame(rows)

    tx = pd.read_parquet(TOPIX_PATH).copy()
    tx["date"] = pd.to_datetime(tx["Date"], errors="coerce")
    tx["close"] = pd.to_numeric(tx["C"], errors="coerce")
    tx = tx.dropna(subset=["date", "close"]).sort_values("date")
    tx["topix_ret1_pct"] = tx["close"].pct_change() * 100

    ctx = breadth.merge(tx[["date", "topix_ret1_pct"]], on="date", how="left")
    return ctx.sort_values("date").reset_index(drop=True)


def load_signals():
    s = pd.read_csv(SIGNALS_PATH, dtype={"code": str})
    s["code"] = s["code"].map(norm_code)
    s["signal_date"] = pd.to_datetime(s["signal_date"], errors="coerce")
    for c in ["avg_turnover20", "atr14_pct", "dd20_pct", "ma25_slope5_pct"]:
        if c in s.columns:
            s[c] = pd.to_numeric(s[c], errors="coerce")
    required = ["code", "signal_date", "avg_turnover20", "atr14_pct", "dd20_pct", "ma25_slope5_pct"]
    missing = [c for c in required if c not in s.columns]
    if missing:
        raise RuntimeError(f"Missing signal columns: {missing}")
    s = s.dropna(subset=["code", "signal_date"]).drop_duplicates(["code", "signal_date"], keep="last")
    s["signal_key"] = s["code"] + "|" + s["signal_date"].dt.strftime("%Y-%m-%d")
    s["universe_frozen_all"] = True
    s["universe_current_liquid"] = s["avg_turnover20"] >= LIQUIDITY_MIN
    s["universe_winner_liquid"] = (
        s["universe_current_liquid"]
        & (s["atr14_pct"] >= 3.0)
        & (s["dd20_pct"] <= -5.5)
    )
    s["universe_individual_strong_liquid"] = (
        s["universe_winner_liquid"]
        & (s["ma25_slope5_pct"] >= -1.5)
    )
    return s


def prepare(g):
    g = g.dropna(subset=["O", "H", "L", "C"]).sort_values("Date").reset_index(drop=True).copy()
    g["MA5"] = g["C"].rolling(5).mean()
    g.attrs["date_to_i"] = {pd.Timestamp(d).normalize(): i for i, d in enumerate(g["Date"])}
    return g


def is_strict_gakutto(g, i):
    if i < 3:
        return False
    vals = [g["MA5"].iloc[k] for k in [i - 3, i - 2, i - 1, i]]
    if any(pd.isna(v) for v in vals):
        return False
    return (
        g["MA5"].iloc[i - 2] >= g["MA5"].iloc[i - 3]
        and g["MA5"].iloc[i - 1] >= g["MA5"].iloc[i - 2]
        and g["MA5"].iloc[i] < g["MA5"].iloc[i - 1]
        and g["C"].iloc[i] < g["O"].iloc[i]
    )


def condition_hit(ctx_row, spec):
    if ctx_row is None:
        return False
    topix = ctx_row.get("topix_ret1_pct")
    decline = ctx_row.get("decline_pct")
    if "topix_max" in spec:
        if topix is None or pd.isna(topix) or float(topix) > spec["topix_max"]:
            return False
    if "decline_min" in spec:
        if decline is None or pd.isna(decline) or float(decline) < spec["decline_min"]:
            return False
    return True


def exit_record(g, sig, entry_i, exit_i, exit_price, reason, entry_price, gakutto_i=None, rescued=False):
    return {
        "signal_key": sig["signal_key"],
        "code": sig["code"],
        "signal_date": sig["signal_date"].strftime("%Y-%m-%d"),
        "entry_date": pd.Timestamp(g["Date"].iloc[entry_i]).strftime("%Y-%m-%d"),
        "exit_date": pd.Timestamp(g["Date"].iloc[exit_i]).strftime("%Y-%m-%d"),
        "entry_price": float(entry_price),
        "exit_price": float(exit_price),
        "return_pct": float((exit_price / entry_price - 1) * 100),
        "exit_reason": reason,
        "hold_sessions_to_exit": int(exit_i - entry_i),
        "gakutto_date": (
            pd.Timestamp(g["Date"].iloc[gakutto_i]).strftime("%Y-%m-%d")
            if gakutto_i is not None else None
        ),
        "rescued": bool(rescued),
    }


def simulate(g, sig, ctx_map, rescue_spec=None, delay=0):
    date_to_i = g.attrs["date_to_i"]
    signal_i = date_to_i.get(pd.Timestamp(sig["signal_date"]).normalize())
    if signal_i is None:
        return None
    entry_i = signal_i + 1
    if entry_i >= len(g):
        return None

    entry_price = float(g["O"].iloc[entry_i])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None

    stop_price = entry_price * (1 + STOP_PCT / 100.0)
    last_hold_i = min(entry_i + MAX_HOLD_SESSIONS - 1, len(g) - 1)
    scheduled_i = None
    rescued = False
    rescued_gakutto_i = None

    for i in range(entry_i, last_hold_i + 1):
        o = float(g["O"].iloc[i])
        l = float(g["L"].iloc[i])

        if scheduled_i is not None and i >= scheduled_i:
            return exit_record(
                g, sig, entry_i, i, o, f"market_rescue_delay_{delay}", entry_price,
                gakutto_i=rescued_gakutto_i, rescued=True
            )

        if o <= stop_price:
            return exit_record(
                g, sig, entry_i, i, o, "stop_gap_-5%", entry_price,
                gakutto_i=rescued_gakutto_i, rescued=rescued
            )
        if l <= stop_price:
            return exit_record(
                g, sig, entry_i, i, stop_price, "stop_-5%", entry_price,
                gakutto_i=rescued_gakutto_i, rescued=rescued
            )

        if scheduled_i is None and is_strict_gakutto(g, i):
            d = pd.Timestamp(g["Date"].iloc[i]).normalize()
            ctx = ctx_map.get(d)
            if rescue_spec is not None and condition_hit(ctx, rescue_spec):
                scheduled_i = i + 1 + delay
                rescued = True
                rescued_gakutto_i = i
                continue

            exit_i = i + 1
            if exit_i < len(g):
                exit_price = float(g["O"].iloc[exit_i])
                if np.isfinite(exit_price) and exit_price > 0:
                    return exit_record(
                        g, sig, entry_i, exit_i, exit_price,
                        "strict_gakutto_next_open", entry_price, gakutto_i=i
                    )
            return None

    exit_i = last_hold_i + 1
    if exit_i < len(g):
        exit_price = float(g["O"].iloc[exit_i])
        if np.isfinite(exit_price) and exit_price > 0:
            return exit_record(
                g, sig, entry_i, exit_i, exit_price,
                "max15_next_open", entry_price,
                gakutto_i=rescued_gakutto_i, rescued=rescued
            )
    return None


def metrics(df):
    if df is None or df.empty:
        return {"n": 0}
    r = pd.to_numeric(df["return_pct"], errors="coerce").dropna()
    if r.empty:
        return {"n": 0}
    gp = float(r[r > 0].sum())
    gl = float(-r[r < 0].sum())
    return {
        "n": int(len(r)),
        "win_rate_pct": round(float((r > 0).mean() * 100), 2),
        "avg_return_pct": round(float(r.mean()), 3),
        "median_return_pct": round(float(r.median()), 3),
        "p5_plus_pct": round(float((r >= 5).mean() * 100), 2),
        "loss5_pct": round(float((r <= -5).mean() * 100), 2),
        "profit_factor": round(gp / gl, 3) if gl > 0 else None,
        "avg_hold_sessions": round(float(pd.to_numeric(df["hold_sessions_to_exit"], errors="coerce").mean()), 2),
    }


def paired_metrics(base, var):
    a = base[["signal_key", "return_pct"]].rename(columns={"return_pct": "base_return"})
    b = var[["signal_key", "return_pct"]].rename(columns={"return_pct": "variant_return"})
    z = a.merge(b, on="signal_key", how="inner")
    if z.empty:
        return {"n": 0}
    z["delta"] = z["variant_return"] - z["base_return"]
    return {
        "n": int(len(z)),
        "mean_delta_pct": round(float(z["delta"].mean()), 3),
        "median_delta_pct": round(float(z["delta"].median()), 3),
        "improved_pct": round(float((z["delta"] > 0).mean() * 100), 2),
        "unchanged_pct": round(float((z["delta"].abs() < 1e-12).mean() * 100), 2),
        "worse_pct": round(float((z["delta"] < 0).mean() * 100), 2),
    }


def recovery_stats(base, prepared, ctx_map, spec, delay):
    vals = []
    hit_count = 0
    for r in base.itertuples(index=False):
        if r.exit_reason != "strict_gakutto_next_open" or not r.gakutto_date:
            continue
        d = pd.Timestamp(r.gakutto_date).normalize()
        ctx = ctx_map.get(d)
        if not condition_hit(ctx, spec):
            continue
        hit_count += 1
        g = prepared.get(str(r.code))
        if g is None:
            continue
        gi = g.attrs["date_to_i"].get(d)
        if gi is None:
            continue
        base_exit_i = gi + 1
        later_i = base_exit_i + delay
        if later_i >= len(g):
            continue
        p0 = float(g["O"].iloc[base_exit_i])
        p1 = float(g["O"].iloc[later_i])
        if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
            vals.append((p1 / p0 - 1) * 100)
    s = pd.Series(vals, dtype=float)
    if s.empty:
        return {"broad_gakutto_count": int(hit_count), "n_with_future": 0}
    return {
        "broad_gakutto_count": int(hit_count),
        "n_with_future": int(len(s)),
        "avg_extra_edge_pct": round(float(s.mean()), 3),
        "median_extra_edge_pct": round(float(s.median()), 3),
        "later_open_higher_pct": round(float((s > 0).mean() * 100), 2),
        "later_open_lower_pct": round(float((s < 0).mean() * 100), 2),
    }


def split_metrics(df):
    if df.empty:
        return {"older": {"n": 0}, "recent": {"n": 0}, "yearly": {}}
    z = df.copy()
    z["signal_date_dt"] = pd.to_datetime(z["signal_date"])
    older = z[z["signal_date_dt"] < SPLIT_DATE]
    recent = z[z["signal_date_dt"] >= SPLIT_DATE]
    yearly = {
        str(int(y)): metrics(g)
        for y, g in z.groupby(z["signal_date_dt"].dt.year)
    }
    return {"older": metrics(older), "recent": metrics(recent), "yearly": yearly}


def main():
    signals = load_signals()
    archive = load_archive()
    ctx = build_market_context(archive)
    ctx_map = {
        pd.Timestamp(r.date).normalize(): {
            "topix_ret1_pct": r.topix_ret1_pct,
            "decline_pct": r.decline_pct,
            "advance_pct": r.advance_pct,
            "equal_weight_mean_pct": r.equal_weight_mean_pct,
            "comparable": r.comparable,
        }
        for r in ctx.itertuples(index=False)
    }

    needed = set(signals["code"].astype(str))
    prepared = {}
    for code, g in archive[archive["Code"].isin(needed)].groupby("Code", sort=False):
        prepared[str(code)] = prepare(g)
    del archive

    cases = []
    missing_price = 0
    for row in signals.to_dict("records"):
        g = prepared.get(str(row["code"]))
        if g is None:
            missing_price += 1
            continue
        cases.append((row, g))

    baseline_rows = []
    for sig, g in cases:
        tr = simulate(g, sig, ctx_map)
        if tr:
            for k in [
                "universe_frozen_all", "universe_current_liquid",
                "universe_winner_liquid", "universe_individual_strong_liquid",
            ]:
                tr[k] = bool(sig[k])
            baseline_rows.append(tr)
    baseline = pd.DataFrame(baseline_rows)

    universe_cols = {
        "frozen_all": "universe_frozen_all",
        "current_liquid": "universe_current_liquid",
        "winner_liquid": "universe_winner_liquid",
        "individual_strong_liquid": "universe_individual_strong_liquid",
    }

    summary = {
        "research_question": "When strict gakutto occurs on a broad market selloff day, is delaying the normal next-open exit beneficial?",
        "signal_source": "results/kuitto_runner_features_trades.csv (frozen 10y kuitto signal set)",
        "archive_range": {
            "signal_min": signals["signal_date"].min().strftime("%Y-%m-%d"),
            "signal_max": signals["signal_date"].max().strftime("%Y-%m-%d"),
        },
        "execution": {
            "entry": "next open after signal",
            "stop_loss_pct": STOP_PCT,
            "normal_gakutto_exit": "strict gakutto confirmed at close -> next open",
            "max_hold": "15 full sessions -> following open",
            "rescue_delay_definition": "delay N means exit N extra sessions after the normal next-open exit; -5% stop and max15 remain active during the delay",
            "market_information_timing": "same-day close breadth and TOPIX return; both are known when strict gakutto is confirmed after close",
        },
        "market_conditions": CONDITIONS,
        "main_condition": MAIN_CONDITION,
        "signal_counts": {
            "frozen_all": int(signals["universe_frozen_all"].sum()),
            "current_liquid": int(signals["universe_current_liquid"].sum()),
            "winner_liquid": int(signals["universe_winner_liquid"].sum()),
            "individual_strong_liquid": int(signals["universe_individual_strong_liquid"].sum()),
            "missing_price_groups": int(missing_price),
        },
        "baseline": {},
        "event_recovery": {},
        "variants": {},
        "main_condition_robustness": {},
        "notes": [
            "Current-liquidity replay applies the present 20-day average turnover >= 500m yen rule to all historical signals.",
            "The archive is current-Prime-listing based and therefore has survivorship bias, consistent with the existing 10y research archive.",
            "Market-rescue thresholds and delays were predefined before this run; no threshold was selected after seeing the result.",
        ],
    }

    for uname, ucol in universe_cols.items():
        b = baseline[baseline[ucol]].copy()
        summary["baseline"][uname] = {
            "metrics": metrics(b),
            "exit_reasons": {str(k): int(v) for k, v in b["exit_reason"].value_counts().to_dict().items()},
            "strict_gakutto_count": int((b["exit_reason"] == "strict_gakutto_next_open").sum()),
        }

    # Event-level diagnostic: after a broad-market gakutto, what did later opens do
    # relative to the normal next-open exit?
    for cname, spec in CONDITIONS.items():
        summary["event_recovery"][cname] = {}
        for delay in DELAYS:
            summary["event_recovery"][cname][str(delay)] = recovery_stats(
                baseline[baseline["universe_current_liquid"]].copy(),
                prepared, ctx_map, spec, delay
            )

    variant_rows = []
    all_variant_frames = {}
    for cname, spec in CONDITIONS.items():
        for delay in DELAYS:
            rows = []
            for sig, g in cases:
                tr = simulate(g, sig, ctx_map, rescue_spec=spec, delay=delay)
                if tr:
                    for k in [
                        "universe_frozen_all", "universe_current_liquid",
                        "universe_winner_liquid", "universe_individual_strong_liquid",
                    ]:
                        tr[k] = bool(sig[k])
                    rows.append(tr)
            v = pd.DataFrame(rows)
            all_variant_frames[(cname, delay)] = v
            summary["variants"].setdefault(cname, {})[str(delay)] = {}

            for uname, ucol in universe_cols.items():
                b = baseline[baseline[ucol]].copy()
                p = v[v[ucol]].copy()
                res = {
                    "metrics": metrics(p),
                    "paired_vs_baseline": paired_metrics(b, p),
                    "rescued_trades": int(p["rescued"].sum()) if not p.empty else 0,
                    "exit_reasons": {str(k): int(vv) for k, vv in p["exit_reason"].value_counts().to_dict().items()} if not p.empty else {},
                }
                summary["variants"][cname][str(delay)][uname] = res
                variant_rows.append({
                    "condition": cname,
                    "delay_extra_sessions": delay,
                    "universe": uname,
                    **res["metrics"],
                    **{f"paired_{k}": val for k, val in res["paired_vs_baseline"].items()},
                    "rescued_trades": res["rescued_trades"],
                })

    # Robustness only for the predeclared main condition, to avoid a giant report.
    for delay in DELAYS:
        v = all_variant_frames[(MAIN_CONDITION, delay)]
        summary["main_condition_robustness"][str(delay)] = {}
        for uname, ucol in universe_cols.items():
            b = baseline[baseline[ucol]].copy()
            p = v[v[ucol]].copy()
            ps = split_metrics(p)
            bs = split_metrics(b)

            paired_yearly = {}
            bz = b.copy()
            pz = p.copy()
            bz["year"] = pd.to_datetime(bz["signal_date"]).dt.year
            pz["year"] = pd.to_datetime(pz["signal_date"]).dt.year
            years = sorted(set(bz["year"].dropna().astype(int)) | set(pz["year"].dropna().astype(int)))
            for y in years:
                paired_yearly[str(y)] = paired_metrics(
                    bz[bz["year"] == y],
                    pz[pz["year"] == y]
                )

            summary["main_condition_robustness"][str(delay)][uname] = {
                "variant": ps,
                "baseline": bs,
                "paired_yearly": paired_yearly,
            }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(variant_rows).to_csv(OUT_CSV, index=False)

    print("=== GAKUTTO MARKET RESCUE SUMMARY ===")
    print("signal_counts", summary["signal_counts"])
    print("baseline_current_liquid", summary["baseline"]["current_liquid"])
    print("event_recovery_main", summary["event_recovery"][MAIN_CONDITION])
    print("variants_main_current_liquid")
    for delay in DELAYS:
        print(delay, summary["variants"][MAIN_CONDITION][str(delay)]["current_liquid"])
    print("variants_main_winner_liquid")
    for delay in DELAYS:
        print(delay, summary["variants"][MAIN_CONDITION][str(delay)]["winner_liquid"])


if __name__ == "__main__":
    main()
