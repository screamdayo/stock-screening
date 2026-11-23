"""Track how a monthly Momentum6M report actually turned out (priority 5 names).

Reads the saved report from the private repo (mounted at .private-data/results/momentum6m_signals/YYYY-MM.json)
and daily bars (data/batches + data/daily_updates, raw yen prices). For each priority name:
  entry  session after the signal date; the report's own buy limit is applied (fills at the open if the open is at
         or below it, at the limit if the low reaches it, otherwise 'unfilled'); the open-buy result is shown too
  exit   first open after the next calendar month-end; until then the latest close is used (status 'holding')
Writes .momentum6m/track.json (copied to the private repo by the workflow). Public logs show only the average.
With --notify the result is sent to Discord, but only once the holding period is over (complete=True).
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from price_adjust import normalize_code

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / ".private-data/results/momentum6m_signals"
OUT = ROOT / ".momentum6m/track.json"
COST = 0.003


def load_bars(codes, since):
    files = sorted((DATA / "batches").glob("*.parquet")) + sorted((DATA / "daily_updates").glob("*.parquet"))
    parts = []
    for f in files:
        d = pd.read_parquet(f, columns=["Date", "Code", "O", "L", "C", "AdjFactor"])
        d["Code"] = d.Code.astype(str).map(normalize_code)
        d["Date"] = pd.to_datetime(d.Date)
        parts.append(d[d.Code.isin(codes) & (d.Date >= since)])
    a = pd.concat(parts, ignore_index=True)
    for c in ("O", "L", "C", "AdjFactor"):
        a[c] = pd.to_numeric(a[c], errors="coerce")
    return a.sort_values(["Code", "Date"]).drop_duplicates(["Code", "Date"], keep="last")


def track(report, bars, all_dates):
    sig = pd.Timestamp(report["signal_date"])
    after = all_dates[all_dates > sig]
    if after.empty:
        return None
    entry_day = after[0]
    month_end = (sig.to_period("M") + 1).end_time.normalize()  # end of the holding (next) month
    exits = all_dates[all_dates > month_end]
    exit_day = exits[0] if len(exits) else None
    last_day = all_dates[-1]
    rows = []
    picks = [c for c in report["candidates"] if c.get("priority_slot")] or report["candidates"][:5]
    for c in picks:
        b = bars[bars.Code == normalize_code(str(c["code"]))].set_index("Date")
        r = {"rank": c.get("rank"), "code": c["code"], "name": c.get("name"),
             "signal_close_yen": c.get("signal_close_yen"), "buy_limit_yen": c.get("buy_limit_yen"),
             "shares_in_report": c.get("shares_at_200k_cap")}
        if entry_day not in b.index:
            r["status"] = "no_price_on_entry_day"
            rows.append(r)
            continue
        e = b.loc[entry_day]
        lim = float(c.get("buy_limit_yen") or c["signal_close_yen"])
        if e.O <= lim:
            fill = e.O
        elif e.L <= lim:
            fill = lim
        else:
            fill = None
        held = b.loc[entry_day:]
        split = bool((held.AdjFactor.fillna(1.0) != 1.0).any())
        if exit_day is not None and exit_day in b.index:
            out, status, out_day = b.loc[exit_day].O, "closed", exit_day
        else:
            out_day = held.C.last_valid_index()
            out, status = held.C.loc[out_day], ("holding" if exit_day is None else "stopped_trading")
        r.update({"entry_date": str(entry_day.date()), "entry_open_yen": float(e.O), "opening_gap_pct": round((e.O / c["signal_close_yen"] - 1) * 100, 2),
                  "filled_by_limit": fill is not None, "fill_yen": float(fill) if fill is not None else None,
                  "valued_at": str(out_day.date()), "value_yen": float(out), "status": status,
                  "return_limit_pct": round((out / fill - 1 - COST) * 100, 2) if fill is not None else None,
                  "return_open_buy_pct": round((out / e.O - 1 - COST) * 100, 2),
                  "split_during_hold": split})
        rows.append(r)
    lim_rets = [x["return_limit_pct"] if x.get("filled_by_limit") else 0.0 for x in rows if "return_open_buy_pct" in x]
    open_rets = [x["return_open_buy_pct"] for x in rows if "return_open_buy_pct" in x]
    return {"signal_month": report["signal_month"], "signal_date": report["signal_date"],
            "entry_date": str(entry_day.date()), "planned_exit_date": str(exit_day.date()) if exit_day is not None else None,
            "data_through": str(last_day.date()), "complete": exit_day is not None,
            "portfolio_return_limit_pct": round(float(np.mean(lim_rets)), 2) if lim_rets else None,
            "portfolio_return_open_buy_pct": round(float(np.mean(open_rets)), 2) if open_rets else None,
            "note": "equal weight per slot, unfilled slot = cash, 0.3% round-trip cost; ignores the 200k share cap",
            "picks": rows, "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z"}


def tests():
    dates = pd.DatetimeIndex(pd.to_datetime(["2026-10-30", "2026-11-02", "2026-11-04", "2026-11-30", "2026-12-01"]))
    bars = pd.DataFrame({"Date": dates.repeat(2), "Code": ["1111", "2222"] * 5,
                         "O": [100, 50, 101, 56, 105, 57, 110, 60, 120, 62], "L": [99, 49, 99.5, 55.5, 104, 56, 108, 59, 118, 61],
                         "C": [100, 50, 103, 57, 106, 58, 111, 61, 121, 63], "AdjFactor": 1.0})
    rep = {"signal_month": "2026-10", "signal_date": "2026-10-30",
           "candidates": [{"rank": 1, "code": "1111", "signal_close_yen": 100, "buy_limit_yen": 100, "priority_slot": True},
                          {"rank": 2, "code": "2222", "signal_close_yen": 50, "buy_limit_yen": 55, "priority_slot": True}]}
    t = track(rep, bars, dates)
    a, b = t["picks"]
    assert a["filled_by_limit"] and a["fill_yen"] == 100 and a["value_yen"] == 120 and a["status"] == "closed"
    assert not b["filled_by_limit"] and b["return_open_buy_pct"] == round((62 / 56 - 1 - COST) * 100, 2)
    t2 = track(rep, bars[bars.Date <= "2026-11-30"], dates[:4])
    assert t2["picks"][0]["status"] == "holding" and t2["picks"][0]["value_yen"] == 111 and not t2["complete"]
    print("entry fill, exit timing and holding valuation tests passed", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--signal-month", required=True)
    p.add_argument("--notify", action="store_true", help="send the result to Discord if it is complete")
    p.add_argument("--dry-run", action="store_true", help="print the Discord message instead of sending")
    args = p.parse_args()
    tests()
    report = json.loads((REPORTS / f"{args.signal_month}.json").read_text(encoding="utf-8"))
    codes = {normalize_code(str(c["code"])) for c in report["candidates"]}
    since = pd.Timestamp(report["signal_date"]) - pd.Timedelta(days=7)
    bars = load_bars(codes, since)
    all_dates = pd.DatetimeIndex(sorted(bars.Date.unique()))
    res = track(report, bars, all_dates)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("signal_month", "data_through", "complete", "portfolio_return_limit_pct",
                                          "portfolio_return_open_buy_pct")}, ensure_ascii=False), flush=True)
    if (args.notify or args.dry_run) and res["complete"]:
        import sys
        sys.path.insert(0, str(ROOT))
        import notifier
        if args.dry_run:
            notifier._post_long = lambda msg: print(msg)
        notifier.notify_momentum6m_result(res)
    elif args.notify:
        print("holding period not over yet: no notification", flush=True)


if __name__ == "__main__":
    main()
