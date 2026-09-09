import glob, json, os
from collections import defaultdict

import pandas as pd
import requests

SL = 5.0
HOLD = 15
GAP_MAX = 0.5
TOP_N = 5
BASE_URL = "https://api.jquants.com/v2"


def normalize_code(code):
    code = str(code).strip().upper()
    if len(code) == 5 and code.endswith("0"):
        return code[:-1]
    return code


def load_sector_map():
    api_key = os.environ.get("JQUANTS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("JQUANTS_API_KEY is required")
    res = requests.get(
        f"{BASE_URL}/equities/master",
        headers={"x-api-key": api_key},
        timeout=30,
    )
    res.raise_for_status()
    data = res.json().get("data", [])
    df = pd.DataFrame(data)
    if df.empty:
        raise RuntimeError("equities/master returned no data")

    sector_col = None
    for candidate in ["S33Nm", "Sect33Nm", "Sector33Nm", "S17Nm", "Sect17Nm", "Sector17Nm"]:
        if candidate in df.columns:
            sector_col = candidate
            break
    if sector_col is None:
        raise RuntimeError(f"No sector column found. columns={list(df.columns)}")

    if "MktNm" in df.columns:
        df = df[df["MktNm"] == "プライム"].copy()
    elif "Mkt" in df.columns:
        # Keep all if market naming differs; price universe will still limit tested codes.
        df = df.copy()

    df["CodeNorm"] = df["Code"].map(normalize_code)
    df[sector_col] = df[sector_col].fillna("不明").astype(str)
    return dict(zip(df["CodeNorm"], df[sector_col])), sector_col


def load_prices():
    frames = []
    for path in glob.glob("docs/prices/*.json"):
        code = os.path.basename(path).replace(".json", "")
        if code.startswith("_"):
            continue
        try:
            rows = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        if not rows:
            continue
        df = pd.DataFrame(rows)
        if not {"t", "o", "h", "l", "c"}.issubset(df.columns):
            continue
        ren = {"t": "Date", "o": "O", "h": "H", "l": "L", "c": "C", "v": "Vo"}
        df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
        if "Vo" not in df.columns:
            df["Vo"] = pd.NA
        df["Code"] = normalize_code(code)
        frames.append(df[["Code", "Date", "O", "H", "L", "C", "Vo"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def entry_signal(g, i):
    r = g.iloc[i]
    if pd.isna(r.MA5) or pd.isna(r.MA25) or r.O <= 0:
        return None
    bull = (r.C / r.O - 1) * 100
    if not (1.5 <= bull <= 3.5):
        return None
    ma_gap = (r.MA5 / r.MA25 - 1) * 100
    if not (-5.0 <= ma_gap <= 0.0):
        return None
    base = i - 1
    past = base - 5
    if past < 0 or pd.isna(g.MA5.iloc[past]) or not g.MA5.iloc[past] > 0:
        return None
    decline = (g.MA5.iloc[base] / g.MA5.iloc[past] - 1) * 100
    if not (-5.5 <= decline <= -2.0):
        return None
    if any(g.MA5.iloc[j] > g.MA5.iloc[j - 1] for j in range(i - 2, i)):
        return None
    if not g.MA5.iloc[i] > g.MA5.iloc[i - 1]:
        return None
    if (r.C / r.MA5 - 1) * 100 > 4.0:
        return None
    if i < 1 or pd.isna(r.Vo) or pd.isna(g.Vo.iloc[i - 1]) or not float(g.Vo.iloc[i - 1]) > 0:
        return None
    volume_ratio = float(r.Vo) / float(g.Vo.iloc[i - 1])
    if volume_ratio < 1.25:
        return None
    return {"ma25_gap": ma_gap, "volume_ratio": volume_ratio}


def strict_gakutto(g, j):
    if j < 3 or pd.isna(g.MA5.iloc[j]) or pd.isna(g.MA5.iloc[j - 1]):
        return False
    if not g.MA5.iloc[j] < g.MA5.iloc[j - 1]:
        return False
    if g.MA5.iloc[j - 1] < g.MA5.iloc[j - 2]:
        return False
    if g.MA5.iloc[j - 2] < g.MA5.iloc[j - 3]:
        return False
    return float(g.C.iloc[j]) < float(g.O.iloc[j])


def simulate(g, i):
    ent = i + 1
    if ent >= len(g):
        return None
    entry = float(g.O.iloc[ent])
    if not entry > 0:
        return None
    sl = entry * (1 - SL / 100)
    end = min(ent + HOLD - 1, len(g) - 1)
    pending = False
    for j in range(ent, end + 1):
        if pending:
            px = float(g.O.iloc[j])
            return (px - entry) / entry * 100, j - ent + 1
        if float(g.L.iloc[j]) <= sl:
            return -SL, j - ent + 1
        if j < end and strict_gakutto(g, j):
            pending = True
    if end + 1 < len(g):
        px = float(g.O.iloc[end + 1])
        return (px - entry) / entry * 100, end - ent + 2
    px = float(g.C.iloc[end])
    return (px - entry) / entry * 100, end - ent + 1


def stats(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return {"count": 0}
    s = df.pnl.astype(float)
    pos = s[s > 0]
    neg = s[s <= 0]
    gl = -neg.sum()
    pf = pos.sum() / gl if gl > 0 else None
    return {
        "count": len(df),
        "win_rate": round((s > 0).mean() * 100, 2),
        "avg_pnl": round(s.mean(), 3),
        "median_pnl": round(s.median(), 3),
        "pf": round(pf, 3) if pf is not None else None,
        "avg_hold": round(df.hold.mean(), 2),
        "signal_days": int(df.date.nunique()),
    }


def three_way(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    df["date_dt"] = pd.to_datetime(df["date"])
    dates = sorted(df.date_dt.dropna().unique())
    if len(dates) < 3:
        return [stats(df)]
    c1 = dates[len(dates) // 3]
    c2 = dates[(2 * len(dates)) // 3]
    return [
        stats(df[df.date_dt < c1]),
        stats(df[(df.date_dt >= c1) & (df.date_dt < c2)]),
        stats(df[df.date_dt >= c2]),
    ]


def select_day(day_rows, sector_limit):
    ordered = sorted(day_rows, key=lambda r: abs(r["ma25_gap"]))
    picked = []
    per_sector = defaultdict(int)
    for r in ordered:
        if len(picked) >= TOP_N:
            break
        if sector_limit is not None and per_sector[r["sector"]] >= sector_limit:
            continue
        picked.append(r)
        per_sector[r["sector"]] += 1
    return picked


def main():
    sector_map, sector_col = load_sector_map()
    prices = load_prices()
    candidates = []

    for code, g in prices.groupby("Code"):
        g = g.sort_values("Date").reset_index(drop=True).copy()
        for c in ["O", "H", "L", "C", "Vo"]:
            g[c] = pd.to_numeric(g[c], errors="coerce")
        g["MA5"] = g.C.rolling(5).mean()
        g["MA25"] = g.C.rolling(25).mean()
        for i in range(25, len(g) - 1):
            feat = entry_signal(g, i)
            if not feat:
                continue
            prev_close = float(g.C.iloc[i])
            next_open = float(g.O.iloc[i + 1])
            gap = (next_open / prev_close - 1) * 100 if prev_close > 0 else None
            if gap is None or gap > GAP_MAX:
                continue
            sim = simulate(g, i)
            if not sim:
                continue
            pnl, hold = sim
            candidates.append({
                "code": code,
                "date": str(g.Date.iloc[i]),
                "gap": gap,
                "pnl": pnl,
                "hold": hold,
                "ma25_gap": feat["ma25_gap"],
                "sector": sector_map.get(code, "不明"),
            })

    by_day = defaultdict(list)
    for r in candidates:
        by_day[r["date"]].append(r)

    limits = {"unlimited": None, "max3": 3, "max2": 2, "max1": 1}
    result = {
        "rule": "gap<=+0.5% / top5 by MA25 proximity / SL -5% / strict gakutto next open / max15 next open",
        "sector_column": sector_col,
        "candidate_count_before_top5": len(candidates),
        "candidate_days": len(by_day),
        "modes": {},
    }

    for name, limit in limits.items():
        selected = []
        for date, day_rows in by_day.items():
            selected.extend(select_day(day_rows, limit))
        result["modes"][name] = {
            "all": stats(selected),
            "three_way": three_way(selected),
        }

    print("KUITTO_SECTOR_ANALYSIS=" + json.dumps(result, ensure_ascii=False))
    os.makedirs("output", exist_ok=True)
    json.dump(result, open("output/kuitto_sector_analysis.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
