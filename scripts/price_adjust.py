"""Shared helpers for joining data/batches (10Y archive) with data/daily_updates.

Why this exists
- J-Quants adjusted prices (AdjO/AdjH/AdjL/AdjC/AdjVo) are adjusted as of the day they were fetched.
  The archive was fetched once; a split that goes ex afterwards makes later rows use a new basis while
  the archived rows stay on the old one, so a naive join shows a fake price step on the ex-date
  (first seen on 2026-09-29: about 40 Prime names, e.g. 8035 looked like -80% in one day).
- Raw prices and AdjFactor do not depend on the fetch date, so they tell us which rows are stale:
  the correct multiplier for a row is the product of AdjFactor over all later dates, and the
  multiplier the vendor actually applied is AdjC / C. Where the two disagree the row is rescaled;
  every other row keeps the vendor's adjusted values untouched.
- daily_updates also contains 5-character codes that are not "common stock + 0" (e.g. 94345).
  Truncating codes to 4 characters merges them into the common stock; normalize_code() does not.
"""
import numpy as np
import pandas as pd

STALE_TOLERANCE = 0.02  # vendor rounding on low-priced names stays below ~1.1%; real splits are far larger


def normalize_code(code):
    """'72030' -> '7203'; anything else (4-char codes, '94345', '130A') is returned unchanged."""
    code = str(code)
    if len(code) == 5 and code.endswith("0"):
        return code[:-1]
    return code


def cumulative_factor(df, code_col="Code", date_col="Date", factor_col="AdjFactor"):
    """Product of AdjFactor over strictly later dates of the same code (1.0 on the newest row).

    AdjFactor sits on the ex-date row (0.5 for a 1:2 split) and applies to every earlier date.
    Rows must be unique per (code, date). Returns a Series aligned to df.index.
    """
    f = pd.to_numeric(df[factor_col], errors="coerce").fillna(1.0)
    f = f.where(f > 0, 1.0)
    order = df.assign(_f=f).sort_values([code_col, date_col], kind="mergesort")
    rev = order["_f"].iloc[::-1]
    cum = rev.groupby(order[code_col].iloc[::-1], sort=False).cumprod().iloc[::-1] / order["_f"]
    return cum.reindex(df.index)


def stale_basis_multiplier(df, raw_col="C", adj_col="AdjC", code_col="Code", date_col="Date",
                           factor_col="AdjFactor"):
    """Multiplier that moves stale adjusted prices onto the newest basis (1.0 for consistent rows).

    Use on the joined archive + update rows of one or many codes:
        m = stale_basis_multiplier(df)
        adjusted price *= m ; adjusted volume /= m
    """
    true_cum = cumulative_factor(df, code_col, date_col, factor_col)
    raw = pd.to_numeric(df[raw_col], errors="coerce")
    adj = pd.to_numeric(df[adj_col], errors="coerce")
    applied = (adj / raw).where((raw > 0) & (adj > 0))
    m = (true_cum / applied).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    m = m.where((m - 1).abs() > STALE_TOLERANCE, 1.0)
    stale = m.ne(1.0)
    if not stale.any():
        return m
    # AdjC / C carries the vendor's rounding, so snap each stale row to the exact product of the
    # most recent split factors it is missing (keeps percentage returns inside a code unchanged).
    f = pd.to_numeric(df[factor_col], errors="coerce").fillna(1.0)
    for code in df.loc[stale, code_col].unique():
        rows = df[code_col].eq(code)
        ev = f[rows & f.ne(1.0) & (f > 0)]
        if ev.empty:
            m[rows & stale] = 1.0
            continue
        ev = ev.loc[df.loc[ev.index, date_col].sort_values(ascending=False).index]
        cands = np.cumprod(ev.to_numpy())
        idx = m.index[rows & stale]
        noisy = m.loc[idx].to_numpy()
        pick = np.abs(np.log(noisy)[:, None] - np.log(cands)[None, :]).argmin(axis=1)
        m.loc[idx] = cands[pick]
    return m
