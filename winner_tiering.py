"""本命/強本命/超強本命の本番ティア付与。バックテストで固定した条件のみ使用する。"""
from datetime import timedelta
import pandas as pd
import download
from logger import get_logger

logger=get_logger(__name__)

WINNER_ATR_MIN=3.0
WINNER_DD20_MAX=-5.5
STRONG_MA25_SLOPE5_MIN=-1.5
TOPIX_BAD_SLOPE_LO=-1.0
TOPIX_BAD_SLOPE_HI=-0.5
TOPIX_BAD_VSMA_LO=-1.0
TOPIX_BAD_VSMA_HI=0.0
REL20_BAD_LO=0.0
REL20_BAD_HI=4.0

def _fetch_topix_context(latest_date):
    start=(pd.Timestamp(latest_date)-timedelta(days=100)).strftime("%Y-%m-%d")
    end=pd.Timestamp(latest_date).strftime("%Y-%m-%d")
    res=download._request_with_retry(f"{download.BASE_URL}/indices/bars/daily/topix",params={"from":start,"to":end})
    res.raise_for_status()
    d=pd.DataFrame(res.json().get("data",[]))
    if d.empty: raise RuntimeError("TOPIX日足が取得できませんでした")
    d["Date"]=pd.to_datetime(d["Date"])
    d["C"]=pd.to_numeric(d["C"],errors="coerce")
    d=d.dropna(subset=["Date","C"]).sort_values("Date").reset_index(drop=True)
    d["RET20"]=(d["C"]/d["C"].shift(20)-1)*100
    d["MA20"]=d["C"].rolling(20).mean()
    d["VSMA20"]=(d["C"]/d["MA20"]-1)*100
    d["MA20_SLOPE5"]=(d["MA20"]/d["MA20"].shift(5)-1)*100
    prev=d[d["Date"]<pd.Timestamp(latest_date)].tail(1)
    if prev.empty: raise RuntimeError("TOPIX前日データが不足しています")
    r=prev.iloc[0]
    return {
      "date":pd.Timestamp(r["Date"]),
      "ret20_pct":float(r["RET20"]),
      "vs_ma20_pct":float(r["VSMA20"]),
      "ma20_slope5_pct":float(r["MA20_SLOPE5"]),
    }

def apply_tiers(results,price_df):
    if not results:return results
    latest=pd.Timestamp(price_df["Date"].max()).normalize()
    try:
        tx=_fetch_topix_context(latest)
    except Exception as e:
        logger.warning("超強本命のTOPIX判定をスキップ: %s",e)
        tx=None

    by={}
    for code,g in price_df.groupby("Code"):
        g=g.sort_values("Date").copy()
        g["C"]=pd.to_numeric(g["C"],errors="coerce")
        g["MA25"]=g["C"].rolling(25).mean()
        if len(g)<26: continue
        r=g.iloc[-1]
        ma25_5ago=g["MA25"].iloc[-6] if len(g)>=31 else pd.NA
        ret20=(r["C"]/g["C"].iloc[-21]-1)*100 if len(g)>=21 and pd.notna(g["C"].iloc[-21]) else None
        slope=(r["MA25"]/ma25_5ago-1)*100 if pd.notna(r["MA25"]) and pd.notna(ma25_5ago) and ma25_5ago else None
        by[str(code)]={"ret20_pct":float(ret20) if ret20 is not None else None,"ma25_slope5_pct":float(slope) if slope is not None else None}

    for r in results:
        f=by.get(str(r.get("code")),{})
        atr=r.get("atr14_pct"); dd=r.get("dd20_pct")
        winner=bool(atr is not None and dd is not None and float(atr)>=WINNER_ATR_MIN and float(dd)<=WINNER_DD20_MAX)
        slope=f.get("ma25_slope5_pct")
        strong=bool(winner and slope is not None and slope>=STRONG_MA25_SLOPE5_MIN)
        ultra=False; rel20=None
        if strong and tx and f.get("ret20_pct") is not None:
            rel20=f["ret20_pct"]-tx["ret20_pct"]
            bad_market=(TOPIX_BAD_SLOPE_LO<=tx["ma20_slope5_pct"]<=TOPIX_BAD_SLOPE_HI) or (TOPIX_BAD_VSMA_LO<=tx["vs_ma20_pct"]<=TOPIX_BAD_VSMA_HI)
            bad_rel=(REL20_BAD_LO<rel20<=REL20_BAD_HI)
            ultra=not bad_market and not bad_rel
        r["winner_filter"]=winner
        r["strong_winner"]=strong
        r["ultra_winner"]=ultra
        r["ma25_slope5_pct"]=round(slope,3) if slope is not None else None
        r["ret20_pct"]=round(f["ret20_pct"],3) if f.get("ret20_pct") is not None else None
        r["rel20_vs_topix_pct"]=round(rel20,3) if rel20 is not None else None
        if tx:
            r["topix_prev_date"]=tx["date"].strftime("%Y-%m-%d")
            r["topix_prev_ret20_pct"]=round(tx["ret20_pct"],3)
            r["topix_prev_vs_ma20_pct"]=round(tx["vs_ma20_pct"],3)
            r["topix_prev_ma20_slope5_pct"]=round(tx["ma20_slope5_pct"],3)
    logger.info("本命ティア: 本命%d / 強本命%d / 超強本命%d",
                sum(bool(r.get("winner_filter")) for r in results),
                sum(bool(r.get("strong_winner")) for r in results),
                sum(bool(r.get("ultra_winner")) for r in results))
    return results
