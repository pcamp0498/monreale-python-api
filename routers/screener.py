from fastapi import APIRouter, Depends, HTTPException, Query
from lib.auth import verify_api_key
from typing import Optional
import math
import time
import threading

router = APIRouter()

FALLBACK_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "AMD", "QCOM",
    "TXN", "INTC", "MU", "CRM", "ADBE", "NOW", "INTU", "PANW", "CRWD", "SNOW",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "V", "MA", "PYPL",
    "JNJ", "UNH", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY", "AMGN",
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TJX", "LOW", "TGT", "COST",
    "WMT", "PG", "KO", "PEP", "PM", "MO", "XOM", "CVX", "COP", "SLB",
    "CAT", "DE", "HON", "UPS", "FDX", "LMT", "RTX", "BA", "GE", "MMM",
    "PLD", "AMT", "EQIX", "NEE", "DUK", "NFLX", "CMCSA", "T", "VZ", "TMUS",
    "SPY", "QQQ", "IWM", "DIA", "GLD", "TLT", "AGG", "VTI",
]

_universe_cache: list = []
_universe_cache_time: float = 0
_CACHE_TTL = 86400


def _get_cached_universe() -> list:
    global _universe_cache, _universe_cache_time
    now = time.time()
    if _universe_cache and (now - _universe_cache_time) < _CACHE_TTL:
        return _universe_cache
    try:
        from lib.polygon_client import get_full_universe
        tickers = get_full_universe(limit=1000)
        print(f"[cache] get_full_universe returned {len(tickers)} items")
        if tickers and len(tickers) > 10:
            _universe_cache = [t["ticker"] for t in tickers if t.get("ticker")]
            _universe_cache_time = now
            print(f"[cache] Cached {len(_universe_cache)} tickers")
            return _universe_cache
    except Exception as e:
        print(f"[cache] Error: {e}")
    return FALLBACK_UNIVERSE


def _clean(v):
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return v


def _safe_str(v) -> str:
    """Safely convert to stripped string, never None."""
    if v is None:
        return ""
    return str(v).strip()


MKT_CAP_BUCKETS = {
    "mega": lambda mc: mc >= 200_000_000_000,
    "large": lambda mc: 10_000_000_000 <= mc < 200_000_000_000,
    "mid": lambda mc: 2_000_000_000 <= mc < 10_000_000_000,
    "small": lambda mc: 300_000_000 <= mc < 2_000_000_000,
    "micro": lambda mc: mc < 300_000_000,
}


# Preload universe in background on import
def _preload():
    try:
        _get_cached_universe()
        print("[screener] Universe preloaded")
    except Exception as e:
        print(f"[screener] Preload failed: {e}")

threading.Thread(target=_preload, daemon=True).start()


@router.get("/screen", dependencies=[Depends(verify_api_key)])
async def screen_universe(
    sector: Optional[str] = Query(None),
    industry: Optional[str] = Query(None),
    market_cap_bucket: Optional[str] = Query(None),
    min_market_cap: Optional[float] = Query(None),
    max_market_cap: Optional[float] = Query(None),
    min_pe: Optional[float] = Query(None),
    max_pe: Optional[float] = Query(None),
    min_revenue_growth: Optional[float] = Query(None),
    min_profit_margin: Optional[float] = Query(None),
    min_day_change: Optional[float] = Query(None),
    max_day_change: Optional[float] = Query(None),
    sort_by: str = Query("market_cap"),
    sort_dir: str = Query("desc"),
    limit: int = Query(50),
    search: Optional[str] = Query(None),
):
    """Screen equity universe using Polygon data."""
    try:
        from lib.polygon_client import get_batch_snapshots, get_ticker_details, get_parsed_financials

        tickers = _get_cached_universe()
        total_universe = len(tickers)

        if search:
            s = search.upper()
            tickers = [t for t in tickers if s in t]
            if s not in tickers:
                tickers.insert(0, s)

        snapshots = {}
        try:
            snapshots = get_batch_snapshots(tickers[:200])
        except Exception:
            pass

        # Clean filter inputs
        sector_filter = _safe_str(sector).lower() if sector else ""
        industry_filter = _safe_str(industry).lower() if industry else ""

        results = []
        for ticker in tickers[:200]:
            try:
                snap = snapshots.get(ticker, {})
                price = snap.get("price")
                day_chg = snap.get("day_change_pct") or snap.get("change_pct")
                volume = snap.get("volume")

                if min_day_change is not None and (day_chg or 0) < min_day_change:
                    continue
                if max_day_change is not None and (day_chg or 0) > max_day_change:
                    continue

                details = {}
                try:
                    details = get_ticker_details(ticker)
                except Exception:
                    pass

                mkt_cap = details.get("market_cap")
                ticker_sector = _safe_str(details.get("sector"))
                ticker_name = _safe_str(details.get("name")) or ticker

                if min_market_cap and (mkt_cap or 0) < min_market_cap:
                    continue
                if max_market_cap and (mkt_cap or float("inf")) > max_market_cap:
                    continue
                if market_cap_bucket and market_cap_bucket in MKT_CAP_BUCKETS:
                    if not MKT_CAP_BUCKETS[market_cap_bucket](mkt_cap or 0):
                        continue

                if sector_filter and sector_filter not in ticker_sector.lower():
                    continue
                if industry_filter and industry_filter not in ticker_sector.lower():
                    continue

                pe = None
                rev_growth = None
                profit_margin = None

                try:
                    fins = get_parsed_financials(ticker, limit=2)
                    if fins:
                        curr = fins[0]
                        prev = fins[1] if len(fins) > 1 else {}
                        revenue = curr.get("revenue")
                        prev_rev = prev.get("revenue")
                        net_income = curr.get("net_income")
                        eps = curr.get("eps_diluted")
                        if price and eps and eps > 0:
                            pe = price / eps
                        if revenue and prev_rev and prev_rev > 0:
                            rev_growth = (revenue - prev_rev) / abs(prev_rev)
                        if net_income and revenue and revenue > 0:
                            profit_margin = net_income / revenue
                except Exception:
                    pass

                if pe is None and price:
                    try:
                        import simfin as sf
                        import os
                        sf.set_api_key(os.environ.get("SIMFIN_API_KEY", "free"))
                        sf.set_data_dir("/tmp/simfin_data")
                        income = sf.load_income(variant="annual", market="us")
                        if income is not None and not income.empty and ticker in income.index.get_level_values(0):
                            df = income.loc[ticker].sort_index(ascending=False)
                            if len(df) >= 1:
                                c = df.iloc[0]
                                p = df.iloc[1] if len(df) > 1 else None
                                eps_sf = float(c.get("Diluted EPS", 0) or 0)
                                rev_sf = float(c.get("Revenue", 0) or 0)
                                net_sf = float(c.get("Net Income", 0) or 0)
                                prev_rev_sf = float(p.get("Revenue", 0) or 0) if p is not None else 0
                                if eps_sf > 0 and pe is None:
                                    pe = price / eps_sf
                                if rev_sf and prev_rev_sf and prev_rev_sf != 0 and rev_growth is None:
                                    rev_growth = (rev_sf - prev_rev_sf) / abs(prev_rev_sf)
                                if net_sf and rev_sf and rev_sf != 0 and profit_margin is None:
                                    profit_margin = net_sf / rev_sf
                    except Exception:
                        pass

                if min_pe and (pe or 0) < min_pe:
                    continue
                if max_pe and pe and pe > max_pe:
                    continue
                if min_revenue_growth and (rev_growth or 0) < min_revenue_growth / 100:
                    continue
                if min_profit_margin and (profit_margin or 0) < min_profit_margin / 100:
                    continue

                results.append({
                    "ticker": ticker,
                    "name": ticker_name,
                    "sector": ticker_sector,
                    "price": _clean(price),
                    "day_change_pct": _clean(day_chg),
                    "volume": _clean(volume),
                    "market_cap": _clean(mkt_cap),
                    "pe_ratio": _clean(pe),
                    "revenue_growth": _clean(rev_growth),
                    "profit_margin": _clean(profit_margin),
                })

            except Exception as e:
                print(f"[screen] Skip {ticker}: {e}")
                continue

        reverse = sort_dir == "desc"
        results.sort(
            key=lambda x: x.get(sort_by) or (0 if reverse else float("inf")),
            reverse=reverse,
        )

        return {
            "count": len(results[:limit]),
            "total_universe": total_universe,
            "results": results[:limit],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sectors", dependencies=[Depends(verify_api_key)])
async def get_sectors():
    try:
        from lib.polygon_client import get_ticker_details
        universe = _get_cached_universe()[:300]
        sectors: set = set()
        for ticker in universe[:100]:
            try:
                details = get_ticker_details(ticker)
                s = _safe_str(details.get("sector"))
                if s:
                    sectors.add(s)
            except Exception:
                continue
        return {"sectors": sorted(sectors)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/debug-universe", dependencies=[Depends(verify_api_key)])
async def debug_universe():
    import os
    from lib.polygon_client import get_full_universe
    has_key = bool(os.environ.get("POLYGON_API_KEY"))
    try:
        raw = get_full_universe(limit=10)
        cached = _get_cached_universe()
        return {
            "has_polygon_key": has_key,
            "raw_returned": len(raw),
            "raw_sample": [t.get("ticker") if isinstance(t, dict) else t for t in raw[:5]],
            "cache_size": len(cached),
            "cache_sample": cached[:5],
            "using_fallback": len(cached) == len(FALLBACK_UNIVERSE),
        }
    except Exception as e:
        return {"has_polygon_key": has_key, "error": str(e)}
