from fastapi import APIRouter, Depends, HTTPException, Query
from lib.auth import verify_api_key
from typing import Optional
import math

router = APIRouter()

UNIVERSE = list(set([
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "AMD", "QCOM",
    "TXN", "INTC", "MU", "AMAT", "LRCX", "KLAC", "SNPS", "CDNS", "ADI", "MRVL",
    "CRM", "ADBE", "NOW", "INTU", "PANW", "CRWD", "FTNT", "SNOW", "DDOG", "MDB",
    "NET", "ZS", "OKTA", "ABNB", "UBER", "LYFT", "DASH", "RBLX", "SPOT",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "V", "MA", "PYPL",
    "COF", "USB", "PNC", "TFC", "MET", "PRU", "AFL", "ALL", "CB", "AIG", "MMC",
    "JNJ", "UNH", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY", "AMGN",
    "GILD", "ISRG", "SYK", "BSX", "MDT", "EW", "REGN", "VRTX", "BIIB", "MRNA",
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TJX", "LOW", "TGT", "COST",
    "WMT", "PG", "KO", "PEP", "PM", "MO", "CL", "EL", "ULTA", "LULU", "ROST",
    "CAT", "DE", "HON", "UPS", "FDX", "LMT", "RTX", "BA", "GE", "MMM", "ETN",
    "EMR", "PH", "ROK", "XYL", "IR", "CARR", "OTIS", "GD", "NOC", "LHX",
    "XOM", "CVX", "COP", "SLB", "MPC", "PSX", "VLO", "EOG", "PXD", "DVN", "HES",
    "PLD", "AMT", "EQIX", "CCI", "SPG", "O", "WELL", "DLR", "PSA", "AVB",
    "LIN", "APD", "ECL", "SHW", "FCX", "NEM", "ALB", "MOS", "CF", "NUE",
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "SRE", "PCG", "ED",
    "NFLX", "CMCSA", "T", "VZ", "TMUS", "DIS", "WBD", "PARA",
    "SPY", "QQQ", "IWM", "DIA", "VTI", "GLD", "TLT", "AGG",
]))


def _clean(v):
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return v


@router.get("/screen", dependencies=[Depends(verify_api_key)])
async def screen_universe(
    sector: Optional[str] = Query(None),
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
    """Screen equity universe with Polygon data."""
    try:
        from lib.polygon_client import get_batch_snapshots, get_ticker_details, get_parsed_financials

        tickers = UNIVERSE.copy()

        if search:
            s = search.upper()
            tickers = [t for t in tickers if s in t]
            if s not in tickers:
                tickers.insert(0, s)

        # Fetch all price snapshots in one batch call
        snapshots = {}
        try:
            snapshots = get_batch_snapshots(tickers[:100])
        except Exception:
            pass

        needs_fundamentals = any([min_pe, max_pe, min_revenue_growth, min_profit_margin])

        results = []
        for ticker in tickers[:100]:
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
            ticker_sector = details.get("sector", "")

            if min_market_cap and (mkt_cap or 0) < min_market_cap:
                continue
            if max_market_cap and (mkt_cap or float("inf")) > max_market_cap:
                continue
            if sector and sector.lower() not in ticker_sector.lower():
                continue

            pe = None
            rev_growth = None
            profit_margin = None

            if needs_fundamentals:
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
                "name": details.get("name", ticker),
                "sector": ticker_sector,
                "price": _clean(price),
                "day_change_pct": _clean(day_chg),
                "volume": _clean(volume),
                "market_cap": _clean(mkt_cap),
                "pe_ratio": _clean(pe),
                "revenue_growth": _clean(rev_growth),
                "profit_margin": _clean(profit_margin),
            })

        reverse = sort_dir == "desc"
        results.sort(
            key=lambda x: x.get(sort_by) or (0 if reverse else float("inf")),
            reverse=reverse,
        )

        return {
            "count": len(results[:limit]),
            "total_universe": len(UNIVERSE),
            "results": results[:limit],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
