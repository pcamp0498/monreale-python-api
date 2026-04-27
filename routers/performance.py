"""Performance attribution endpoints (Sprint 9A).

All math runs on this server. Inputs are plain dicts the Next.js layer
fetched from Supabase under the user's RLS context.
"""
from fastapi import APIRouter, Depends, HTTPException

from lib.auth import verify_api_key
from lib import performance_math as pm

router = APIRouter()


@router.post("/calculate", dependencies=[Depends(verify_api_key)])
async def calculate_performance(body: dict):
    """Headline stats: TWR, MWR, alpha, beta, Sharpe, Sortino, max DD, win rate, etc."""
    try:
        trades = body.get("trades") or []
        dividends = body.get("dividends") or []
        benchmark = body.get("benchmark_ticker") or "SPY"
        rf_rate = float(body.get("risk_free_rate") or 0.04)
        if not trades:
            raise HTTPException(status_code=400, detail="trades is required and must be non-empty")
        return pm.compute_headline_stats(trades, dividends, benchmark_ticker=benchmark, rf_rate=rf_rate)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/closed-positions", dependencies=[Depends(verify_api_key)])
async def closed_positions(body: dict):
    """FIFO-matched closed positions with entry/exit dates and tax classification."""
    try:
        trades = body.get("trades") or []
        if not trades:
            raise HTTPException(status_code=400, detail="trades is required and must be non-empty")
        closed = pm.match_fifo_lots(trades)
        # Sort by exit_date desc by default; client can re-sort
        closed.sort(key=lambda c: c.get("exit_date", ""), reverse=True)
        return {
            "closed_positions": closed,
            "count": len(closed),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/buckets", dependencies=[Depends(verify_api_key)])
async def attribution_buckets(body: dict):
    """Attribution: by holding period, by sector, by year. Sector lookup via Polygon."""
    try:
        trades = body.get("trades") or []
        if not trades:
            raise HTTPException(status_code=400, detail="trades is required and must be non-empty")

        closed = pm.match_fifo_lots(trades)

        # Sector lookup — best-effort via Polygon ticker_details, no caching for now
        sector_map: dict[str, str] = {}
        unique_tickers = sorted({c.get("ticker", "") for c in closed if c.get("ticker")})
        try:
            from lib.polygon_client import get_ticker_details
            for tk in unique_tickers:
                try:
                    details = get_ticker_details(tk)
                    sector_map[tk] = details.get("sector") or "Unknown"
                except Exception:
                    sector_map[tk] = "Unknown"
        except Exception:
            sector_map = {tk: "Unknown" for tk in unique_tickers}

        return {
            "by_holding_period": pm.bucket_by_holding_period(closed),
            "by_sector": pm.bucket_by_sector(closed, sector_map),
            "by_year": pm.bucket_by_year(closed),
            "n_closed_positions": len(closed),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
