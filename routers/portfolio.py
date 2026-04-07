from fastapi import APIRouter, Depends, HTTPException
from lib.auth import verify_api_key
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()

class OptimizeRequest(BaseModel):
    tickers: List[str]
    method: str = "max_sharpe"
    risk_free_rate: float = 0.053

class XirrRequest(BaseModel):
    cash_flows: List[float]
    dates: List[str]

class PerformanceRequest(BaseModel):
    returns: List[float]
    benchmark_returns: Optional[List[float]] = None
    risk_free_rate: float = 0.053

@router.post("/optimize", dependencies=[Depends(verify_api_key)])
async def optimize_portfolio(request: OptimizeRequest):
    try:
        from pypfopt import EfficientFrontier, risk_models, expected_returns
        import yfinance as yf
        import pandas as pd

        raw = yf.download(
            request.tickers,
            period="2y",
            progress=False,
            auto_adjust=True,
            threads=False
        )

        if raw is None or raw.empty:
            raise HTTPException(status_code=400, detail=f"yfinance returned no data for {request.tickers}")

        # Handle MultiIndex columns (multiple tickers) vs flat (single ticker)
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        elif "Close" in raw.columns:
            prices = raw[["Close"]]
            prices.columns = request.tickers
        else:
            prices = raw

        prices = prices.dropna(how="all")

        if prices.empty:
            raise HTTPException(status_code=400, detail=f"Price data empty after cleaning for {request.tickers}")

        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.sample_cov(prices)
        ef = EfficientFrontier(mu, S)

        if request.method == "min_volatility":
            ef.min_volatility()
        else:
            ef.max_sharpe(risk_free_rate=request.risk_free_rate)

        cleaned = ef.clean_weights()
        perf = ef.portfolio_performance(verbose=False, risk_free_rate=request.risk_free_rate)

        return {
            "weights": dict(cleaned),
            "expected_return": perf[0],
            "volatility": perf[1],
            "sharpe_ratio": perf[2],
            "method": request.method,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimizer error: {str(e)}")

@router.post("/xirr", dependencies=[Depends(verify_api_key)])
async def calculate_xirr(request: XirrRequest):
    try:
        from pyxirr import xirr
        from datetime import datetime
        dates = [datetime.strptime(d, "%Y-%m-%d") for d in request.dates]
        result = xirr(dates, request.cash_flows)
        return {"xirr": result, "xirr_pct": result * 100 if result else None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/performance", dependencies=[Depends(verify_api_key)])
async def calculate_performance(request: PerformanceRequest):
    try:
        import quantstats as qs
        import pandas as pd
        rs = pd.Series(request.returns)
        rf_daily = request.risk_free_rate / 252
        metrics = {
            "total_return": float(qs.stats.comp(rs)),
            "sharpe": float(qs.stats.sharpe(rs, rf=rf_daily)),
            "sortino": float(qs.stats.sortino(rs, rf=rf_daily)),
            "max_drawdown": float(qs.stats.max_drawdown(rs)),
            "volatility": float(qs.stats.volatility(rs)),
            "win_rate": float(qs.stats.win_rate(rs)),
            "best_day": float(qs.stats.best(rs)),
            "worst_day": float(qs.stats.worst(rs)),
            "avg_return": float(qs.stats.avg_return(rs)),
        }
        if request.benchmark_returns:
            bs = pd.Series(request.benchmark_returns)
            greeks = qs.stats.greeks(rs, bs)
            metrics["alpha"] = float(greeks.alpha) if hasattr(greeks, "alpha") else None
            metrics["beta"] = float(greeks.beta) if hasattr(greeks, "beta") else None
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))