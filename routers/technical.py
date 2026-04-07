from fastapi import APIRouter, Depends, HTTPException
from lib.auth import verify_api_key
from pydantic import BaseModel
from typing import List

router = APIRouter()

PERIOD_DAYS = {"1m": 30, "3m": 90, "6m": 180, "1y": 365, "2y": 730}


class TechnicalRequest(BaseModel):
    ticker: str
    indicators: List[str]
    period: str = "1y"


@router.post("/indicators", dependencies=[Depends(verify_api_key)])
async def get_technical_indicators(request: TechnicalRequest):
    try:
        import ta
        import pandas as pd
        from lib.polygon_client import get_price_history

        days = PERIOD_DAYS.get(request.period, 365)
        results_raw = get_price_history(request.ticker.upper(), days=days)

        df = pd.DataFrame(results_raw)
        df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
        df = df.rename(
            columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}
        )
        df = df.set_index("timestamp")

        if df.empty:
            raise HTTPException(status_code=404, detail="No price data found")

        results = {}

        if "rsi" in request.indicators:
            rsi = ta.momentum.RSIIndicator(df["Close"], window=14)
            current_rsi = float(rsi.rsi().iloc[-1])
            results["rsi"] = {
                "current": current_rsi,
                "signal": "oversold"
                if current_rsi < 30
                else "overbought"
                if current_rsi > 70
                else "neutral",
            }

        if "macd" in request.indicators:
            macd = ta.trend.MACD(df["Close"])
            results["macd"] = {
                "current_macd": float(macd.macd().iloc[-1]),
                "current_signal": float(macd.macd_signal().iloc[-1]),
                "bullish_crossover": bool(
                    macd.macd().iloc[-1] > macd.macd_signal().iloc[-1]
                ),
            }

        if "bbands" in request.indicators:
            bb = ta.volatility.BollingerBands(df["Close"])
            results["bbands"] = {
                "current_upper": float(bb.bollinger_hband().iloc[-1]),
                "current_lower": float(bb.bollinger_lband().iloc[-1]),
                "pct_b": float(bb.bollinger_pband().iloc[-1]),
            }

        if "atr" in request.indicators:
            atr_ind = ta.volatility.AverageTrueRange(
                df["High"], df["Low"], df["Close"]
            )
            results["atr"] = {
                "current": float(atr_ind.average_true_range().iloc[-1])
            }

        if "obv" in request.indicators:
            obv = ta.volume.OnBalanceVolumeIndicator(df["Close"], df["Volume"])
            results["obv"] = {
                "current": float(obv.on_balance_volume().iloc[-1]),
                "trend": "accumulation"
                if obv.on_balance_volume().iloc[-1]
                > obv.on_balance_volume().iloc[-20]
                else "distribution",
            }

        if "sma" in request.indicators:
            results["sma"] = {
                "sma20": float(ta.trend.sma_indicator(df["Close"], 20).iloc[-1]),
                "sma50": float(ta.trend.sma_indicator(df["Close"], 50).iloc[-1]),
                "sma200": float(
                    ta.trend.sma_indicator(df["Close"], 200).iloc[-1]
                ),
                "above_200": bool(
                    df["Close"].iloc[-1]
                    > ta.trend.sma_indicator(df["Close"], 200).iloc[-1]
                ),
            }

        results["ticker"] = request.ticker.upper()
        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
