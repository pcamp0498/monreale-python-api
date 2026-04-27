from fastapi import APIRouter, Depends, HTTPException, Query
from lib.auth import verify_api_key
from typing import Optional
import math

router = APIRouter()


@router.get("/fred/{series_id}", dependencies=[Depends(verify_api_key)])
async def get_fred_series(
    series_id: str,
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
):
    """Fetch FRED economic series data."""
    try:
        from fredapi import Fred
        import os

        fred = Fred(api_key=os.getenv("FRED_API_KEY", ""))

        kwargs = {}
        if start_date:
            kwargs["observation_start"] = start_date
        if end_date:
            kwargs["observation_end"] = end_date

        series = fred.get_series(series_id, **kwargs)

        # Get series info
        try:
            info = fred.get_series_info(series_id)
            title = info.get("title", series_id)
            units = info.get("units", "")
        except Exception:
            title = series_id
            units = ""

        return {
            "series_id": series_id,
            "title": title,
            "units": units,
            "data": [
                {
                    "date": str(date.date()),
                    "value": float(value) if not math.isnan(value) else None,
                }
                for date, value in series.items()
            ][
                -100:
            ],  # Last 100 observations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/economic-calendar", dependencies=[Depends(verify_api_key)])
async def get_economic_calendar():
    """Get latest values for key economic data series from FRED."""
    try:
        from fredapi import Fred
        import os
        from datetime import datetime

        fred = Fred(api_key=os.getenv("FRED_API_KEY", ""))

        key_series = [
            {"id": "CPIAUCSL", "name": "CPI (Inflation)", "frequency": "Monthly"},
            {"id": "UNRATE", "name": "Unemployment Rate", "frequency": "Monthly"},
            {"id": "GDP", "name": "GDP Growth", "frequency": "Quarterly"},
            {"id": "FEDFUNDS", "name": "Fed Funds Rate", "frequency": "Monthly"},
            {"id": "T10Y2Y", "name": "Yield Curve (10Y-2Y)", "frequency": "Daily"},
            {"id": "SOFR30DAYAVG", "name": "SOFR 30-Day Avg", "frequency": "Daily"},
            {"id": "BAMLH0A0HYM2", "name": "HY OAS Spread", "frequency": "Daily"},
        ]

        results = []
        for s in key_series:
            try:
                data = fred.get_series(s["id"])
                latest = data.iloc[-1] if len(data) > 0 else None
                latest_date = (
                    str(data.index[-1].date()) if len(data) > 0 else None
                )
                prior = data.iloc[-2] if len(data) > 1 else None

                results.append(
                    {
                        "series_id": s["id"],
                        "name": s["name"],
                        "frequency": s["frequency"],
                        "latest_value": (
                            round(float(latest), 4) if latest is not None else None
                        ),
                        "latest_date": latest_date,
                        "prior_value": (
                            round(float(prior), 4) if prior is not None else None
                        ),
                        "change": (
                            round(float(latest - prior), 4)
                            if latest is not None and prior is not None
                            else None
                        ),
                    }
                )
            except Exception:
                continue

        # Hardcoded 2026 release schedule — refresh quarterly
        scheduled = [
            {"date": "2026-04-25", "event": "PCE Price Index (March)", "importance": "high", "category": "inflation"},
            {"date": "2026-04-29", "event": "Consumer Confidence", "importance": "medium", "category": "consumer"},
            {"date": "2026-04-30", "event": "GDP Advance Q1 2026", "importance": "high", "category": "growth"},
            {"date": "2026-04-30", "event": "ADP Employment", "importance": "medium", "category": "employment"},
            {"date": "2026-05-01", "event": "Non-Farm Payrolls (April)", "importance": "high", "category": "employment"},
            {"date": "2026-05-01", "event": "Unemployment Rate", "importance": "high", "category": "employment"},
            {"date": "2026-05-06", "event": "FOMC Meeting Day 1", "importance": "high", "category": "fed"},
            {"date": "2026-05-07", "event": "FOMC Rate Decision", "importance": "high", "category": "fed"},
            {"date": "2026-05-13", "event": "CPI (April)", "importance": "high", "category": "inflation"},
            {"date": "2026-05-15", "event": "PPI (April)", "importance": "medium", "category": "inflation"},
            {"date": "2026-05-15", "event": "Retail Sales (April)", "importance": "high", "category": "consumer"},
            {"date": "2026-05-22", "event": "Existing Home Sales", "importance": "medium", "category": "housing"},
            {"date": "2026-05-28", "event": "GDP Second Estimate Q1", "importance": "medium", "category": "growth"},
            {"date": "2026-05-29", "event": "PCE Price Index (April)", "importance": "high", "category": "inflation"},
            {"date": "2026-06-03", "event": "Non-Farm Payrolls (May)", "importance": "high", "category": "employment"},
            {"date": "2026-06-10", "event": "CPI (May)", "importance": "high", "category": "inflation"},
            {"date": "2026-06-17", "event": "FOMC Meeting Day 1", "importance": "high", "category": "fed"},
            {"date": "2026-06-18", "event": "FOMC Rate Decision", "importance": "high", "category": "fed"},
        ]
        today_str = datetime.now().strftime("%Y-%m-%d")
        upcoming = [e for e in scheduled if e["date"] >= today_str]
        upcoming.sort(key=lambda x: x["date"])

        return {
            "series": results,
            "events": upcoming[:20],
            "as_of": datetime.now().isoformat(),
            "source": "Monreale Economic Calendar 2026",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
