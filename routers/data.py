from fastapi import APIRouter, Depends, HTTPException, Query
from lib.auth import verify_api_key
from pydantic import BaseModel
from typing import Optional
import yfinance as yf

router = APIRouter()


class FundamentalsResponse(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    ev_ebitda: Optional[float] = None
    price_to_book: Optional[float] = None
    price_to_sales: Optional[float] = None
    fcf_yield: Optional[float] = None
    revenue_ttm: Optional[float] = None
    ebitda_ttm: Optional[float] = None
    net_income_ttm: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    beta: Optional[float] = None
    dividend_yield: Optional[float] = None
    week_52_high: Optional[float] = None
    week_52_low: Optional[float] = None
    avg_volume_30d: Optional[float] = None
    revenue_growth_yoy: Optional[float] = None
    earnings_growth_yoy: Optional[float] = None


@router.get("/fundamentals/{ticker}", dependencies=[Depends(verify_api_key)])
async def get_fundamentals(ticker: str) -> FundamentalsResponse:
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info

        revenue_growth = None
        try:
            income = stock.income_stmt
            if income is not None and len(income.columns) >= 2:
                rev_curr = income.loc["Total Revenue"].iloc[0] if "Total Revenue" in income.index else None
                rev_prev = income.loc["Total Revenue"].iloc[1] if "Total Revenue" in income.index else None
                if rev_curr and rev_prev and rev_prev != 0:
                    revenue_growth = (rev_curr - rev_prev) / abs(rev_prev)
        except Exception:
            pass

        fcf = info.get("freeCashflow")
        mcap = info.get("marketCap")

        return FundamentalsResponse(
            ticker=ticker.upper(),
            company_name=info.get("longName"),
            market_cap=mcap,
            pe_ratio=info.get("trailingPE"),
            forward_pe=info.get("forwardPE"),
            ev_ebitda=info.get("enterpriseToEbitda"),
            price_to_book=info.get("priceToBook"),
            price_to_sales=info.get("priceToSalesTrailing12Months"),
            fcf_yield=(fcf / mcap) if fcf and mcap else None,
            revenue_ttm=info.get("totalRevenue"),
            ebitda_ttm=info.get("ebitda"),
            net_income_ttm=info.get("netIncomeToCommon"),
            gross_margin=info.get("grossMargins"),
            operating_margin=info.get("operatingMargins"),
            roe=info.get("returnOnEquity"),
            roa=info.get("returnOnAssets"),
            debt_to_equity=info.get("debtToEquity"),
            current_ratio=info.get("currentRatio"),
            beta=info.get("beta"),
            dividend_yield=info.get("dividendYield"),
            week_52_high=info.get("fiftyTwoWeekHigh"),
            week_52_low=info.get("fiftyTwoWeekLow"),
            avg_volume_30d=info.get("averageVolume"),
            revenue_growth_yoy=revenue_growth,
            earnings_growth_yoy=info.get("earningsGrowth"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch fundamentals: {str(e)}")


@router.get("/options/{ticker}", dependencies=[Depends(verify_api_key)])
async def get_options_chain(ticker: str, expiry: Optional[str] = Query(None)):
    try:
        stock = yf.Ticker(ticker.upper())
        expirations = stock.options
        if not expirations:
            return {"ticker": ticker, "expirations": [], "chain": None}

        if expiry and expiry in expirations:
            chain = stock.option_chain(expiry)
            return {
                "ticker": ticker,
                "expirations": list(expirations),
                "expiry": expiry,
                "calls": chain.calls.to_dict("records"),
                "puts": chain.puts.to_dict("records"),
            }

        return {"ticker": ticker, "expirations": list(expirations), "nearest_expiry": expirations[0] if expirations else None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/insiders/{ticker}", dependencies=[Depends(verify_api_key)])
async def get_insider_transactions(ticker: str):
    try:
        stock = yf.Ticker(ticker.upper())
        insider = stock.insider_transactions
        if insider is None or insider.empty:
            return {"ticker": ticker, "transactions": []}
        return {"ticker": ticker, "transactions": insider.head(20).to_dict("records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/institutional/{ticker}", dependencies=[Depends(verify_api_key)])
async def get_institutional_holders(ticker: str):
    try:
        stock = yf.Ticker(ticker.upper())
        holders = stock.institutional_holders
        if holders is None or holders.empty:
            return {"ticker": ticker, "holders": []}
        return {"ticker": ticker, "holders": holders.head(20).to_dict("records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
