from fastapi import APIRouter, Depends, HTTPException, Query
from lib.auth import verify_api_key
from pydantic import BaseModel
from typing import Optional

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


def _safe_float(val) -> Optional[float]:
    """Safely convert a value to float, returning None on failure."""
    if val is None:
        return None
    try:
        f = float(val)
        return f if f == f else None  # NaN check
    except (ValueError, TypeError):
        return None


def _extract_from_polygon(financials: list) -> dict:
    """Extract key metrics from Polygon financials API results."""
    if not financials:
        return {}

    latest = financials[0]
    inc = latest.get("financials", {}).get("income_statement", {})
    bs = latest.get("financials", {}).get("balance_sheet", {})
    cf = latest.get("financials", {}).get("cash_flow_statement", {})

    revenue = _safe_float(inc.get("revenues", {}).get("value"))
    net_income = _safe_float(inc.get("net_income_loss", {}).get("value"))
    gross_profit = _safe_float(inc.get("gross_profit", {}).get("value"))
    operating_income = _safe_float(
        inc.get("operating_income_loss", {}).get("value")
    )
    ebitda = _safe_float(
        inc.get("income_loss_from_continuing_operations_before_tax", {}).get(
            "value"
        )
    )

    total_assets = _safe_float(bs.get("assets", {}).get("value"))
    total_equity = _safe_float(bs.get("equity", {}).get("value"))
    total_debt = _safe_float(
        bs.get("long_term_debt", {}).get("value")
    )
    current_assets = _safe_float(bs.get("current_assets", {}).get("value"))
    current_liabilities = _safe_float(
        bs.get("current_liabilities", {}).get("value")
    )

    operating_cf = _safe_float(
        cf.get("net_cash_flow_from_operating_activities", {}).get("value")
    )

    # Calculate margins
    gross_margin = (gross_profit / revenue) if gross_profit and revenue else None
    operating_margin = (
        (operating_income / revenue) if operating_income and revenue else None
    )
    roe = (net_income / total_equity) if net_income and total_equity else None
    roa = (net_income / total_assets) if net_income and total_assets else None
    debt_to_equity = (
        (total_debt / total_equity) * 100
        if total_debt and total_equity and total_equity != 0
        else None
    )
    current_ratio = (
        (current_assets / current_liabilities)
        if current_assets and current_liabilities and current_liabilities != 0
        else None
    )

    # Revenue growth YoY
    revenue_growth = None
    if len(financials) >= 2:
        prev_inc = financials[1].get("financials", {}).get("income_statement", {})
        prev_revenue = _safe_float(prev_inc.get("revenues", {}).get("value"))
        if revenue and prev_revenue and prev_revenue != 0:
            revenue_growth = (revenue - prev_revenue) / abs(prev_revenue)

    return {
        "revenue_ttm": revenue,
        "ebitda_ttm": ebitda,
        "net_income_ttm": net_income,
        "gross_margin": gross_margin,
        "operating_margin": operating_margin,
        "roe": roe,
        "roa": roa,
        "debt_to_equity": debt_to_equity,
        "current_ratio": current_ratio,
        "revenue_growth_yoy": revenue_growth,
        "free_cash_flow": operating_cf,
    }


@router.get("/fundamentals/{ticker}", dependencies=[Depends(verify_api_key)])
async def get_fundamentals(ticker: str) -> FundamentalsResponse:
    try:
        from lib.polygon_client import get_financials, get_snapshot
        import requests
        import os

        ticker_upper = ticker.upper()
        polygon_key = os.getenv("POLYGON_API_KEY", "")

        # Fetch Polygon financials + ticker details + snapshot in parallel-ish
        polygon_metrics = {}
        company_name = None
        market_cap = None
        pe_ratio = None
        forward_pe = None
        ev_ebitda = None
        price_to_book = None
        price_to_sales = None
        beta = None
        dividend_yield = None
        week_52_high = None
        week_52_low = None
        avg_volume = None

        # 1) Polygon financials
        try:
            financials = get_financials(ticker_upper, limit=4)
            polygon_metrics = _extract_from_polygon(financials)
        except Exception as e:
            print(f"[fundamentals] Polygon financials failed: {e}")

        # 2) Polygon ticker details for company info + market cap
        try:
            details_url = f"https://api.polygon.io/v3/reference/tickers/{ticker_upper}"
            details_res = requests.get(
                details_url, params={"apiKey": polygon_key}, timeout=10
            )
            if details_res.ok:
                info = details_res.json().get("results", {})
                company_name = info.get("name")
                market_cap = _safe_float(info.get("market_cap"))
        except Exception as e:
            print(f"[fundamentals] Polygon ticker details failed: {e}")

        # 3) Polygon snapshot for current price metrics
        try:
            snap = get_snapshot(ticker_upper)
            price = snap.get("price")
            if price and market_cap:
                # Derive P/E if we have net income
                ni = polygon_metrics.get("net_income_ttm")
                if ni and ni > 0:
                    shares = market_cap / price
                    eps = ni / shares if shares else None
                    pe_ratio = price / eps if eps else None
        except Exception:
            pass

        # 4) Fallback to yfinance for fields Polygon doesn't provide
        try:
            import yfinance as yf

            stock = yf.Ticker(ticker_upper)
            info = stock.info or {}
            if not company_name:
                company_name = info.get("longName")
            if not market_cap:
                market_cap = _safe_float(info.get("marketCap"))
            if not pe_ratio:
                pe_ratio = _safe_float(info.get("trailingPE"))
            forward_pe = _safe_float(info.get("forwardPE"))
            ev_ebitda = _safe_float(info.get("enterpriseToEbitda"))
            price_to_book = _safe_float(info.get("priceToBook"))
            price_to_sales = _safe_float(
                info.get("priceToSalesTrailing12Months")
            )
            beta = _safe_float(info.get("beta"))
            dividend_yield = _safe_float(info.get("dividendYield"))
            week_52_high = _safe_float(info.get("fiftyTwoWeekHigh"))
            week_52_low = _safe_float(info.get("fiftyTwoWeekLow"))
            avg_volume = _safe_float(info.get("averageVolume"))

            # Fill any missing fundamentals from yfinance
            if not polygon_metrics.get("revenue_ttm"):
                polygon_metrics["revenue_ttm"] = _safe_float(
                    info.get("totalRevenue")
                )
            if not polygon_metrics.get("ebitda_ttm"):
                polygon_metrics["ebitda_ttm"] = _safe_float(info.get("ebitda"))
            if not polygon_metrics.get("net_income_ttm"):
                polygon_metrics["net_income_ttm"] = _safe_float(
                    info.get("netIncomeToCommon")
                )
            if not polygon_metrics.get("gross_margin"):
                polygon_metrics["gross_margin"] = _safe_float(
                    info.get("grossMargins")
                )
            if not polygon_metrics.get("operating_margin"):
                polygon_metrics["operating_margin"] = _safe_float(
                    info.get("operatingMargins")
                )
            if not polygon_metrics.get("roe"):
                polygon_metrics["roe"] = _safe_float(info.get("returnOnEquity"))
            if not polygon_metrics.get("roa"):
                polygon_metrics["roa"] = _safe_float(info.get("returnOnAssets"))
            if not polygon_metrics.get("debt_to_equity"):
                polygon_metrics["debt_to_equity"] = _safe_float(
                    info.get("debtToEquity")
                )
            if not polygon_metrics.get("current_ratio"):
                polygon_metrics["current_ratio"] = _safe_float(
                    info.get("currentRatio")
                )
            if not polygon_metrics.get("revenue_growth_yoy"):
                polygon_metrics["revenue_growth_yoy"] = _safe_float(
                    info.get("revenueGrowth")
                )
        except Exception as e:
            print(f"[fundamentals] yfinance fallback failed (non-fatal): {e}")

        fcf = polygon_metrics.get("free_cash_flow")
        fcf_yield = (fcf / market_cap) if fcf and market_cap else None

        return FundamentalsResponse(
            ticker=ticker_upper,
            company_name=company_name,
            market_cap=market_cap,
            pe_ratio=pe_ratio,
            forward_pe=forward_pe,
            ev_ebitda=ev_ebitda,
            price_to_book=price_to_book,
            price_to_sales=price_to_sales,
            fcf_yield=fcf_yield,
            revenue_ttm=polygon_metrics.get("revenue_ttm"),
            ebitda_ttm=polygon_metrics.get("ebitda_ttm"),
            net_income_ttm=polygon_metrics.get("net_income_ttm"),
            gross_margin=polygon_metrics.get("gross_margin"),
            operating_margin=polygon_metrics.get("operating_margin"),
            roe=polygon_metrics.get("roe"),
            roa=polygon_metrics.get("roa"),
            debt_to_equity=polygon_metrics.get("debt_to_equity"),
            current_ratio=polygon_metrics.get("current_ratio"),
            beta=beta,
            dividend_yield=dividend_yield,
            week_52_high=week_52_high,
            week_52_low=week_52_low,
            avg_volume_30d=avg_volume,
            revenue_growth_yoy=polygon_metrics.get("revenue_growth_yoy"),
            earnings_growth_yoy=None,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch fundamentals: {str(e)}"
        )


@router.get("/options/{ticker}", dependencies=[Depends(verify_api_key)])
async def get_options_chain(ticker: str, expiry: Optional[str] = Query(None)):
    """Options data — yfinance only (Polygon options requires paid plan)."""
    try:
        import yfinance as yf

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

        return {
            "ticker": ticker,
            "expirations": list(expirations),
            "nearest_expiry": expirations[0] if expirations else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/insiders/{ticker}", dependencies=[Depends(verify_api_key)])
async def get_insider_transactions(ticker: str):
    """Insider transactions — yfinance only."""
    try:
        import yfinance as yf

        stock = yf.Ticker(ticker.upper())
        insider = stock.insider_transactions
        if insider is None or insider.empty:
            return {"ticker": ticker, "transactions": []}
        return {
            "ticker": ticker,
            "transactions": insider.head(20).to_dict("records"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/institutional/{ticker}", dependencies=[Depends(verify_api_key)])
async def get_institutional_holders(ticker: str):
    """Institutional holders — yfinance only."""
    try:
        import yfinance as yf

        stock = yf.Ticker(ticker.upper())
        holders = stock.institutional_holders
        if holders is None or holders.empty:
            return {"ticker": ticker, "holders": []}
        return {
            "ticker": ticker,
            "holders": holders.head(20).to_dict("records"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
