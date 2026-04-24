from fastapi import APIRouter, Depends, HTTPException
from lib.auth import verify_api_key
from typing import Optional
import os
import requests

router = APIRouter()

PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"


def perplexity_search(query: str, model: str = "sonar", system: str = "You are a financial research analyst. Be specific, cite sources, and focus on data relevant to institutional investors.", max_tokens: int = 800) -> dict:
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        return {"answer": "Perplexity API key not configured.", "citations": [], "model": model, "usage": {}}

    try:
        resp = requests.post(
            PERPLEXITY_API_URL,
            json={
                "model": model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": query}],
                "max_tokens": max_tokens,
                "return_citations": True,
                "return_related_questions": False,
            },
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "answer": data["choices"][0]["message"]["content"],
            "citations": data.get("citations", []),
            "model": model,
            "usage": data.get("usage", {}),
        }
    except Exception as e:
        return {"answer": f"Research unavailable: {str(e)}", "citations": [], "model": model, "usage": {}}


def perplexity_search_with_fallback(query: str, model: str = "sonar-pro", **kwargs) -> dict:
    """Search with automatic retry if response indicates missing data."""
    result = perplexity_search(query, model=model, **kwargs)
    answer = (result.get("answer") or "").lower()
    no_data = ["don't have search results", "don't have data", "no information available", "cannot find", "i'm unable"]
    if any(phrase in answer for phrase in no_data):
        fallback_query = query + "\n\nNote: Use the most recent available data if today's data is not yet available. Indicate the date of the data you are using."
        result = perplexity_search(fallback_query, model=model, **kwargs)
    return result


@router.get("/morning-brief", dependencies=[Depends(verify_api_key)])
async def morning_brief_intel(date: Optional[str] = None):
    """Real-time market intelligence for morning brief."""
    try:
        from datetime import datetime
        today = date or datetime.now().strftime("%B %d, %Y")

        market_query = f"""Search for the latest US stock market news and performance for {today}.

I need:
1. How did the S&P 500, NASDAQ, and Dow perform?
2. What sectors led and lagged today?
3. What were the top news stories moving markets?
4. Any Federal Reserve or economic data releases?

Use the most recent data available. If today's data isn't available, use the most recent trading day's data and note the date. Write 2-3 paragraphs for institutional investors. Be specific about percentage moves and catalysts."""

        market = perplexity_search_with_fallback(
            market_query,
            model="sonar-pro",
            system="You are a Bloomberg markets reporter. Be specific, cite data, use exact percentages.",
        )
        macro = perplexity_search_with_fallback(
            f"What were the key macroeconomic events, Fed statements, or economic data releases today or this week {today}? Focus on inflation, employment, GDP, interest rates, Fed speakers. Use most recent available data.",
            model="sonar-pro",
            max_tokens=400,
        )
        sectors = perplexity_search_with_fallback(
            f"Which sectors led and lagged in US equities today {today}? What were the catalysts for sector rotation? 3-4 sentences. Use most recent trading day data if today's is unavailable.",
            model="sonar-pro",
            max_tokens=300,
        )

        return {"date": today, "market_summary": market, "macro_events": macro, "sector_rotation": sectors}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/industry-analysis", dependencies=[Depends(verify_api_key)])
async def industry_analysis(body: dict):
    """Industry analysis for deal intelligence."""
    try:
        industry = body.get("industry", "")
        company = body.get("company", "")
        if not industry:
            raise HTTPException(status_code=400, detail="industry required")

        overview = perplexity_search(
            f"Institutional-grade industry analysis for {industry} in 2025-2026. Cover: market size/TAM, growth drivers, major players, margins, key risks. Cite sources."
        )
        competitive = perplexity_search(
            f"Major competitors in {industry}? Typical EV/EBITDA and revenue multiples? Recent M&A transactions (2023-2026)?"
        )
        company_result = None
        if company:
            company_result = perplexity_search(
                f"Brief background on {company} in {industry}. Market position, competitive advantages, recent performance, key risks."
            )

        return {"industry": industry, "company": company, "overview": overview, "competitive_landscape": competitive, "company_background": company_result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/comparable-transactions", dependencies=[Depends(verify_api_key)])
async def comparable_transactions(body: dict):
    """Find comparable M&A transactions."""
    try:
        industry = body.get("industry", "")
        deal_type = body.get("deal_type", "pe")
        years_back = body.get("years_back", 3)
        deal_size = body.get("deal_size", "")

        result = perplexity_search(
            f"List recent {deal_type} transactions in {industry} from past {years_back} years. For each: target, acquirer, EV, EV/Revenue, EV/EBITDA, close date. Focus on {deal_size or 'mid-market'} deals.",
            system="You are an M&A research analyst. Provide specific transaction data with sources.",
            max_tokens=1000,
        )

        return {"industry": industry, "deal_type": deal_type, "transactions": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/management-background", dependencies=[Depends(verify_api_key)])
async def management_background(body: dict):
    """Research management team."""
    try:
        company = body.get("company", "")
        if not company:
            raise HTTPException(status_code=400, detail="company required")

        result = perplexity_search(
            f"Key executives at {company}? For CEO and CFO: background, prior roles, track record, tenure, any concerns. Investment-relevant only.",
            system="You are a due diligence researcher. Be factual, cite sources, flag concerns.",
        )

        return {"company": company, "management_research": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/market-sizing", dependencies=[Depends(verify_api_key)])
async def market_sizing(body: dict):
    """TAM/SAM/SOM analysis."""
    try:
        sector = body.get("sector", "")
        geography = body.get("geography", "United States")
        desc = body.get("company_description", "")

        result = perplexity_search(
            f"TAM/SAM/SOM analysis for {sector} in {geography}. {f'Company: {desc}' if desc else ''} Include: TAM with source, SAM, CAGR, key drivers, recent funding. Be specific with numbers.",
            system="You are a VC research analyst. Provide specific market data with citations.",
        )

        return {"sector": sector, "geography": geography, "market_analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sba-market-validation", dependencies=[Depends(verify_api_key)])
async def sba_market_validation(body: dict):
    """Market validation for SBA deals."""
    try:
        business_type = body.get("business_type", "")
        location = body.get("location", "")
        asking_price = body.get("asking_price", "")

        result = perplexity_search(
            f"Market conditions for acquiring a {business_type} in {location}. Industry health, local conditions, typical multiples, key risks, SBA appetite. {f'Is {asking_price} reasonable?' if asking_price else ''}",
            system="You are an SBA lending specialist. Provide practical market intelligence.",
        )

        return {"business_type": business_type, "location": location, "market_validation": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
