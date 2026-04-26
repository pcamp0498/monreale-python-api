from fastapi import APIRouter, Depends, HTTPException
from lib.auth import verify_api_key
from typing import Optional
import os
import requests

router = APIRouter()

PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"


def perplexity_search(query: str, model: str = "sonar-pro", system: str = "You are a financial research analyst. Be specific, cite sources, and focus on data relevant to institutional investors.", max_tokens: int = 800) -> dict:
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

        market_query = f"""What happened in US financial markets today, {today}? Include:
- S&P 500, NASDAQ, Dow Jones performance with exact percentage changes
- Which sectors led and which lagged
- Key catalysts and news driving moves
- Any Federal Reserve or economic data releases

Use the most recent trading day data available. Write 2-3 paragraphs for institutional investors. Be specific with numbers and cite sources."""

        market = perplexity_search_with_fallback(
            market_query,
            model="sonar-pro",
            system="You are a Bloomberg markets reporter. Be specific, cite data, use exact percentages.",
        )
        macro_query = f"""What were the key macroeconomic events, Fed statements, or economic data releases today or this week {today}?

Focus on: inflation, employment, GDP, interest rates, Fed speakers.

IMPORTANT FORMATTING RULES:
- Write in plain paragraphs only
- No tables, no markdown tables (no | characters)
- No bullet point lists
- No headers or subheaders
- Plain prose only, 3-4 sentences maximum
- Cite sources with [1][2] notation inline"""

        macro = perplexity_search_with_fallback(
            macro_query,
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


def get_insider_transactions(ticker: str) -> dict:
    """Fetch recent Form 4 insider transactions from SEC EDGAR full-text search."""
    headers = {
        "User-Agent": "MonrealeOS research@monrealecapital.com",
        "Accept-Encoding": "gzip, deflate",
    }
    try:
        # SEC EDGAR full-text search restricted to Form 4
        from datetime import datetime, timedelta
        startdt = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
        enddt = datetime.now().strftime("%Y-%m-%d")
        url = (
            f"https://efts.sec.gov/LATEST/search-index"
            f"?q=%22{ticker.upper()}%22&forms=4&dateRange=custom&startdt={startdt}&enddt={enddt}"
        )
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return {"source": "SEC EDGAR Form 4", "ticker": ticker, "transactions": [], "note": f"EDGAR returned {resp.status_code}"}

        data = resp.json()
        hits = data.get("hits", {}).get("hits", [])
        transactions = []
        for hit in hits[:15]:
            src = hit.get("_source", {})
            ciks = src.get("ciks", [])
            adsh = src.get("adsh", "")
            cik = ciks[0] if ciks else ""
            sec_url = ""
            if cik and adsh:
                accession_clean = adsh.replace("-", "")
                sec_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=4&dateb=&owner=include&count=40"
                # Direct accession link
                sec_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_clean}/{adsh}-index.htm"
            display_names = src.get("display_names", [])
            transactions.append({
                "filer_name": display_names[0] if display_names else "Unknown",
                "filed_date": src.get("file_date", ""),
                "form_type": (src.get("form", "4") if isinstance(src.get("form"), str) else "4"),
                "description": src.get("file_description") or src.get("description") or "",
                "url": sec_url or f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker.upper()}%22&forms=4",
            })
        return {"source": "SEC EDGAR Form 4", "ticker": ticker, "transactions": transactions, "count": len(transactions)}
    except Exception as e:
        return {"source": "SEC EDGAR", "ticker": ticker, "transactions": [], "error": str(e)}


@router.post("/management-intelligence", dependencies=[Depends(verify_api_key)])
async def management_intelligence(body: dict):
    """Comprehensive management research: Perplexity bios + SEC EDGAR Form 4 + analyst consensus + warning flags."""
    try:
        company = body.get("company", "")
        ticker = body.get("ticker", "")
        if not company and not ticker:
            raise HTTPException(status_code=400, detail="company or ticker required")

        company_label = company or ticker
        ticker_label = ticker if ticker else "public company"

        mgmt_query = f"""Research the executive team at {company_label} ({ticker_label}).

For the CEO and CFO provide:
- Full name and title
- Educational background
- Career history (previous companies and roles)
- Tenure at {company_label}
- Notable achievements or controversies
- Compensation if publicly available

Write in plain prose, no tables. Focus on investment-relevant information. Cite all sources."""

        management = perplexity_search(
            mgmt_query,
            model="sonar-pro",
            system="You are a due diligence analyst. Be factual, cite sources, flag any concerns clearly.",
            max_tokens=900,
        )

        insider_transactions = {"transactions": [], "source": "SEC EDGAR Form 4"}
        if ticker:
            insider_transactions = get_insider_transactions(ticker)

        analyst_consensus = None
        if ticker:
            consensus_query = f"""What is the current analyst consensus for {ticker} ({company_label})?

Provide:
- Buy/Hold/Sell rating breakdown
- Average price target
- Highest and lowest price targets
- Number of analysts covering
- Recent rating changes (last 30 days)

Use the most recent data available. Plain prose, no tables."""
            analyst_consensus = perplexity_search(
                consensus_query,
                model="sonar-pro",
                max_tokens=400,
            )

        warning_query = f"""Are there any red flags or concerns about {company_label} ({ticker if ticker else ''})?

Check for:
- Auditor changes or audit concerns
- CFO or executive departures recently
- High short interest (>10% of float)
- Insider selling patterns
- SEC investigations or legal issues
- Earnings quality concerns
- Revenue recognition issues

Be specific. Only report confirmed issues, not speculation. Cite sources. Plain prose only."""

        warning_flags = perplexity_search(
            warning_query,
            model="sonar-pro",
            system="You are a forensic analyst. Be factual and cite sources. Only report confirmed issues.",
            max_tokens=600,
        )

        return {
            "company": company,
            "ticker": ticker,
            "management": management,
            "insider_transactions": insider_transactions,
            "analyst_consensus": analyst_consensus,
            "warning_flags": warning_flags,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quality-of-earnings", dependencies=[Depends(verify_api_key)])
async def quality_of_earnings(body: dict):
    """Quality of earnings analysis: Polygon financials + Perplexity narrative."""
    try:
        ticker = body.get("ticker", "")
        if not ticker:
            raise HTTPException(status_code=400, detail="ticker required")

        api_key = os.environ.get("POLYGON_API_KEY")
        url = f"https://api.polygon.io/vX/reference/financials?ticker={ticker.upper()}&limit=4&apiKey={api_key}"

        revenue = revenue_prior = net_income = total_assets = 0
        accruals_ratio = None
        revenue_growth = None
        polygon_error = None

        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                if len(results) >= 2:
                    def safe_get(d: dict, key: str):
                        val = d.get(key, {})
                        if isinstance(val, dict):
                            return val.get("value", 0) or 0
                        return val or 0

                    current = results[0].get("financials", {})
                    prior = results[1].get("financials", {})
                    inc = current.get("income_statement", {})
                    bal = current.get("balance_sheet", {})
                    inc_prior = prior.get("income_statement", {})

                    revenue = safe_get(inc, "revenues")
                    revenue_prior = safe_get(inc_prior, "revenues")
                    net_income = safe_get(inc, "net_income_loss")
                    total_assets = safe_get(bal, "assets")

                    if total_assets:
                        accruals_ratio = round(net_income / total_assets, 4)
                    if revenue_prior:
                        revenue_growth = round((revenue - revenue_prior) / abs(revenue_prior) * 100, 2)
                else:
                    polygon_error = "Insufficient financial data (need 2+ periods)"
            else:
                polygon_error = f"Polygon returned {resp.status_code}"
        except Exception as e:
            polygon_error = str(e)

        quality_query = f"""Analyze the earnings quality for {ticker.upper()}.

Recent financials (most recent period):
- Revenue: ${revenue:,.0f}
- Revenue growth: {revenue_growth}%
- Net income: ${net_income:,.0f}

Provide:
1. EPS surprise history (last 4 quarters)
2. Any earnings management concerns
3. Revenue recognition red flags
4. Cash flow vs earnings quality
5. Accruals and working capital trends

Be specific with data. Plain prose, no tables. Cite sources."""

        ai_analysis = perplexity_search(
            quality_query,
            model="sonar-pro",
            max_tokens=700,
        )

        if accruals_ratio is None:
            accruals_interpretation = "Unable to calculate from available data"
        elif accruals_ratio > 0.1:
            accruals_interpretation = "HIGH accruals — potential earnings quality concern"
        elif accruals_ratio < -0.1:
            accruals_interpretation = "LOW/negative accruals — typically conservative"
        else:
            accruals_interpretation = "Normal accruals level"

        return {
            "ticker": ticker.upper(),
            "revenue": revenue,
            "revenue_growth_pct": revenue_growth,
            "net_income": net_income,
            "total_assets": total_assets,
            "accruals_ratio": accruals_ratio,
            "accruals_interpretation": accruals_interpretation,
            "polygon_error": polygon_error,
            "ai_analysis": ai_analysis,
            "data_source": "Polygon + Perplexity",
        }
    except HTTPException:
        raise
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
