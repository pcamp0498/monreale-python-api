import os
import requests
from datetime import datetime, timedelta

POLYGON_BASE = "https://api.polygon.io"
API_KEY = os.getenv("POLYGON_API_KEY", "")


def get_price_history(ticker: str, days: int = 730) -> list:
    """Fetch daily price history from Polygon."""
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    url = f"{POLYGON_BASE}/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 730,
        "apiKey": API_KEY,
    }

    response = requests.get(url, params=params, timeout=10)

    if response.status_code != 200:
        raise ValueError(f"Polygon error {response.status_code} for {ticker}")

    data = response.json()

    if data.get("resultsCount", 0) == 0:
        raise ValueError(f"No price data returned for {ticker}")

    return data["results"]  # List of {o, h, l, c, v, t} dicts


def get_prices_dataframe(tickers: list, days: int = 730):
    """Fetch price history for multiple tickers and return as DataFrame."""
    import pandas as pd

    price_data = {}

    for ticker in tickers:
        try:
            results = get_price_history(ticker, days)
            closes = {
                datetime.fromtimestamp(r["t"] / 1000).strftime("%Y-%m-%d"): r["c"]
                for r in results
            }
            price_data[ticker] = closes
        except Exception as e:
            print(f"Failed to fetch {ticker}: {e}")
            continue

    if not price_data:
        raise ValueError(f"Could not fetch price data for any of {tickers}")

    df = pd.DataFrame(price_data)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.dropna(how="all")

    return df


def get_batch_snapshots(tickers: list) -> dict:
    """Get price snapshots for multiple tickers in one API call."""
    ticker_str = ",".join(tickers[:100])
    url = f"{POLYGON_BASE}/v2/snapshot/locale/us/markets/stocks/tickers"

    try:
        resp = requests.get(
            url, params={"tickers": ticker_str, "apiKey": API_KEY}, timeout=15
        )
        if resp.status_code != 200:
            return {}

        result = {}
        for snap in resp.json().get("tickers", []):
            t = snap.get("ticker")
            day = snap.get("day", {})
            prev = snap.get("prevDay", {})
            close = day.get("c") or snap.get("lastTrade", {}).get("p")
            prev_close = prev.get("c")
            day_change_pct = None
            if close and prev_close and prev_close > 0:
                day_change_pct = (close - prev_close) / prev_close * 100
            result[t] = {
                "price": close,
                "day_change_pct": day_change_pct,
                "volume": day.get("v"),
                "open": day.get("o"),
                "high": day.get("h"),
                "low": day.get("l"),
            }
        return result
    except Exception:
        return {}


def get_snapshot(ticker: str) -> dict:
    """Get current price snapshot for a ticker."""
    url = f"{POLYGON_BASE}/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
    params = {"apiKey": API_KEY}

    response = requests.get(url, params=params, timeout=10)

    if response.status_code != 200:
        raise ValueError(f"Polygon snapshot error for {ticker}")

    data = response.json()
    ticker_data = data.get("ticker", {})
    day = ticker_data.get("day", {})
    prev_day = ticker_data.get("prevDay", {})

    return {
        "ticker": ticker,
        "price": day.get("c") or prev_day.get("c"),
        "open": day.get("o"),
        "high": day.get("h"),
        "low": day.get("l"),
        "volume": day.get("v"),
        "prev_close": prev_day.get("c"),
        "change": ticker_data.get("todaysChange"),
        "change_pct": ticker_data.get("todaysChangePerc"),
    }


def get_financials(ticker: str, limit: int = 4) -> list:
    """Fetch raw financial data from Polygon financials API."""
    url = f"{POLYGON_BASE}/vX/reference/financials"
    params = {
        "ticker": ticker.upper(),
        "limit": limit,
        "apiKey": API_KEY,
    }

    response = requests.get(url, params=params, timeout=10)

    if response.status_code != 200:
        raise ValueError(f"Polygon financials error {response.status_code} for {ticker}")

    data = response.json()
    return data.get("results", [])


def get_parsed_financials(ticker: str, timeframe: str = "annual", limit: int = 4) -> list:
    """Fetch and parse Polygon financials into a flat list of dicts."""
    url = f"{POLYGON_BASE}/vX/reference/financials"
    params = {
        "ticker": ticker.upper(),
        "timeframe": timeframe,
        "limit": limit,
        "apiKey": API_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            return []
        results = resp.json().get("results", [])
    except Exception:
        return []

    def _val(section, key):
        item = section.get(key, {})
        return item.get("value") if isinstance(item, dict) else None

    parsed = []
    for r in results:
        fin = r.get("financials", {})
        inc = fin.get("income_statement", {})
        bal = fin.get("balance_sheet", {})
        cf = fin.get("cash_flow_statement", {})
        parsed.append({
            "period": f"{r.get('fiscal_period', '')} {r.get('fiscal_year', '')}",
            "fiscal_year": r.get("fiscal_year"),
            "fiscal_period": r.get("fiscal_period"),
            "revenue": _val(inc, "revenues"),
            "gross_profit": _val(inc, "gross_profit"),
            "operating_income": _val(inc, "operating_income_loss"),
            "net_income": _val(inc, "net_income_loss"),
            "ebitda": _val(inc, "ebitda"),
            "eps_basic": _val(inc, "basic_earnings_per_share"),
            "eps_diluted": _val(inc, "diluted_earnings_per_share"),
            "interest_expense": _val(inc, "interest_expense_operating"),
            "total_assets": _val(bal, "assets"),
            "total_liabilities": _val(bal, "liabilities"),
            "equity": _val(bal, "equity"),
            "cash": _val(bal, "cash_and_cash_equivalents") or _val(bal, "cash_and_short_term_investments"),
            "total_debt": _val(bal, "long_term_debt") or _val(bal, "noncurrent_liabilities"),
            "current_assets": _val(bal, "current_assets"),
            "current_liabilities": _val(bal, "current_liabilities"),
            "operating_cf": _val(cf, "net_cash_flow_from_operating_activities"),
            "capex": _val(cf, "net_cash_flow_from_investing_activities"),
            "dividends_paid": _val(cf, "payment_of_dividends"),
        })
    return parsed


def get_ticker_details(ticker: str) -> dict:
    """Get company details from Polygon reference API."""
    try:
        resp = requests.get(
            f"{POLYGON_BASE}/v3/reference/tickers/{ticker.upper()}",
            params={"apiKey": API_KEY},
            timeout=10,
        )
        if resp.status_code != 200:
            return {}
        data = resp.json().get("results", {})
        return {
            "ticker": ticker.upper(),
            "name": data.get("name"),
            "description": data.get("description"),
            "sector": data.get("sic_description"),
            "market_cap": data.get("market_cap"),
            "shares_outstanding": data.get("weighted_shares_outstanding") or data.get("share_class_shares_outstanding"),
            "homepage": data.get("homepage_url"),
            "primary_exchange": data.get("primary_exchange"),
            "sic_code": data.get("sic_code"),
        }
    except Exception:
        return {}


def get_dividends(ticker: str, limit: int = 8) -> list:
    """Get dividend history from Polygon."""
    try:
        resp = requests.get(
            f"{POLYGON_BASE}/v3/reference/dividends",
            params={"ticker": ticker.upper(), "limit": limit, "order": "desc", "apiKey": API_KEY},
            timeout=10,
        )
        if resp.status_code != 200:
            return []
        return resp.json().get("results", [])
    except Exception:
        return []
