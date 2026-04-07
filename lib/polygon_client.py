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
    """Fetch financial data from Polygon financials API."""
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
