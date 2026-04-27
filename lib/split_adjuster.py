"""Stock split adjuster using Polygon /v3/reference/splits.

For each split that occurred AFTER a trade date, multiply shares by
split_to/split_from and divide price by the same ratio.
"""
from __future__ import annotations

import os
import requests

POLYGON_BASE = "https://api.polygon.io"


def fetch_splits(ticker: str) -> list[dict]:
    """Fetch all splits for a ticker. Returns [] on failure or missing key."""
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        return []
    try:
        resp = requests.get(
            f"{POLYGON_BASE}/v3/reference/splits",
            params={
                "ticker": ticker.upper(),
                "limit": 1000,
                "apiKey": api_key,
            },
            timeout=10,
        )
        if resp.status_code != 200:
            return []
        return resp.json().get("results", []) or []
    except Exception as e:
        print(f"[split_adjuster] fetch_splits({ticker}) error: {e}")
        return []


def adjust_trade_for_splits(trade: dict, splits: list[dict]) -> dict:
    """Return a new trade dict with shares/price adjusted for splits AFTER the
    trade's executed_at. Does not mutate input. Sets is_split_adjusted=True
    only when at least one split was applied.
    """
    executed_at = trade.get("executed_at")
    shares = trade.get("shares")
    price = trade.get("price")
    if not executed_at or shares is None or price is None or not splits:
        return trade

    trade_date = str(executed_at)[:10]
    adj_shares = float(shares)
    adj_price = float(price)
    applied = 0

    for split in splits:
        ex_date = str(split.get("execution_date") or "")[:10]
        if not ex_date or ex_date <= trade_date:
            continue
        try:
            split_from = float(split.get("split_from") or 1)
            split_to = float(split.get("split_to") or 1)
        except (TypeError, ValueError):
            continue
        if split_from <= 0 or split_to <= 0:
            continue
        ratio = split_to / split_from
        adj_shares *= ratio
        adj_price /= ratio
        applied += 1

    if applied == 0:
        return trade

    return {
        **trade,
        "shares": adj_shares,
        "price": adj_price,
        "is_split_adjusted": True,
    }
