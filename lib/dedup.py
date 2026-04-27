"""Dedup hash for trades and dividends.

Pipe-joined sha256 of the per-row natural key. Guarantees the same trade
parsed twice produces the same hash, so the (user_id, dedup_hash) UNIQUE
constraint catches re-imports of the same CSV.
"""
from __future__ import annotations

import hashlib


def compute_dedup_hash(
    user_id: str,
    ticker: str | None,
    action: str | None,
    shares,
    price,
    executed_at: str | None,
) -> str:
    parts = [
        str(user_id or ""),
        str(ticker or "").upper(),
        str(action or ""),
        str(shares if shares is not None else ""),
        str(price if price is not None else ""),
        str(executed_at or ""),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def compute_dividend_dedup_hash(
    user_id: str,
    ticker: str | None,
    amount,
    paid_at: str | None,
    dividend_type: str | None,
) -> str:
    parts = [
        str(user_id or ""),
        str(ticker or "").upper(),
        str(amount if amount is not None else ""),
        str(paid_at or ""),
        str(dividend_type or "cash"),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
