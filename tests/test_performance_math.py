"""Unit tests for the performance attribution math.

Run with: python -m pytest tests/test_performance_math.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.performance_math import (
    match_fifo_lots,
    bucket_by_holding_period,
    bucket_by_sector,
    bucket_by_year,
)


def _trade(ticker, action, shares, price, executed_at, amount=None, tid=None):
    if amount is None:
        amount = shares * price * (-1 if action == "buy" else 1)
    return {
        "id": tid,
        "ticker": ticker,
        "action": action,
        "shares": shares,
        "price": price,
        "amount": amount,
        "executed_at": executed_at,
    }


def test_winning_round_trip():
    trades = [
        _trade("AAPL", "buy", 10, 100.0, "2025-01-01", tid="t1"),
        _trade("AAPL", "sell", 10, 120.0, "2025-02-01", tid="t2"),
    ]
    closed = match_fifo_lots(trades)
    assert len(closed) == 1
    c = closed[0]
    assert c["pnl_dollars"] == 200.0
    assert c["pnl_pct"] == 20.0
    assert c["holding_period_days"] == 31
    assert c["is_long_term"] is False
    assert c["entry_trade_id"] == "t1"
    assert c["exit_trade_id"] == "t2"


def test_losing_round_trip():
    trades = [
        _trade("MSFT", "buy", 5, 200.0, "2025-01-01"),
        _trade("MSFT", "sell", 5, 180.0, "2025-01-15"),
    ]
    closed = match_fifo_lots(trades)
    assert len(closed) == 1
    assert closed[0]["pnl_dollars"] == -100.0
    assert closed[0]["pnl_pct"] == -10.0


def test_fifo_partial_match():
    """Buy → buy → sell of half: only the first lot closes (partially)."""
    trades = [
        _trade("AAPL", "buy", 10, 100.0, "2025-01-01"),
        _trade("AAPL", "buy", 10, 110.0, "2025-01-15"),
        _trade("AAPL", "sell", 5, 120.0, "2025-02-01"),
    ]
    closed = match_fifo_lots(trades)
    # 5 shares from the FIRST (older) lot at $100
    assert len(closed) == 1
    assert closed[0]["shares"] == 5.0
    assert closed[0]["entry_date"] == "2025-01-01"
    assert closed[0]["pnl_dollars"] == 100.0  # (120-100)*5


def test_fifo_sell_spans_two_lots():
    trades = [
        _trade("AAPL", "buy", 6, 100.0, "2025-01-01"),
        _trade("AAPL", "buy", 6, 110.0, "2025-01-15"),
        _trade("AAPL", "sell", 10, 120.0, "2025-02-01"),
    ]
    closed = match_fifo_lots(trades)
    # First chunk: 6 shares from lot 1; second chunk: 4 shares from lot 2
    assert len(closed) == 2
    chunks = sorted(closed, key=lambda c: c["entry_date"])
    assert chunks[0]["shares"] == 6.0 and chunks[0]["pnl_dollars"] == 120.0  # (120-100)*6
    assert chunks[1]["shares"] == 4.0 and chunks[1]["pnl_dollars"] == 40.0   # (120-110)*4


def test_short_term_vs_long_term():
    short = match_fifo_lots([
        _trade("X", "buy", 1, 1.0, "2025-01-01"),
        _trade("X", "sell", 1, 2.0, "2025-01-31"),  # 30 days
    ])
    long_ = match_fifo_lots([
        _trade("Y", "buy", 1, 1.0, "2024-01-01"),
        _trade("Y", "sell", 1, 2.0, "2025-01-02"),  # 367 days
    ])
    assert short[0]["is_long_term"] is False
    assert short[0]["holding_period_days"] == 30
    assert long_[0]["is_long_term"] is True
    assert long_[0]["holding_period_days"] == 367


def test_holding_period_buckets():
    closed = [
        {"holding_period_days": 0, "pnl_dollars": 5, "pnl_pct": 1.0, "ticker": "A"},
        {"holding_period_days": 5, "pnl_dollars": 10, "pnl_pct": 2.0, "ticker": "B"},
        {"holding_period_days": 30, "pnl_dollars": 15, "pnl_pct": 3.0, "ticker": "C"},
        {"holding_period_days": 400, "pnl_dollars": 100, "pnl_pct": 10.0, "ticker": "D"},
    ]
    buckets = bucket_by_holding_period(closed)
    assert buckets["intraday"]["count"] == 1
    assert buckets["<1w"]["count"] == 1
    assert buckets["1-3m"]["count"] == 1
    assert buckets[">1y"]["count"] == 1
    assert buckets["1-4w"]["count"] == 0


def test_sector_bucket():
    closed = [
        {"ticker": "AAPL", "pnl_dollars": 100, "pnl_pct": 10.0},
        {"ticker": "MSFT", "pnl_dollars": 50, "pnl_pct": 5.0},
        {"ticker": "JPM",  "pnl_dollars": -20, "pnl_pct": -2.0},
    ]
    sectors = {"AAPL": "Tech", "MSFT": "Tech", "JPM": "Financials"}
    out = bucket_by_sector(closed, sectors)
    assert out["Tech"]["count"] == 2
    assert out["Tech"]["total_pnl"] == 150.0
    assert out["Financials"]["count"] == 1


def test_year_bucket():
    closed = [
        {"exit_date": "2024-06-01", "pnl_dollars": 10, "pnl_pct": 1.0},
        {"exit_date": "2025-01-15", "pnl_dollars": 20, "pnl_pct": 2.0},
        {"exit_date": "2025-08-01", "pnl_dollars": 30, "pnl_pct": 3.0},
    ]
    out = bucket_by_year(closed)
    assert out["2024"]["count"] == 1
    assert out["2025"]["count"] == 2
    assert out["2025"]["total_pnl"] == 50.0
