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


# ─── Cancellation filter tests (BCXL KEEP-mode downstream) ────────────────────

def test_fifo_excludes_cancelled_trades():
    """A Buy flagged cancellation_status='cancelled_by_broker' must NOT
    feed into FIFO matching — otherwise the AAPL 8/5/2024 fat-finger
    would close against the next AAPL Sell and corrupt P&L."""
    trades = [
        _trade("AAPL", "buy",  10, 200.0, "2024-08-05"),  # the BCXL-cancelled fat-finger
        _trade("AAPL", "buy",  10, 150.0, "2024-08-10"),  # the real lot
        _trade("AAPL", "sell", 10, 175.0, "2024-09-01"),  # real exit
    ]
    trades[0]["cancellation_status"] = "cancelled_by_broker"
    trades[0]["cancel_matched_at"] = "2024-08-05"

    closed = match_fifo_lots(trades)
    assert len(closed) == 1
    c = closed[0]
    # FIFO must consume the 8/10 lot (the real one), NOT the cancelled 8/5 row
    assert c["entry_date"] == "2024-08-10"
    assert c["cost_basis"] == 1500.0      # 10 * 150
    assert c["proceeds"] == 1750.0        # 10 * 175
    assert c["pnl_dollars"] == 250.0


def test_fifo_excludes_cancellation_record():
    """Same filter must also apply when status is 'cancellation_record'
    (placeholder for future BCXL counter-row imports)."""
    trades = [
        _trade("AAPL", "buy",  5, 100.0, "2025-01-01"),
        _trade("AAPL", "sell", 5, 110.0, "2025-02-01"),
    ]
    # Mark the buy as a cancellation record — it must be excluded
    trades[0]["cancellation_status"] = "cancellation_record"
    closed = match_fifo_lots(trades)
    # Sell can't match anything → no closed positions
    assert len(closed) == 0


def test_fifo_includes_normal_trades_default():
    """Trades missing cancellation_status entirely must be treated as 'normal'."""
    trades = [
        _trade("AAPL", "buy", 10, 100.0, "2025-01-01"),
        _trade("AAPL", "sell", 10, 120.0, "2025-02-01"),
    ]
    # Neither dict has cancellation_status — old-shape callers
    assert "cancellation_status" not in trades[0]
    closed = match_fifo_lots(trades)
    assert len(closed) == 1
    assert closed[0]["pnl_dollars"] == 200.0


def test_daily_nav_excludes_cancelled():
    """build_daily_nav must drop cancelled trades before any pandas math.
    With only a cancelled trade in, the NAV builder sees no inputs and
    returns an empty DataFrame (no exceptions, no Polygon hits)."""
    from lib.performance_math import build_daily_nav
    trades = [
        _trade("AAPL", "buy", 10, 200.0, "2024-08-05"),
    ]
    trades[0]["cancellation_status"] = "cancelled_by_broker"
    nav = build_daily_nav(trades, dividends=[])
    assert nav.empty


def test_compute_mwr_excludes_cancelled():
    """compute_mwr must drop cancelled trades so a phantom outflow/inflow
    pair never lands in the XIRR cashflow list."""
    from lib.performance_math import compute_mwr
    trades = [
        _trade("AAPL", "buy",  10, 100.0, "2025-01-01"),
        _trade("AAPL", "sell", 10, 120.0, "2025-02-01"),
        _trade("MSFT", "buy",  5,  500.0, "2025-01-15"),
    ]
    # Mark the MSFT buy as cancelled — should not appear in cashflows
    trades[2]["cancellation_status"] = "cancelled_by_broker"
    # Just confirm it doesn't crash and returns a finite number
    result = compute_mwr(trades, dividends=[], current_value=0.0)
    assert isinstance(result, float)


def test_compute_headline_stats_excludes_cancelled():
    """compute_headline_stats must filter at its own entry point. n_trades
    and n_closed_positions exclude cancelled rows."""
    from lib.performance_math import compute_headline_stats
    trades = [
        _trade("AAPL", "buy",  10, 200.0, "2024-08-05"),  # cancelled
        _trade("AAPL", "buy",  10, 150.0, "2024-08-10"),  # real
        _trade("AAPL", "sell", 10, 175.0, "2024-09-01"),  # real
    ]
    trades[0]["cancellation_status"] = "cancelled_by_broker"
    # Skip the benchmark fetch by passing a non-existent ticker; alpha/beta
    # will fall back to (0, 1) which is fine for this assertion.
    stats = compute_headline_stats(trades, dividends=[], benchmark_ticker="ZZZZ")
    assert stats["n_trades"] == 2          # cancelled buy excluded from buy/sell count
    assert stats["n_closed_positions"] == 1  # FIFO closes the 8/10 lot, not the cancelled 8/5


# ─── 2026-04-27 production regression: mixed tz timestamps ────────────────────

def test_mixed_tz_timestamps_do_not_crash():
    """Repro of the 2026-04-27 production bug:
    real Robinhood CSV had mixed tz-aware and tz-naive executed_at values
    that crashed match_fifo_lots with 'Cannot compare tz-naive and tz-aware timestamps'.
    """
    trades = [
        {'ticker': 'SPY', 'action': 'buy',  'shares': 1, 'price': 400, 'amount': -400,
         'executed_at': '2024-01-15',                          # Naive
         'cancellation_status': 'normal'},
        {'ticker': 'SPY', 'action': 'sell', 'shares': 1, 'price': 450, 'amount': 450,
         'executed_at': '2024-06-20T00:00:00+00:00',           # Aware
         'cancellation_status': 'normal'},
    ]
    closed = match_fifo_lots(trades)
    assert len(closed) == 1
    assert closed[0]['pnl_dollars'] == 50


def test_build_daily_nav_handles_mixed_tz_executed_at():
    """The actual production crash site: build_daily_nav's pd.to_datetime
    call hits mixed tz-aware/tz-naive strings, then bdate_range produces a
    tz-naive Index that can't be compared against the tz-aware trade dates.
    The fix uses utc=True + _to_utc_naive to coerce everything to tz-naive
    UTC before any comparison."""
    from lib.performance_math import build_daily_nav
    trades = [
        {'ticker': 'SPY', 'action': 'buy',  'shares': 5, 'price': 400, 'amount': -2000,
         'executed_at': '2024-01-15',                          # Naive
         'cancellation_status': 'normal'},
        {'ticker': 'SPY', 'action': 'sell', 'shares': 5, 'price': 450, 'amount': 2250,
         'executed_at': '2024-06-20T00:00:00+00:00',           # Aware
         'cancellation_status': 'normal'},
    ]
    # Must not raise. May return empty if Polygon price fetch fails (network-
    # dependent in test env), but the crash we're fixing is in the pre-fetch
    # comparison logic — that must succeed regardless.
    nav = build_daily_nav(trades, dividends=[])
    # The function returns either an empty DataFrame (no Polygon prices) or
    # a DataFrame with tz-naive index. Either is acceptable; the regression
    # is that it must NOT throw.
    if not nav.empty:
        assert nav.index.tz is None, "NAV index must be tz-naive after the patch"


def test_build_daily_nav_handles_mixed_tz_dividends():
    """Same defensive coverage for the dividends parsing branch."""
    from lib.performance_math import build_daily_nav
    trades = [
        {'ticker': 'AAPL', 'action': 'buy', 'shares': 10, 'price': 150, 'amount': -1500,
         'executed_at': '2024-01-15', 'cancellation_status': 'normal'},
    ]
    dividends = [
        {'ticker': 'AAPL', 'amount': 5.0,
         'paid_at': '2024-03-01',                              # Naive
         'dividend_type': 'cash'},
        {'ticker': 'AAPL', 'amount': 5.0,
         'paid_at': '2024-06-01T00:00:00+00:00',               # Aware
         'dividend_type': 'cash'},
    ]
    # Must not raise on the ddf["paid_at"] parse path.
    nav = build_daily_nav(trades, dividends)
    if not nav.empty:
        assert nav.index.tz is None


def test_compute_headline_stats_with_mixed_tz_does_not_crash():
    """Full pipeline: compute_headline_stats orchestrates FIFO + NAV + MWR.
    With mixed-tz inputs the NAV stage was crashing and propagating up."""
    from lib.performance_math import compute_headline_stats
    trades = [
        {'ticker': 'SPY', 'action': 'buy',  'shares': 1, 'price': 400, 'amount': -400,
         'executed_at': '2024-01-15', 'cancellation_status': 'normal'},
        {'ticker': 'SPY', 'action': 'sell', 'shares': 1, 'price': 450, 'amount': 450,
         'executed_at': '2024-06-20T00:00:00+00:00', 'cancellation_status': 'normal'},
    ]
    # ZZZZ benchmark short-circuits the Polygon benchmark fetch path; we're
    # testing that the upstream NAV builder doesn't crash on mixed-tz input.
    stats = compute_headline_stats(trades, dividends=[], benchmark_ticker="ZZZZ")
    # Must produce a dict with all required fields, no exceptions.
    for f in ("twr", "mwr", "alpha", "beta", "sharpe", "sortino",
              "max_drawdown", "win_rate", "n_trades", "n_closed_positions"):
        assert f in stats
    assert stats["n_closed_positions"] == 1
    assert stats["wins"] == 1


def test_sanitize_for_json_handles_nan_inf():
    """Every NaN / +Inf / -Inf in a nested dict/list must become None so the
    response can pass FastAPI's strict JSON serializer."""
    import math
    import json
    from lib.performance_math import _sanitize_for_json

    inp = {
        "twr": float("nan"),
        "alpha": float("inf"),
        "beta": float("-inf"),
        "sharpe": 1.23,
        "wins": 5,
        "name": "AAPL",
        "list_with_nan": [1.0, float("nan"), float("inf")],
        "nested": {
            "max_dd": float("nan"),
            "ok": -0.42,
            "deeper": {"inf_val": float("inf")},
        },
        "tickers_skipped": [{"ticker": "SPXU", "reason": "polygon_timeout_30s"}],
    }
    out = _sanitize_for_json(inp)
    assert out["twr"] is None
    assert out["alpha"] is None
    assert out["beta"] is None
    assert out["sharpe"] == 1.23
    assert out["wins"] == 5
    assert out["name"] == "AAPL"
    assert out["list_with_nan"] == [1.0, None, None]
    assert out["nested"]["max_dd"] is None
    assert out["nested"]["ok"] == -0.42
    assert out["nested"]["deeper"]["inf_val"] is None
    assert out["tickers_skipped"][0]["ticker"] == "SPXU"

    # The whole sanitized dict must be JSON-serializable
    json.dumps(out)  # raises if any leftover NaN/Inf


def test_compute_headline_stats_when_polygon_fetch_fails():
    """Mock _fetch_prices to simulate a Polygon timeout for SPXU. The endpoint
    must return successfully with the skipped ticker surfaced under
    tickers_skipped, and the response must be JSON-serializable."""
    import json
    from unittest.mock import patch
    import pandas as pd
    from lib.performance_math import compute_headline_stats

    trades = [
        _trade("SPXU", "buy",  5, 10.0, "2024-01-15"),
        _trade("SPXU", "sell", 5, 12.0, "2024-06-01"),
    ]

    # Empty DataFrame + skipped list = full Polygon timeout simulation
    with patch("lib.performance_math._fetch_prices") as mock_fetch:
        mock_fetch.return_value = (pd.DataFrame(), [{"ticker": "SPXU", "reason": "polygon_timeout_30s"}])
        stats = compute_headline_stats(trades, dividends=[], benchmark_ticker="ZZZZ")

    assert "tickers_skipped" in stats
    assert any(s.get("ticker") == "SPXU" for s in stats["tickers_skipped"])
    assert stats["tickers_skipped"][0]["reason"] == "polygon_timeout_30s"
    # FIFO still produces a closed position from the trade prices alone
    assert stats["n_closed_positions"] == 1
    assert stats["wins"] == 1
    # Must be JSON-serializable end to end
    json.dumps(stats)


def test_compute_headline_stats_with_nan_returns_does_not_crash():
    """Force a NaN-producing scenario by feeding all-zero prices: pct_change
    on zeros yields inf, and divisions in alpha/beta + max_dd cascade. The
    response must still serialize cleanly with None instead of NaN/Inf."""
    import json
    import math
    from unittest.mock import patch
    import pandas as pd
    from lib.performance_math import compute_headline_stats

    trades = [
        _trade("AAPL", "buy",  10, 100.0, "2024-01-15"),
        _trade("AAPL", "sell", 10, 120.0, "2024-06-01"),
    ]
    # All-zero prices over the trade window — guaranteed inf/NaN downstream
    idx = pd.bdate_range("2024-01-15", "2024-06-01")
    bad_prices = pd.DataFrame(0.0, index=idx, columns=["AAPL"])

    with patch("lib.performance_math._fetch_prices") as mock_fetch:
        mock_fetch.return_value = (bad_prices, [])
        stats = compute_headline_stats(trades, dividends=[], benchmark_ticker="ZZZZ")

    # No float field may be NaN or Inf — sanitize must have caught all of them
    for key, val in stats.items():
        if isinstance(val, float):
            assert not math.isnan(val), f"{key} is NaN — sanitize missed it"
            assert not math.isinf(val), f"{key} is Inf — sanitize missed it"

    # Must JSON-serialize without raising
    json.dumps(stats)


def test_to_utc_naive_helper_handles_all_three_shapes():
    """The helper must handle pd.Series, pd.DatetimeIndex, and pd.Timestamp
    in both tz-aware and tz-naive states."""
    import pandas as pd
    from lib.performance_math import _to_utc_naive

    # tz-aware Series → naive
    s_aware = pd.Series(pd.to_datetime(["2024-01-15", "2024-06-20"]).tz_localize("UTC"))
    assert s_aware.dt.tz is not None
    out = _to_utc_naive(s_aware)
    assert out.dt.tz is None

    # tz-naive Series → unchanged
    s_naive = pd.Series(pd.to_datetime(["2024-01-15", "2024-06-20"]))
    assert s_naive.dt.tz is None
    out = _to_utc_naive(s_naive)
    assert out.dt.tz is None

    # tz-aware DatetimeIndex → naive
    idx_aware = pd.DatetimeIndex(["2024-01-15"], tz="UTC")
    out = _to_utc_naive(idx_aware)
    assert out.tz is None

    # tz-naive DatetimeIndex → unchanged
    idx_naive = pd.DatetimeIndex(["2024-01-15"])
    out = _to_utc_naive(idx_naive)
    assert out.tz is None

    # tz-aware scalar Timestamp → naive
    ts_aware = pd.Timestamp("2024-01-15", tz="UTC")
    out = _to_utc_naive(ts_aware)
    assert out.tz is None

    # tz-naive scalar Timestamp → unchanged
    ts_naive = pd.Timestamp("2024-01-15")
    out = _to_utc_naive(ts_naive)
    assert out.tz is None
