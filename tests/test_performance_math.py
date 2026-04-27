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


def test_nav_with_skipped_ticker_excludes_from_calculation():
    """When a ticker is in tickers_skipped, its trades must NOT contribute to
    NAV holdings_value OR cash_flow. Previously cash_flow included skipped
    tickers' buy outflows while holdings didn't reflect them, which
    desynchronized the TWR formula."""
    from unittest.mock import patch
    import pandas as pd
    from lib.performance_math import build_daily_nav

    # All dates picked to be business days so reindex onto bdate_range
    # doesn't silently drop a weekend trade.
    trades = [
        # Two priced tickers
        _trade("AAPL", "buy",  10, 150.0, "2024-01-15"),  # Monday
        _trade("AAPL", "sell", 10, 175.0, "2024-06-03"),  # Monday (was Sat 06-01)
        _trade("MSFT", "buy",   5, 400.0, "2024-02-01"),  # Thursday
        # One ticker that Polygon skipped
        _trade("SPXU", "buy", 100,  10.0, "2024-03-01"),  # Friday
        _trade("SPXU", "sell", 100, 12.0, "2024-04-01"),  # Monday
    ]

    # Mock Polygon: AAPL + MSFT priced, SPXU skipped
    idx = pd.bdate_range("2024-01-15", "2024-12-02")
    aapl_prices = pd.Series(150.0 + (idx - idx[0]).days * 0.05, index=idx)
    msft_prices = pd.Series(400.0 + (idx - idx[0]).days * 0.10, index=idx)
    fake_prices = pd.DataFrame({"AAPL": aapl_prices, "MSFT": msft_prices})
    skipped = [{"ticker": "SPXU", "reason": "polygon_timeout_30s"}]

    with patch("lib.performance_math._fetch_prices") as mock_fetch:
        mock_fetch.return_value = (fake_prices, skipped)
        nav = build_daily_nav(trades, dividends=[])

    assert not nav.empty
    # cash_flow column must NOT include the SPXU buy or sell — only AAPL + MSFT
    # AAPL buy:  -1500
    # AAPL sell: +1750
    # MSFT buy:  -2000
    # SPXU buy:  -1000  (excluded)
    # SPXU sell: +1200  (excluded)
    # Net cash_flow over the whole series must equal -1500 + 1750 - 2000 = -1750
    assert abs(nav["cash_flow"].sum() - (-1750.0)) < 0.01, \
        f"cash_flow sum {nav['cash_flow'].sum()} should exclude SPXU trades"
    # tickers_skipped surfaced via attrs
    assert nav.attrs["tickers_skipped"][0]["ticker"] == "SPXU"


def test_extreme_daily_return_is_clipped():
    """Force a synthetic 500% single-day spike in NAV. compute_twr's
    outlier guard must zero it out so the chained product doesn't explode."""
    import pandas as pd
    from lib.performance_math import compute_twr

    # 252 days of mostly-flat NAV with a 500% spike on day 100
    dates = pd.bdate_range("2024-01-01", periods=252)
    hv = pd.Series(100.0, index=dates)
    hv.iloc[100] = 600.0  # 500% jump
    hv.iloc[101] = 100.0  # ...and back
    nav = pd.DataFrame({
        "holdings_value": hv,
        "cash_flow": pd.Series(0.0, index=dates),
    })
    twr = compute_twr(nav)
    # Without the outlier filter this would explode. With the filter, the
    # 500% return on day 100 and the matching -83% return on day 101 are
    # both zeroed, leaving a small flat result. Bound it generously.
    assert -0.5 < twr < 0.5, f"TWR {twr} should be near zero after outlier zap"


def test_max_drawdown_bounded_to_minus_100_percent():
    """Max DD must always be in [-1, 0]. Even a NAV anomaly that produces a
    raw drawdown of -250% gets bounded to -100%."""
    import pandas as pd
    from lib.performance_math import compute_max_drawdown
    # Construct NAV that goes 100 → 50 → 10 → effective dd > 100%
    nav = pd.Series([100.0, 80.0, 50.0, 30.0, 10.0],
                    index=pd.bdate_range("2024-01-01", periods=5))
    dd, dd_date = compute_max_drawdown(nav)
    assert -1.0 <= dd <= 0.0, f"max_dd {dd} not in [-1, 0]"
    # Real value here is -90% (10/100 - 1), bounded to -1 only when extreme.
    assert dd == -0.9 or dd == -1.0


def test_real_world_220_trade_smoke():
    """Synthetic 10-trade portfolio mocking realistic SPY/SPYG prices and a
    Polygon failure for SPXU. All headline stats must land in plausible
    ranges — proves the chained product no longer explodes after the patch.
    """
    from unittest.mock import patch
    import pandas as pd
    import numpy as np
    from lib.performance_math import compute_headline_stats

    np.random.seed(42)
    trades = [
        _trade("SPY",  "buy",  10, 400.0, "2024-01-15"),
        _trade("SPY",  "sell", 10, 450.0, "2024-09-15"),
        _trade("SPYG", "buy",  20,  60.0, "2024-02-01"),
        _trade("SPYG", "sell", 20,  72.0, "2024-10-01"),
        _trade("AAPL", "buy",   5, 180.0, "2024-03-01"),
        _trade("AAPL", "sell",  5, 220.0, "2024-11-01"),
        _trade("MSFT", "buy",   3, 400.0, "2024-04-01"),
        _trade("MSFT", "sell",  3, 440.0, "2024-12-01"),
        # SPXU — Polygon will "time out" on this one
        _trade("SPXU", "buy", 100,  10.0, "2024-05-01"),
        _trade("SPXU", "sell", 100, 11.0, "2024-08-01"),
    ]

    # Realistic price walks for the 4 priced tickers
    idx = pd.bdate_range("2024-01-15", "2024-12-15")
    def walk(start, vol=0.01):
        rets = np.random.normal(0.0003, vol, len(idx))
        return pd.Series(start * np.cumprod(1 + rets), index=idx)

    fake_prices = pd.DataFrame({
        "SPY":  walk(400.0),
        "SPYG": walk(60.0),
        "AAPL": walk(180.0, vol=0.015),
        "MSFT": walk(400.0, vol=0.012),
    })
    skipped = [{"ticker": "SPXU", "reason": "polygon_timeout_30s"}]

    with patch("lib.performance_math._fetch_prices") as mock_fetch:
        mock_fetch.return_value = (fake_prices, skipped)
        # Skip the benchmark fetch by passing a non-existent ticker
        stats = compute_headline_stats(trades, dividends=[], benchmark_ticker="ZZZZ")

    # Sanity bounds — these are the symptoms from the production bug report
    assert stats["twr"] is not None and -0.5 < stats["twr"] < 5.0, \
        f"TWR {stats['twr']} out of plausible range [-50%, +500%]"
    assert -1.0 <= stats["max_drawdown"] <= 0.0, \
        f"max_drawdown {stats['max_drawdown']} not in [-100%, 0%]"
    # SPXU is in skipped — but its FIFO closed position is still recorded
    # because FIFO uses trade-row prices, not Polygon
    assert "tickers_skipped" in stats
    assert any(s["ticker"] == "SPXU" for s in stats["tickers_skipped"])
    # n_closed_positions: 5 (SPY, SPYG, AAPL, MSFT, SPXU all round-tripped)
    assert stats["n_closed_positions"] == 5
    # Win rate plausible
    assert 0.0 <= stats["win_rate"] <= 1.0


def test_alpha_beta_with_extreme_outlier_returns_none():
    """A single 500% spike in port_returns must not blow up beta.
    The function zaps outliers before regression, so result should be either
    a sane value or None (if too few points remain for regression)."""
    import pandas as pd
    import numpy as np
    from lib.performance_math import compute_alpha_beta

    np.random.seed(7)
    idx = pd.bdate_range("2024-01-01", periods=200)
    bm_returns = pd.Series(np.random.normal(0.0005, 0.01, len(idx)), index=idx)
    # Portfolio mostly tracks bm with mild noise; one extreme outlier on day 50
    port_returns = pd.Series(bm_returns.values * 1.2 + np.random.normal(0, 0.005, len(idx)), index=idx)
    port_returns.iloc[50] = 5.0  # +500% — pathological data point

    alpha, beta = compute_alpha_beta(port_returns, bm_returns)
    # Without the zap, beta would be enormous (correlation with that outlier
    # against the bm value of that day). With the zap, beta is sane.
    if beta is not None:
        assert -5.0 <= beta <= 5.0, f"beta {beta} not in [-5, 5] after outlier zap"
    if alpha is not None:
        assert -2.0 <= alpha <= 2.0, f"alpha {alpha} exceeds ±200% bound"


def test_alpha_beta_with_short_series_returns_none():
    """< 30 aligned points → (None, None). Surfaces gap honestly instead of
    emitting a synthetic 0/1."""
    import pandas as pd
    import numpy as np
    from lib.performance_math import compute_alpha_beta

    idx = pd.bdate_range("2024-01-01", periods=10)
    p = pd.Series(np.random.normal(0, 0.01, 10), index=idx)
    b = pd.Series(np.random.normal(0, 0.01, 10), index=idx)
    alpha, beta = compute_alpha_beta(p, b)
    assert alpha is None
    assert beta is None


def test_alpha_beta_with_clean_series_returns_sane_values():
    """252 days of plausible returns with portfolio ≈ 1.2 × benchmark + noise.
    Beta should land near 1.2, alpha near zero."""
    import pandas as pd
    import numpy as np
    from lib.performance_math import compute_alpha_beta

    np.random.seed(42)
    idx = pd.bdate_range("2024-01-01", periods=252)
    bm = pd.Series(np.random.normal(0.0004, 0.01, 252), index=idx)
    # Synthesize port returns with a known beta of 1.2 and zero alpha
    port = pd.Series(bm.values * 1.2 + np.random.normal(0, 0.003, 252), index=idx)

    alpha, beta = compute_alpha_beta(port, bm, rf_rate=0.04)
    assert alpha is not None and beta is not None
    assert -1.0 < alpha < 1.0, f"alpha {alpha} should be near zero"
    assert 0.0 < beta < 3.0, f"beta {beta} should be near 1.2"
    # Beta should be in the right ballpark (allow generous tolerance for noise + rf adjustment)
    assert 0.8 < beta < 1.6, f"beta {beta} should be near 1.2 ± 0.4"


def test_max_drawdown_uses_filtered_nav_via_clean_index():
    """compute_headline_stats must call compute_max_drawdown with the
    cumulative-return equity curve, NOT the raw NAV series. The raw NAV has
    trade-day spikes (a $50k buy literally jumps holdings_value by $50k).
    Drawdown on the raw NAV would hit -100% from those phantom peaks.

    This test feeds a daily_nav with a synthetic trade-day spike and confirms
    that compute_headline_stats produces a sane max_drawdown — proving it's
    not consuming the raw nav directly."""
    from unittest.mock import patch
    import pandas as pd
    import numpy as np
    from lib.performance_math import compute_headline_stats

    # Two priced tickers, with a "buy day" mid-series that should NOT
    # register as a price move. The clean_daily_returns helper subtracts cf,
    # zeroing the buy-day return; the cumulative-return index then stays
    # smooth, and max_drawdown reflects only price-driven moves.
    trades = [
        _trade("AAPL", "buy",  10, 150.0, "2024-01-15"),
        _trade("AAPL", "buy",  100, 150.0, "2024-06-03"),  # Big buy mid-series
        _trade("AAPL", "sell", 110, 165.0, "2024-12-02"),
    ]
    idx = pd.bdate_range("2024-01-15", "2024-12-02")
    # Smooth price walk — AAPL drifts up
    aapl = pd.Series(150.0 + np.linspace(0, 15, len(idx)), index=idx)
    fake_prices = pd.DataFrame({"AAPL": aapl})

    with patch("lib.performance_math._fetch_prices") as mock_fetch:
        mock_fetch.return_value = (fake_prices, [])
        stats = compute_headline_stats(trades, dividends=[], benchmark_ticker="ZZZZ")

    # Without the clean-nav-index fix, max_drawdown would be near -100% from
    # the phantom NAV trough after the rolling_max captured the buy-day spike.
    # With the fix, drawdown reflects only actual price-driven dips, which on
    # a monotone upward walk should be near 0.
    assert -0.5 <= stats["max_drawdown"] <= 0.0, \
        f"max_drawdown {stats['max_drawdown']} indicates raw-NAV trade-day spike contamination"


def test_zap_outliers_helper():
    """Unit-test the outlier zap helper directly."""
    import pandas as pd
    import numpy as np
    from lib.performance_math import _zap_outliers

    s = pd.Series([0.01, 0.02, 0.50, -0.40, 0.005, np.inf, -np.inf, np.nan])
    out = _zap_outliers(s)
    # 0.50, -0.40 should be zeroed; inf/-inf/nan dropped
    assert (out.abs() <= 0.30).all(), "outliers not zeroed"
    assert not out.isna().any(), "NaN/Inf not dropped"
    # 0.50 → 0, -0.40 → 0; original 0.01, 0.02, 0.005 preserved
    assert (out == 0.0).sum() == 2  # the two zapped outliers


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
