"""Sprint 9C.4 tests — options layer in performance math.

Run: python -m pytest tests/test_performance_math_options.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.performance_math import (
    build_options_cash_flows,
    compute_headline_stats,
)


# Equity fixture — minimal portfolio so combined-scope has something to
# accrete options against.
def _eq(action, ticker, shares, price, executed_at, amount=None):
    if amount is None:
        # Robinhood convention: buy = negative cash, sell = positive
        amount = -shares * price if action == "buy" else shares * price
    return {
        "action": action,
        "ticker": ticker,
        "shares": shares,
        "price": price,
        "amount": amount,
        "fees": 0.0,
        "executed_at": executed_at,
        "asset_type": "equity",
        "is_split_adjusted": False,
        "cancellation_status": "normal",
    }


def _opt(trans_code, ticker, strike, opt_type, contracts, total_amount,
         executed_at, expiration="2026-06-19"):
    """Build a parsed options-trade row in the same shape lib/options_fifo expects."""
    return {
        "underlying_ticker": ticker,
        "expiration_date": expiration,
        "strike": float(strike),
        "option_type": opt_type,
        "trans_code": trans_code,
        "contracts": float(contracts),
        "premium_per_contract": (
            abs(total_amount) / contracts / 100 if total_amount and contracts else None
        ),
        "total_amount": float(total_amount) if total_amount is not None else None,
        "executed_at": executed_at,
        "settled_at": executed_at,
        "cancellation_status": "normal",
    }


# ─── build_options_cash_flows ────────────────────────────────────────────

def test_options_cash_flows_basic():
    """BTO emits negative, STC positive, OEXP zero (no row), CONV skipped."""
    trades = [
        _opt("BTO", "AAPL", 200, "call", 1, -500.0, "2026-01-15"),
        _opt("STC", "AAPL", 200, "call", 1, 700.0, "2026-02-01"),
        _opt("OEXP", "TSLA", 100, "put", 1, None, "2026-03-15"),
        _opt("CONV", "PZZA", 54, "put", 1, None, "2018-11-09"),
    ]
    out = build_options_cash_flows(trades)
    # OEXP and CONV both excluded
    assert len(out) == 2
    # Sorted by date ascending
    assert out[0][0] < out[1][0]
    # Sign convention enforced
    assert out[0][1] == -500.0  # BTO
    assert out[1][1] == 700.0   # STC


def test_options_cash_flows_empty_input():
    assert build_options_cash_flows([]) == []
    assert build_options_cash_flows(None) == []


# ─── compute_headline_stats — scope routing ─────────────────────────────

def test_options_only_scope_basic():
    """3 options trades (BTO, STC, OEXP) → headline stats reflect options only."""
    options = [
        _opt("BTO", "AAPL", 200, "call", 1, -500.0, "2025-01-15"),
        _opt("STC", "AAPL", 200, "call", 1, 700.0, "2025-02-01"),
        # An expired-worthless long put — full premium loss on close
        _opt("BTO", "TSLA", 100, "put", 1, -300.0, "2025-01-10",
             expiration="2025-03-15"),
        _opt("OEXP", "TSLA", 100, "put", 1, None, "2025-03-15",
             expiration="2025-03-15"),
    ]
    result = compute_headline_stats(
        trades=[],
        dividends=[],
        options_trades=options,
        scope="options",
    )
    assert result["scope"] == "options"
    # n_options_trades = real (non-CONV, non-cancelled) options. OEXP counts.
    assert result["n_options_trades"] == 4
    # Closed positions: AAPL profitable + TSLA expired
    assert result["n_options_closed"] == 2
    assert result["n_equity_closed"] == 0
    # Realized P&L: AAPL +$200, TSLA -$300 → -$100
    assert abs(result["total_realized_pnl"] - (-100.0)) < 0.01
    # Win rate: 1 winner / 2 closed = 50%
    assert abs(result["win_rate"] - 0.5) < 0.01


def test_combined_scope_aggregates():
    """Equity + options both feed the same NAV time series."""
    eq_trades = [
        _eq("buy",  "SPY", 10, 400.0, "2025-01-15"),
        _eq("sell", "SPY", 10, 420.0, "2025-02-15"),
    ]
    options = [
        _opt("BTO", "AAPL", 200, "call", 1, -500.0, "2025-01-20"),
        _opt("STC", "AAPL", 200, "call", 1, 700.0, "2025-02-10"),
    ]
    result = compute_headline_stats(
        trades=eq_trades,
        dividends=[],
        options_trades=options,
        scope="combined",
    )
    assert result["scope"] == "combined"
    # Both sides accounted for in trade counts
    assert result["n_equity_trades"] == 2
    assert result["n_options_trades"] == 2
    # Closed: 1 equity (SPY round trip) + 1 options (AAPL) = 2
    assert result["n_closed_positions"] == 2
    assert result["n_equity_closed"] == 1
    assert result["n_options_closed"] == 1
    # Combined P&L: SPY +$200, AAPL +$200 → +$400
    assert abs(result["total_realized_pnl"] - 400.0) < 0.5


def test_equity_only_scope_excludes_options():
    """scope='equity' ignores options_trades even if passed."""
    eq_trades = [
        _eq("buy",  "SPY", 10, 400.0, "2025-01-15"),
        _eq("sell", "SPY", 10, 420.0, "2025-02-15"),
    ]
    options = [
        _opt("BTO", "AAPL", 200, "call", 1, -500.0, "2025-01-20"),
        _opt("STC", "AAPL", 200, "call", 1, 700.0, "2025-02-10"),
    ]
    result = compute_headline_stats(
        trades=eq_trades,
        dividends=[],
        options_trades=options,
        scope="equity",
    )
    assert result["scope"] == "equity"
    assert result["n_options_trades"] == 0
    assert result["n_options_closed"] == 0
    assert result["n_equity_closed"] == 1
    # Only SPY $200 P&L — options ignored
    assert abs(result["total_realized_pnl"] - 200.0) < 0.5


def test_options_with_no_trades():
    """scope='options' with empty options_trades → headline stats 400s
    at the route level, but compute_headline_stats handles []
    gracefully when called directly (used by tests)."""
    # Headline stats called directly with empty options list — should
    # produce zero counts, not crash. The route layer rejects this
    # input with a 400 before it gets here, but the math layer should
    # be defensive.
    result = compute_headline_stats(
        trades=[],
        dividends=[],
        options_trades=[],
        scope="options",
    )
    assert result["scope"] == "options"
    assert result["n_options_trades"] == 0
    assert result["n_closed_positions"] == 0
    assert result["total_realized_pnl"] == 0


def test_conv_excluded_from_cash_flows_and_pnl():
    """CONV rows are skipped — no cash flow, no closed-position P&L
    contribution (the placeholder records carry realized_pnl=0 but we
    still filter outcome='conversion_unhandled' to avoid noise in
    aggregations like best_trade / worst_trade)."""
    options = [
        _opt("BTO", "AAPL", 200, "call", 1, -500.0, "2025-01-15"),
        _opt("STC", "AAPL", 200, "call", 1, 700.0, "2025-02-01"),
        _opt("CONV", "PZZA", 54, "put", 1, None, "2025-03-15"),
    ]
    result = compute_headline_stats(
        trades=[],
        dividends=[],
        options_trades=options,
        scope="options",
    )
    # n_options_trades excludes CONV
    assert result["n_options_trades"] == 2
    # Only the AAPL closed_position is counted
    assert result["n_options_closed"] == 1
    # P&L is just the AAPL +$200 — CONV contributes 0 (filtered)
    assert abs(result["total_realized_pnl"] - 200.0) < 0.01

    # Cash flow helper also skips CONV rows entirely
    cfs = build_options_cash_flows(options)
    assert len(cfs) == 2  # BTO, STC only — no CONV row


def test_options_with_tz_aware_executed_at():
    """Bug 1 fix: Supabase returns executed_at as TIMESTAMPTZ (ISO with
    timezone offset). The options helpers must handle tz-aware strings
    without raising "Cannot compare tz-naive and tz-aware timestamps".
    """
    options = [
        # Mixed: some with explicit Z suffix, some with offset, some naive.
        _opt("BTO", "AAPL", 200, "call", 1, -500.0, "2025-01-15T14:30:00Z"),
        _opt("STC", "AAPL", 200, "call", 1, 700.0, "2025-02-01T18:00:00+00:00"),
        _opt("BTO", "TSLA", 100, "put", 1, -300.0, "2025-01-10T09:30:00-05:00",
             expiration="2025-03-15"),
        _opt("OEXP", "TSLA", 100, "put", 1, None, "2025-03-15",
             expiration="2025-03-15"),  # naive
    ]
    # Should not raise on scope='options' (which routes through
    # _build_options_only_nav, the function that broke in production).
    result = compute_headline_stats(
        trades=[],
        dividends=[],
        options_trades=options,
        scope="options",
    )
    assert result["scope"] == "options"
    # Both AAPL and TSLA closed positions detected — pipeline didn't crash.
    assert result["n_options_closed"] == 2

    # build_options_cash_flows must also accept tz-aware strings.
    cfs = build_options_cash_flows(options)
    # 3 cash-flow rows: 2 BTOs and 1 STC. OEXP excluded.
    assert len(cfs) == 3


def test_compute_mwr_clamps_explosive_xirr():
    """Bug 2 fix: pyxirr can diverge to astronomical values on
    dense/conflicting cash flows. compute_mwr must clamp |result| > 100
    and return None so the dashboard renders '—' instead of e+35."""
    from lib.performance_math import compute_mwr

    # Construct a degenerate cash-flow series: many same-day flips at
    # opposite signs. This is the kind of pattern that makes Newton-Raphson
    # diverge.
    bad_trades = []
    base_date = "2025-01-15"
    for i in range(50):
        bad_trades.append(_eq("buy",  "AAA", 100, 10.0, base_date, amount=-10000.0))
        bad_trades.append(_eq("sell", "AAA", 100, 10.0, base_date, amount=+10000.0))
    # The bounded check should kick in if XIRR diverges. Even if XIRR is
    # well-behaved on this fixture, the test asserts the contract: result
    # is either a sane float in [-100, 100] or None.
    result = compute_mwr(bad_trades, [], current_value=0.0)
    assert result is None or (-100.0 <= result <= 100.0), \
        f"MWR must be None or in [-100, 100], got {result}"


def test_oexp_zero_cash_flow_no_double_count():
    """An OEXP'd long position should realize the full premium loss exactly
    once — at OEXP via the inventory-drop mechanism, NOT via a phantom cash
    flow on the OEXP date."""
    options = [
        _opt("BTO", "TSLA", 100, "put", 1, -300.0, "2025-01-10",
             expiration="2025-03-15"),
        _opt("OEXP", "TSLA", 100, "put", 1, None, "2025-03-15",
             expiration="2025-03-15"),
    ]
    result = compute_headline_stats(
        trades=[],
        dividends=[],
        options_trades=options,
        scope="options",
    )
    # Realized P&L = exactly -$300 (full premium loss). If OEXP emitted a
    # phantom cash flow, the loss would compound to -$600.
    assert abs(result["total_realized_pnl"] - (-300.0)) < 0.01

    # build_options_cash_flows must not include the OEXP row.
    cfs = build_options_cash_flows(options)
    assert len(cfs) == 1
    assert cfs[0][1] == -300.0  # only the BTO outflow
