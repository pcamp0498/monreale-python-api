"""Unit tests for lib/options_spreads — multi-leg spread detection.

Run: python -m pytest tests/test_options_spreads.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.options_spreads import detect_spreads


def _leg(option_type, position_side, strike,
         contracts=1, total_cost=0.0, total_proceeds=0.0, realized_pnl=0.0,
         expiration="2026-06-20", open_date="2026-01-15", close_date="2026-02-01",
         ticker="NVDA", user_id="user1", leg_id=None, outcome="closed"):
    if leg_id is None:
        leg_id = f"{ticker}-{strike}-{option_type}-{position_side}-{open_date}"
    return {
        "id": leg_id,
        "user_id": user_id,
        "underlying_ticker": ticker,
        "expiration_date": expiration,
        "strike": float(strike),
        "option_type": option_type,
        "position_side": position_side,
        "contracts": float(contracts),
        "total_cost": float(total_cost),
        "total_proceeds": float(total_proceeds),
        "realized_pnl": float(realized_pnl),
        "outcome": outcome,
        "open_date": open_date,
        "close_date": close_date,
        "days_held": 17,
    }


# ─── Vertical spreads ───────────────────────────────────────────────────

def test_bull_call_spread():
    """Long lower-strike call + short higher-strike call → bull_call_spread."""
    legs = [
        _leg("call", "long",  100, total_cost=500.0, total_proceeds=300.0, realized_pnl=-200.0),
        _leg("call", "short", 110, total_cost=100.0, total_proceeds=200.0, realized_pnl=+100.0),
    ]
    spreads = detect_spreads(legs)
    assert len(spreads) == 1
    s = spreads[0]
    assert s["spread_type"] == "bull_call_spread"
    assert s["detection_confidence"] == "high"
    assert s["realized_pnl"] == -100.0
    # net_cost = long-side cost; net_proceeds = short-side proceeds
    assert s["net_cost"] == 500.0
    assert s["net_proceeds"] == 200.0
    # basis = |500 - 200| = 300; pct = -100/300 = -33.33%
    assert s["realized_pnl_pct"] is not None
    assert abs(s["realized_pnl_pct"] + 33.3333) < 0.01
    assert sorted(s["leg_position_ids"]) == sorted([l["id"] for l in legs])


def test_bear_call_spread():
    """Short lower-strike call + long higher-strike call → bear_call_spread."""
    legs = [
        _leg("call", "short", 100),
        _leg("call", "long",  110),
    ]
    spreads = detect_spreads(legs)
    assert spreads[0]["spread_type"] == "bear_call_spread"
    assert spreads[0]["detection_confidence"] == "high"


def test_bull_put_spread():
    """Long lower-strike put + short higher-strike put → bull_put_spread (credit)."""
    legs = [
        _leg("put", "long",  100),
        _leg("put", "short", 110),
    ]
    spreads = detect_spreads(legs)
    assert spreads[0]["spread_type"] == "bull_put_spread"


def test_bear_put_spread():
    """Long higher-strike put + short lower-strike put → bear_put_spread (debit)."""
    legs = [
        _leg("put", "long",  110),
        _leg("put", "short", 100),
    ]
    spreads = detect_spreads(legs)
    assert spreads[0]["spread_type"] == "bear_put_spread"


# ─── Straddles & strangles ──────────────────────────────────────────────

def test_long_straddle():
    legs = [
        _leg("call", "long", 100),
        _leg("put",  "long", 100),
    ]
    spreads = detect_spreads(legs)
    assert spreads[0]["spread_type"] == "long_straddle"
    assert spreads[0]["detection_confidence"] == "high"


def test_short_straddle():
    legs = [
        _leg("call", "short", 100),
        _leg("put",  "short", 100),
    ]
    spreads = detect_spreads(legs)
    assert spreads[0]["spread_type"] == "short_straddle"


def test_long_strangle():
    legs = [
        _leg("call", "long", 110),
        _leg("put",  "long",  90),
    ]
    spreads = detect_spreads(legs)
    assert spreads[0]["spread_type"] == "long_strangle"


def test_short_strangle():
    legs = [
        _leg("call", "short", 110),
        _leg("put",  "short",  90),
    ]
    spreads = detect_spreads(legs)
    assert spreads[0]["spread_type"] == "short_strangle"


# ─── Stacked positions (Patrick's NIO/LULU pattern) ─────────────────────

def test_stacked_long_calls():
    """Same direction, different strikes, both long, both calls."""
    legs = [
        _leg("call", "long", 30),
        _leg("call", "long", 35),
    ]
    spreads = detect_spreads(legs)
    assert spreads[0]["spread_type"] == "stacked_long_calls"
    assert spreads[0]["detection_confidence"] == "medium"


def test_stacked_long_puts():
    legs = [
        _leg("put", "long", 30),
        _leg("put", "long", 35),
    ]
    spreads = detect_spreads(legs)
    assert spreads[0]["spread_type"] == "stacked_long_puts"
    assert spreads[0]["detection_confidence"] == "medium"


def test_stacked_short_calls():
    legs = [
        _leg("call", "short", 30),
        _leg("call", "short", 35),
    ]
    spreads = detect_spreads(legs)
    assert spreads[0]["spread_type"] == "stacked_short_calls"


def test_stacked_short_puts():
    legs = [
        _leg("put", "short", 30),
        _leg("put", "short", 35),
    ]
    spreads = detect_spreads(legs)
    assert spreads[0]["spread_type"] == "stacked_short_puts"


# ─── 4-leg structures ────────────────────────────────────────────────────

def test_iron_condor():
    """4 distinct strikes, one long + one short on each side."""
    legs = [
        _leg("put",  "long",   90),
        _leg("put",  "short",  95),
        _leg("call", "short", 105),
        _leg("call", "long",  110),
    ]
    spreads = detect_spreads(legs)
    assert spreads[0]["spread_type"] == "iron_condor"
    assert spreads[0]["detection_confidence"] == "high"


def test_iron_butterfly():
    """3 distinct strikes — short body shares the same middle strike."""
    legs = [
        _leg("put",  "long",   90),
        _leg("put",  "short", 100),
        _leg("call", "short", 100),
        _leg("call", "long",  110),
    ]
    spreads = detect_spreads(legs)
    assert spreads[0]["spread_type"] == "iron_butterfly"


def test_4leg_all_long_is_custom():
    """4 longs with no shorts is not an iron — flag custom."""
    legs = [
        _leg("call", "long",  90),
        _leg("call", "long", 100),
        _leg("put",  "long", 100),
        _leg("put",  "long", 110),
    ]
    spreads = detect_spreads(legs)
    assert spreads[0]["spread_type"] == "custom"


# ─── Calendar / diagonal (different expirations) ────────────────────────

def test_calendar_spread():
    legs = [
        _leg("call", "long",  100, expiration="2026-06-20"),
        _leg("call", "short", 100, expiration="2026-09-19"),
    ]
    spreads = detect_spreads(legs)
    assert spreads[0]["spread_type"] == "calendar"
    # Primary expiration = earliest leg
    assert spreads[0]["expiration_date"] == "2026-06-20"


def test_diagonal_spread():
    legs = [
        _leg("call", "long",  100, expiration="2026-06-20"),
        _leg("call", "short", 110, expiration="2026-09-19"),
    ]
    spreads = detect_spreads(legs)
    assert spreads[0]["spread_type"] == "diagonal"


# ─── Custom & boundary cases ────────────────────────────────────────────

def test_custom_3legs():
    legs = [
        _leg("call", "long",  100),
        _leg("call", "short", 110),
        _leg("put",  "long",   90),
    ]
    spreads = detect_spreads(legs)
    assert spreads[0]["spread_type"] == "custom"
    assert spreads[0]["detection_confidence"] == "low"


def test_does_not_merge_different_open_dates():
    """Genuinely independent trades on different broker days stay split."""
    legs = [
        _leg("call", "long",  100, open_date="2026-01-15"),
        _leg("call", "short", 110, open_date="2026-02-15"),
    ]
    spreads = detect_spreads(legs)
    # Two singleton groups → no spreads emitted
    assert spreads == []


def test_does_not_merge_different_tickers():
    legs = [
        _leg("call", "long",  100, ticker="NVDA"),
        _leg("call", "short", 110, ticker="AMD"),
    ]
    spreads = detect_spreads(legs)
    assert spreads == []


def test_idempotent_re_detection():
    """Same input → same output. Required for UPSERT idempotency."""
    legs = [
        _leg("call", "long",  100, leg_id="a"),
        _leg("call", "short", 110, leg_id="b"),
    ]
    s1 = detect_spreads(legs)
    s2 = detect_spreads(legs)
    assert s1 == s2
    assert sorted(s1[0]["leg_position_ids"]) == ["a", "b"]


def test_skips_conversion_unhandled():
    """CONV placeholders have no real math — must not corrupt aggregation."""
    legs = [
        _leg("call", "long",  100, total_cost=500.0, realized_pnl=-100.0),
        _leg("call", "short", 110, total_proceeds=200.0, realized_pnl=+50.0),
        _leg("call", "long",  120, outcome="conversion_unhandled", leg_id="conv-leg"),
    ]
    spreads = detect_spreads(legs)
    # The 2 real legs cluster; CONV leg drops out → 2-leg vertical, not 3-leg custom
    assert len(spreads) == 1
    assert spreads[0]["spread_type"] == "bull_call_spread"
    assert "conv-leg" not in spreads[0]["leg_position_ids"]


def test_empty_input():
    assert detect_spreads([]) == []


def test_single_leg_emits_no_spread():
    spreads = detect_spreads([_leg("call", "long", 100)])
    assert spreads == []
