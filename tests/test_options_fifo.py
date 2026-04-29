"""Unit tests for lib/options_fifo.

Run: python -m pytest tests/test_options_fifo.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.options_fifo import match_options_positions


def _trade(trans_code, contracts, total_amount, executed_at,
           ticker="NVDA", strike=900.0, opt_type="call",
           expiration="2026-06-20", trade_id=None):
    return {
        "id": trade_id,
        "trans_code": trans_code,
        "contracts": contracts,
        "total_amount": total_amount,
        "executed_at": executed_at,
        "underlying_ticker": ticker,
        "strike": strike,
        "option_type": opt_type,
        "expiration_date": expiration,
        "premium_per_contract": (total_amount / contracts) if total_amount and contracts else None,
        "cancellation_status": "normal",
    }


# ─── Long positions: BTO + STC ──────────────────────────────────────────

def test_simple_long_position_closed_profit():
    """1 BTO @ $500 cost, then 1 STC @ $800 proceeds → +$300 realized."""
    trades = [
        _trade("BTO", 1, 500.0, "2026-01-15", trade_id="t1"),
        _trade("STC", 1, 800.0, "2026-02-01", trade_id="t2"),
    ]
    result = match_options_positions(trades)
    assert len(result["closed_positions"]) == 1
    assert result["open_positions"] == []
    assert result["match_warnings"] == []

    p = result["closed_positions"][0]
    assert p["position_side"] == "long"
    assert p["outcome"] == "closed"
    assert p["contracts"] == 1.0
    assert p["total_cost"] == 500.0
    assert p["total_proceeds"] == 800.0
    assert p["realized_pnl"] == 300.0
    assert p["days_held"] == 17
    assert p["open_trade_ids"] == ["t1"]
    assert p["close_trade_ids"] == ["t2"]
    # Long return: +$300 / $500 cost = +60.0%
    assert p["realized_pnl_pct"] == 60.0


def test_simple_long_position_closed_loss():
    """1 BTO @ $1000 cost, 1 STC @ $300 proceeds → -$700 realized."""
    trades = [
        _trade("BTO", 1, 1000.0, "2026-01-15"),
        _trade("STC", 1, 300.0, "2026-02-15"),
    ]
    result = match_options_positions(trades)
    p = result["closed_positions"][0]
    assert p["realized_pnl"] == -700.0
    assert p["total_cost"] == 1000.0
    assert p["total_proceeds"] == 300.0


def test_partial_close():
    """3 BTO contracts, then STC 1 contract — 2 still open."""
    trades = [
        _trade("BTO", 3, 1500.0, "2026-01-15"),  # $500/contract avg
        _trade("STC", 1, 700.0, "2026-02-01"),
    ]
    result = match_options_positions(trades)
    assert len(result["closed_positions"]) == 1
    assert len(result["open_positions"]) == 1

    closed = result["closed_positions"][0]
    assert closed["contracts"] == 1.0
    assert closed["total_cost"] == 500.0  # pro-rata: $1500 × (1/3)
    assert closed["total_proceeds"] == 700.0
    assert closed["realized_pnl"] == 200.0

    opened = result["open_positions"][0]
    assert opened["contracts"] == 2.0
    assert opened["total_cost"] == 1000.0  # remaining $1500 × (2/3)


def test_multiple_BTO_then_one_STC_FIFO():
    """Two separate BTO lots, one STC — FIFO matches the OLDEST lot first."""
    trades = [
        _trade("BTO", 1, 400.0, "2026-01-10", trade_id="lot1"),  # older
        _trade("BTO", 1, 600.0, "2026-01-20", trade_id="lot2"),  # newer
        _trade("STC", 1, 700.0, "2026-02-01", trade_id="close"),
    ]
    result = match_options_positions(trades)
    assert len(result["closed_positions"]) == 1
    assert len(result["open_positions"]) == 1

    closed = result["closed_positions"][0]
    assert closed["open_trade_ids"] == ["lot1"]  # FIFO: oldest closed
    assert closed["total_cost"] == 400.0
    assert closed["realized_pnl"] == 300.0  # 700 - 400

    # The newer $600 lot is still open
    assert result["open_positions"][0]["open_trade_ids"] == ["lot2"]
    assert result["open_positions"][0]["total_cost"] == 600.0


def test_expired_worthless_long():
    """BTO + OEXP (no STC) → expired_worthless, full premium loss."""
    trades = [
        _trade("BTO", 2, 800.0, "2026-01-15"),
        _trade("OEXP", 2, None, "2026-06-20"),
    ]
    result = match_options_positions(trades)
    assert len(result["closed_positions"]) == 1
    assert result["open_positions"] == []

    p = result["closed_positions"][0]
    assert p["outcome"] == "expired_worthless"
    assert p["position_side"] == "long"
    assert p["realized_pnl"] == -800.0
    assert p["total_cost"] == 800.0
    assert p["total_proceeds"] == 0.0


# ─── Short positions: STO + BTC / OEXP ──────────────────────────────────

def test_short_position_closed():
    """STO opened @ +$500 proceeds, BTC closed @ -$200 cost → +$300 realized."""
    trades = [
        _trade("STO", 1, 500.0, "2026-01-15"),
        _trade("BTC", 1, 200.0, "2026-02-01"),
    ]
    result = match_options_positions(trades)
    assert len(result["closed_positions"]) == 1

    p = result["closed_positions"][0]
    assert p["position_side"] == "short"
    assert p["outcome"] == "closed"
    assert p["total_proceeds"] == 500.0
    assert p["total_cost"] == 200.0
    assert p["realized_pnl"] == 300.0


def test_expired_short_position():
    """STO + OEXP (premium retained) → expired_worthless, +premium realized."""
    trades = [
        _trade("STO", 1, 400.0, "2026-01-15"),
        _trade("OEXP", 1, None, "2026-06-20"),
    ]
    result = match_options_positions(trades)
    p = result["closed_positions"][0]
    assert p["outcome"] == "expired_worthless"
    assert p["position_side"] == "short"
    assert p["realized_pnl"] == 400.0
    assert p["total_proceeds"] == 400.0
    assert p["total_cost"] == 0.0


# ─── Position-key isolation ─────────────────────────────────────────────

def test_position_key_isolation():
    """NVDA $850 call ≠ NVDA $900 call — they don't FIFO-match each other."""
    trades = [
        _trade("BTO", 1, 500.0, "2026-01-15", strike=850.0),
        _trade("BTO", 1, 700.0, "2026-01-15", strike=900.0),
        _trade("STC", 1, 600.0, "2026-02-01", strike=900.0),
    ]
    result = match_options_positions(trades)
    # The STC $900 closes against the BTO $900, NOT the BTO $850
    assert len(result["closed_positions"]) == 1
    assert result["closed_positions"][0]["strike"] == 900.0
    assert result["closed_positions"][0]["realized_pnl"] == -100.0  # 600-700

    # The BTO $850 stays open
    assert len(result["open_positions"]) == 1
    assert result["open_positions"][0]["strike"] == 850.0


def test_position_key_isolation_call_vs_put():
    """NVDA $900 call ≠ NVDA $900 put — different option_type, different key."""
    trades = [
        _trade("BTO", 1, 500.0, "2026-01-15", opt_type="call"),
        _trade("STC", 1, 600.0, "2026-02-01", opt_type="put"),  # mismatch
    ]
    result = match_options_positions(trades)
    # STC put can't close BTO call — warning emitted, no closed_position
    assert result["closed_positions"] == []
    assert len(result["open_positions"]) == 1
    assert any("STC with no open long" in w for w in result["match_warnings"])


def test_match_warnings_emitted_on_oversold():
    """STC requesting more contracts than the long queue holds → warning."""
    trades = [
        _trade("BTO", 1, 500.0, "2026-01-15"),
        _trade("STC", 3, 1800.0, "2026-02-01"),  # requesting 3, only 1 open
    ]
    result = match_options_positions(trades)
    assert len(result["closed_positions"]) == 1  # the one match that did happen
    assert result["closed_positions"][0]["contracts"] == 1.0
    assert any("oversold long" in w for w in result["match_warnings"])


def test_oexp_with_no_open_position_warns():
    """Defensive: OEXP fired against an empty position ledger."""
    trades = [
        _trade("OEXP", 1, None, "2026-06-20"),
    ]
    result = match_options_positions(trades)
    assert result["closed_positions"] == []
    assert any("no matching open position" in w for w in result["match_warnings"])


def test_btc_against_no_short_position_warns():
    """Symmetric: BTC against empty short queue → warning, no leak into long queue."""
    trades = [
        _trade("BTO", 1, 500.0, "2026-01-15"),  # long queue has 1
        _trade("BTC", 1, 200.0, "2026-02-01"),  # but BTC needs short queue
    ]
    result = match_options_positions(trades)
    assert result["closed_positions"] == []  # no cross-contamination
    assert len(result["open_positions"]) == 1  # long stays open
    assert any("BTC with no open short" in w for w in result["match_warnings"])


def test_cancelled_options_excluded():
    """Trades flagged as cancelled don't enter the FIFO walk at all."""
    trades = [
        _trade("BTO", 1, 500.0, "2026-01-15"),
        {**_trade("STC", 1, 800.0, "2026-02-01"), "cancellation_status": "cancelled_by_broker"},
    ]
    result = match_options_positions(trades)
    # The STC is cancelled — long stays open
    assert result["closed_positions"] == []
    assert len(result["open_positions"]) == 1


# ─── Same-day open/close tiebreaker ─────────────────────────────────────

def test_fifo_same_date_open_before_close():
    """When BTO and STC share an executed_at, BTO must process first.

    Why: Robinhood's activity CSV is descending-time-order, so the STC row
    often appears BEFORE its same-day BTO partner in the input. A naive
    chronological sort would preserve that order on equal timestamps and
    emit a phantom 'STC with no open long' warning. The tiebreaker on the
    sort key (0 for opens, 1 for closes) forces opens-first.
    """
    trades = [
        # STC listed first in the input (mirrors Robinhood CSV order)
        _trade("STC", 1, 800.0, "2026-01-15", trade_id="close"),
        _trade("BTO", 1, 500.0, "2026-01-15", trade_id="open"),
    ]
    result = match_options_positions(trades)
    assert len(result["closed_positions"]) == 1
    assert result["open_positions"] == []
    assert result["match_warnings"] == []  # no phantom warning

    p = result["closed_positions"][0]
    assert p["open_trade_ids"] == ["open"]
    assert p["close_trade_ids"] == ["close"]
    assert p["realized_pnl"] == 300.0
    assert p["days_held"] == 0  # same-day round trip


# ─── CONV (deferred — Sprint 9C.6) ──────────────────────────────────────

def test_conv_emits_conversion_unhandled_and_manual_review():
    """CONV trans_code surfaces in manual_review_required and emits a
    deterministic conversion_unhandled record with realized_pnl=0 so it
    is excluded from P&L totals downstream."""
    trades = [
        {
            "id": "conv1",
            "trans_code": "CONV",
            "contracts": 1,
            "total_amount": None,
            "executed_at": "2026-03-01",
            "underlying_ticker": "FOO",
            "strike": 50.0,
            "option_type": "call",
            "expiration_date": "2026-06-20",
            "premium_per_contract": None,
            "cancellation_status": "normal",
        },
    ]
    result = match_options_positions(trades)
    assert len(result["closed_positions"]) == 1
    p = result["closed_positions"][0]
    assert p["outcome"] == "conversion_unhandled"
    assert p["realized_pnl"] == 0.0
    assert p["realized_pnl_pct"] is None
    assert p["total_cost"] is None
    assert p["total_proceeds"] is None

    review = result["manual_review_required"]
    assert len(review) == 1
    assert review[0]["reason"] == "conversion_unhandled"
    assert review[0]["trans_code"] == "CONV"
    assert review[0]["deferred_to_sprint"] == "9C.6"


# ─── realized_pnl_pct field ─────────────────────────────────────────────

def test_realized_pnl_pct_long_close():
    """Long close: pnl / cost. $300 / $500 = 60.0%."""
    trades = [
        _trade("BTO", 1, 500.0, "2026-01-15"),
        _trade("STC", 1, 800.0, "2026-02-01"),
    ]
    result = match_options_positions(trades)
    assert result["closed_positions"][0]["realized_pnl_pct"] == 60.0


def test_realized_pnl_pct_short_close():
    """Short close: pnl / proceeds (premium received as basis).
    $300 kept / $500 premium = 60.0%."""
    trades = [
        _trade("STO", 1, 500.0, "2026-01-15"),
        _trade("BTC", 1, 200.0, "2026-02-01"),
    ]
    result = match_options_positions(trades)
    assert result["closed_positions"][0]["realized_pnl_pct"] == 60.0


def test_realized_pnl_pct_expired_long_is_negative_100():
    trades = [
        _trade("BTO", 1, 500.0, "2026-01-15"),
        _trade("OEXP", 1, None, "2026-06-20"),
    ]
    result = match_options_positions(trades)
    assert result["closed_positions"][0]["realized_pnl_pct"] == -100.0


def test_realized_pnl_pct_expired_short_is_positive_100():
    trades = [
        _trade("STO", 1, 400.0, "2026-01-15"),
        _trade("OEXP", 1, None, "2026-06-20"),
    ]
    result = match_options_positions(trades)
    assert result["closed_positions"][0]["realized_pnl_pct"] == 100.0
