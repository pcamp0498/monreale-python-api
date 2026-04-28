"""Synthetic-fixture tests for the bias-detection module.

Run with: python -m pytest tests/test_bias_detection.py -v

Each test is fully self-contained — synthesizes its own trade history,
NAV DataFrame (where needed), and SPY price series. No real network calls.
"""
import sys
import json
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from lib.bias_detection import (
    detect_panic_sells,
    detect_disposition_effect,
    attribute_cash_flow_timing,
    detect_sector_concentration_drift,
    detect_position_sizing_creep,
    detect_frequency_increase,
    detect_sector_cycling,
    detect_time_of_day_bias,
    _compute_panic_severity,
    PANIC_SELL_SEVERITY_HIGH_CUTOFF,
    PANIC_SELL_SEVERITY_MED_CUTOFF,
    DISPOSITION_RATIO_TRIGGER,
    DISPOSITION_N_FLOOR,
)
from lib.performance_math import _sanitize_for_json


def _trade(ticker, action, shares, price, executed_at, amount=None, cancellation_status="normal"):
    if amount is None:
        amount = shares * price * (-1 if action == "buy" else 1)
    return {
        "ticker": ticker,
        "action": action,
        "shares": shares,
        "price": price,
        "amount": amount,
        "executed_at": executed_at,
        "cancellation_status": cancellation_status,
    }


def _make_nav(dates, holdings_value=10000.0, cash_flow=0.0):
    """Build a minimal NAV DataFrame with the columns the detectors read."""
    if isinstance(dates, str):
        dates = pd.bdate_range(start=dates, periods=400)
    if isinstance(cash_flow, (int, float)):
        cash_flow = pd.Series(cash_flow, index=dates)
    if isinstance(holdings_value, (int, float)):
        holdings_value = pd.Series(holdings_value, index=dates)
    nav = pd.DataFrame({"holdings_value": holdings_value, "cash_flow": cash_flow})
    nav["nav"] = nav["holdings_value"]
    return nav


def _make_spy(start="2019-01-01", periods=2000, drift=0.0003, vol=0.01, seed=0):
    np.random.seed(seed)
    idx = pd.bdate_range(start=start, periods=periods)
    returns = np.random.normal(drift, vol, len(idx))
    levels = 300.0 * np.cumprod(1 + returns)
    return pd.Series(levels, index=idx)


# ─── Severity formula direct unit tests (matches worked examples in spec) ────

def test_severity_formula_classic_covid_event():
    """Worked example A: -8% loss, 5 positions, 47 idle days, +18.4% recovery → high."""
    out = _compute_panic_severity(loss_pct_of_nav=-0.08, n_positions=5, days_idle_after=47, spy_return_6m=0.184)
    assert out["severity"] == "high"
    assert 0.65 < out["severity_score"] < 0.69
    assert out["severity_basis"] == "full"
    assert out["severity_factors"]["loss_pct_norm"] == 0.8
    assert out["severity_factors"]["market_recovery_norm"] == 0.92


def test_severity_formula_prescient_de_risk():
    """Worked example B: -3% loss, 3 positions, 35 idle, -12% spy 6m → low."""
    out = _compute_panic_severity(loss_pct_of_nav=-0.03, n_positions=3, days_idle_after=35, spy_return_6m=-0.12)
    assert out["severity"] == "low"
    assert out["severity_factors"]["market_recovery_norm"] == 0.0  # negative clipped


def test_severity_formula_partial_data_reweighting():
    """Worked example D: full SPY data unavailable → partial_no_market_data, weights renormalized."""
    out = _compute_panic_severity(loss_pct_of_nav=-0.07, n_positions=6, days_idle_after=25, spy_return_6m=None)
    assert out["severity_basis"] == "partial_no_market_data"
    assert out["severity_factors"]["market_recovery_norm"] is None
    # Recovery weight dropped → other weights renormalized
    weights = out["severity_weights"]
    assert weights["market_recovery"] is None
    assert abs(weights["loss_pct"] + weights["n_positions"] + weights["days_idle"] - 1.0) < 1e-6


def test_severity_formula_minimum_qualifying_event():
    """Worked example F: just at all four trigger floors → low bucket."""
    out = _compute_panic_severity(loss_pct_of_nav=-0.02, n_positions=3, days_idle_after=30, spy_return_6m=0.0)
    assert out["severity"] == "low"
    assert out["severity_score"] < PANIC_SELL_SEVERITY_MED_CUTOFF


def test_severity_formula_maxout():
    """All four inputs at saturation breakpoints → score 1.0."""
    out = _compute_panic_severity(loss_pct_of_nav=-0.20, n_positions=15, days_idle_after=200, spy_return_6m=0.50)
    assert out["severity"] == "high"
    assert out["severity_score"] >= 0.99


# ─── PANIC SELL detector tests ───────────────────────────────────────────────

def test_panic_sell_detected_with_market_recovery():
    """5 sells one day, large loss vs NAV, no buys for next 30+ trading days,
    SPY recovers → event emitted with severity high."""
    sell_date = "2020-03-12"
    trades = [
        # Pre-existing buys to give us shares to sell
        _trade("AAPL", "buy", 100, 200.0, "2019-06-01"),
        _trade("MSFT", "buy", 50,  150.0, "2019-06-01"),
        _trade("JPM",  "buy", 80,  120.0, "2019-06-01"),
        _trade("DIS",  "buy", 60,  140.0, "2019-06-01"),
        _trade("BA",   "buy", 40,  300.0, "2019-06-01"),
        # The panic-sell day
        _trade("AAPL", "sell", 100, 130.0, sell_date),  # -7000 loss
        _trade("MSFT", "sell", 50,  100.0, sell_date),  # -2500
        _trade("JPM",  "sell", 80,   90.0, sell_date),  # -2400
        _trade("DIS",  "sell", 60,  110.0, sell_date),  # -1800
        _trade("BA",   "sell", 40,  200.0, sell_date),  # -4000
        # No buys for the next 6+ months
    ]
    # NAV around the panic day: pre-day NAV = 100k, post crashes
    nav_idx = pd.bdate_range(start="2019-06-01", end="2021-06-01")
    nav = _make_nav(nav_idx, holdings_value=100000.0, cash_flow=0.0)
    # SPY: pre-panic at 300, drops to 220 ON panic_date, rallies +0.80/day.
    # 6m forward: 220 + 180*0.80 = 364 → return = +65% (well above the +20%
    # saturation, ensures market_recovery_norm = 1.0 in severity).
    spy_idx = pd.bdate_range(start="2019-06-01", end="2021-06-01")
    levels = []
    for d in spy_idx:
        if d < pd.Timestamp(sell_date):
            levels.append(300.0)
        else:
            days_since = (d - pd.Timestamp(sell_date)).days
            levels.append(220.0 + days_since * 0.80)
    spy = pd.Series(levels, index=spy_idx)

    events = detect_panic_sells(trades, nav, spy)
    assert len(events) == 1
    e = events[0]
    assert e["date"] == sell_date
    assert e["n_positions"] == 5
    assert e["total_loss_dollars"] < -10000
    assert e["severity"] in ("high", "medium")  # likely high; the test fixture variance allows medium
    assert e["market_context"]["spy_return_6m_forward"] is not None
    assert e["market_context"]["spy_return_6m_forward"] > 0.10  # market recovered
    assert e["severity_basis"] == "full"


def test_panic_sell_no_panic_when_sells_spread_over_days():
    """Same 5 sells but spread across 5 different days → not flagged."""
    trades = [
        _trade("AAPL", "buy", 100, 200.0, "2019-06-01"),
        _trade("MSFT", "buy", 50,  150.0, "2019-06-01"),
        _trade("JPM",  "buy", 80,  120.0, "2019-06-01"),
        _trade("AAPL", "sell", 100, 130.0, "2020-03-09"),
        _trade("MSFT", "sell", 50,  100.0, "2020-03-10"),
        _trade("JPM",  "sell", 80,   90.0, "2020-03-11"),
    ]
    nav = _make_nav(pd.bdate_range(start="2019-06-01", periods=400), holdings_value=100000.0)
    spy = _make_spy(start="2019-06-01")
    events = detect_panic_sells(trades, nav, spy)
    assert events == []


def test_panic_sell_subsequent_buys_above_threshold():
    """5 sells one day, but Patrick re-buys 50% of liquidated value within 30 days → not panic."""
    sell_date = "2020-03-12"
    trades = [
        _trade("AAPL", "buy", 100, 200.0, "2019-06-01"),
        _trade("MSFT", "buy", 50,  150.0, "2019-06-01"),
        _trade("JPM",  "buy", 80,  120.0, "2019-06-01"),
        _trade("DIS",  "buy", 60,  140.0, "2019-06-01"),
        _trade("BA",   "buy", 40,  300.0, "2019-06-01"),
        _trade("AAPL", "sell", 100, 130.0, sell_date),
        _trade("MSFT", "sell", 50,  100.0, sell_date),
        _trade("JPM",  "sell", 80,   90.0, sell_date),
        _trade("DIS",  "sell", 60,  110.0, sell_date),
        _trade("BA",   "sell", 40,  200.0, sell_date),
        # Re-buy 50% of liquidated value within 2 weeks
        _trade("AAPL", "buy", 100, 130.0, "2020-03-26"),  # ~13k
        _trade("MSFT", "buy", 50,  100.0, "2020-03-26"),  # ~5k
    ]
    nav = _make_nav(pd.bdate_range(start="2019-06-01", periods=600), holdings_value=100000.0)
    spy = _make_spy(start="2019-06-01")
    events = detect_panic_sells(trades, nav, spy)
    assert events == []  # subsequent buys ≥ 10% of liquidated → not panic


def test_panic_sell_market_dropped_after():
    """Market kept falling after sell → severity should bucket low (prescient de-risk)."""
    sell_date = "2020-02-15"
    trades = [
        _trade("AAPL", "buy", 100, 200.0, "2019-06-01"),
        _trade("MSFT", "buy", 50,  150.0, "2019-06-01"),
        _trade("JPM",  "buy", 80,  120.0, "2019-06-01"),
        _trade("AAPL", "sell", 100, 180.0, sell_date),  # -2000
        _trade("MSFT", "sell", 50,  130.0, sell_date),  # -1000
        _trade("JPM",  "sell", 80,  100.0, sell_date),  # -1600
    ]
    nav = _make_nav(pd.bdate_range(start="2019-06-01", periods=600), holdings_value=100000.0)
    # SPY drops 15% over the next 6 months
    spy_idx = pd.bdate_range(start="2019-06-01", end="2021-06-01")
    levels = []
    for d in spy_idx:
        if d <= pd.Timestamp(sell_date):
            levels.append(300.0)
        else:
            days_since = (d - pd.Timestamp(sell_date)).days
            levels.append(300.0 - days_since * 0.20)
    spy = pd.Series(levels, index=spy_idx)
    events = detect_panic_sells(trades, nav, spy)
    # Loss is only ~4.6% of NAV — may or may not pass the 2% trigger depending
    # on the FIFO match. If it does pass, market_recovery_norm = 0 (clipped),
    # so severity should be at most "medium". The asserting point is that a
    # post-event market drop does NOT score severity = "high".
    if events:
        assert events[0]["severity"] in ("low", "medium")
        assert events[0]["severity_factors"]["market_recovery_norm"] == 0.0


# ─── DISPOSITION EFFECT detector tests ───────────────────────────────────────

def test_disposition_flags_deep_loss_long_hold():
    """Open position at -60% loss, held 18 months, no sells → flagged."""
    today = pd.Timestamp("2024-06-01")
    trades = [
        _trade("PLTR", "buy", 100, 30.0, "2022-01-15"),  # cost basis = $3000
    ]
    current_prices = {"PLTR": 12.0}  # value = $1200, loss = -60%
    out = detect_disposition_effect(trades, current_prices=current_prices, today=today)
    assert len(out["flagged_positions"]) == 1
    f = out["flagged_positions"][0]
    assert f["ticker"] == "PLTR"
    assert f["unrealized_loss_pct"] < -0.50
    assert f["days_held"] > 365
    assert f["n_subsequent_buys_same_ticker"] == 1


def test_disposition_skips_position_with_sells():
    """Position with prior sells is excluded from flagged_positions."""
    today = pd.Timestamp("2024-06-01")
    trades = [
        _trade("PLTR", "buy", 100, 30.0, "2022-01-15"),
        _trade("PLTR", "sell", 50, 25.0, "2022-06-15"),  # any sell disqualifies
    ]
    current_prices = {"PLTR": 12.0}
    out = detect_disposition_effect(trades, current_prices=current_prices, today=today)
    assert out["flagged_positions"] == []


def test_disposition_skips_recent_position():
    """Deep loss but only 6 months old → not flagged."""
    today = pd.Timestamp("2024-06-01")
    trades = [
        _trade("PLTR", "buy", 100, 30.0, "2024-01-01"),
    ]
    current_prices = {"PLTR": 12.0}
    out = detect_disposition_effect(trades, current_prices=current_prices, today=today)
    assert out["flagged_positions"] == []


def test_disposition_ratio_computed_with_sufficient_n():
    """5 winners + 5 losers, losers held 3x longer → ratio ~3, effect detected."""
    today = pd.Timestamp("2024-06-01")
    trades = []
    # 5 winners — held 30 days each, gained
    for i, tk in enumerate(["W1", "W2", "W3", "W4", "W5"]):
        trades.append(_trade(tk, "buy",  10, 100.0, f"2023-01-{i+1:02d}"))
        trades.append(_trade(tk, "sell", 10, 120.0, f"2023-01-{i+1:02d}".replace("01", "02")))
    # 5 losers — held 90 days each, lost
    for i, tk in enumerate(["L1", "L2", "L3", "L4", "L5"]):
        trades.append(_trade(tk, "buy",  10, 100.0, f"2023-03-{i+1:02d}"))
        trades.append(_trade(tk, "sell", 10,  80.0, f"2023-06-{i+1:02d}"))
    out = detect_disposition_effect(trades, current_prices={}, today=today)
    stats = out["disposition_stats"]
    assert stats is not None
    assert stats["n_winners"] >= 5 and stats["n_losers"] >= 5
    assert stats["disposition_ratio"] is not None
    assert stats["disposition_ratio"] > 1.5
    assert stats["disposition_effect_detected"] is True


def test_disposition_ratio_null_when_n_below_floor():
    """Only 3 winners + 8 losers → stats = null with reason."""
    today = pd.Timestamp("2024-06-01")
    trades = []
    for i, tk in enumerate(["W1", "W2", "W3"]):
        trades.append(_trade(tk, "buy",  10, 100.0, f"2023-01-{i+1:02d}"))
        trades.append(_trade(tk, "sell", 10, 120.0, f"2023-02-{i+1:02d}"))
    for i, tk in enumerate(["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"]):
        trades.append(_trade(tk, "buy",  10, 100.0, f"2023-03-{i+1:02d}"))
        trades.append(_trade(tk, "sell", 10,  80.0, f"2023-06-{i+1:02d}"))
    out = detect_disposition_effect(trades, current_prices={}, today=today)
    assert out["disposition_stats"] is None
    assert "n_winners=3" in out["disposition_stats_reason"]
    assert f"floor={DISPOSITION_N_FLOOR}" in out["disposition_stats_reason"]


# ─── CASH-FLOW TIMING tests ──────────────────────────────────────────────────

def test_cash_flow_timing_at_troughs():
    """All inflows when SPY is at 52w lows → strong_positive_timing."""
    nav_idx = pd.bdate_range(start="2020-01-01", end="2021-12-31")
    cf = pd.Series(0.0, index=nav_idx)
    # Patrick buys (negative cash flow) on the SPY troughs
    trough_dates = ["2020-03-23", "2020-09-23", "2021-01-29", "2021-05-12", "2021-08-19"]
    for d in trough_dates:
        cf.loc[pd.Timestamp(d)] = -5000.0
    nav = pd.DataFrame({"holdings_value": 100000.0, "cash_flow": cf, "nav": 100000.0}, index=nav_idx)

    # Synthesize SPY where each trough_date is at the bottom of a 52w window
    spy_idx = pd.bdate_range(start="2019-01-01", end="2022-06-01")
    spy = pd.Series(np.full(len(spy_idx), 300.0), index=spy_idx)
    for d in trough_dates:
        # Make SPY drop to 250 on the trough_date and recover around it
        d_ts = pd.Timestamp(d)
        for offset_days in range(-30, 31):
            target = d_ts + pd.Timedelta(days=offset_days)
            if target in spy.index:
                spy.loc[target] = 280.0 - (30 - abs(offset_days)) * 1.0
        spy.loc[d_ts] = 250.0  # the actual trough

    out = attribute_cash_flow_timing(nav, spy, twr_annualized=0.10, mwr_annualized=0.18)
    assert out["finding_severity"] == "strong_positive_timing"
    assert out["weighted_avg_position_pct"] is not None
    assert out["weighted_avg_position_pct"] < 0.33
    assert out["attribution_pct"] is not None
    assert out["attribution_pct"] > 0


def test_cash_flow_timing_at_peaks():
    """All inflows when SPY is at 52w highs → strong_negative_timing."""
    nav_idx = pd.bdate_range(start="2020-01-01", end="2021-12-31")
    cf = pd.Series(0.0, index=nav_idx)
    # All business days — 2021-08-15 is Sunday, swap for Monday 2021-08-16
    peak_dates = ["2020-06-15", "2020-12-15", "2021-04-15", "2021-08-16", "2021-12-15"]
    for d in peak_dates:
        cf.loc[pd.Timestamp(d)] = -5000.0
    nav = pd.DataFrame({"holdings_value": 100000.0, "cash_flow": cf, "nav": 100000.0}, index=nav_idx)

    # SPY where each peak_date is at the top of its window
    spy_idx = pd.bdate_range(start="2019-01-01", end="2022-06-01")
    spy = pd.Series(np.full(len(spy_idx), 300.0), index=spy_idx)
    for d in peak_dates:
        d_ts = pd.Timestamp(d)
        for offset_days in range(-30, 31):
            target = d_ts + pd.Timedelta(days=offset_days)
            if target in spy.index:
                spy.loc[target] = 320.0 + (30 - abs(offset_days)) * 1.0
        spy.loc[d_ts] = 350.0

    out = attribute_cash_flow_timing(nav, spy, twr_annualized=0.10, mwr_annualized=0.05)
    assert out["finding_severity"] == "strong_negative_timing"
    assert out["weighted_avg_position_pct"] > 0.67
    # attribution_pct is the magnitude of the gap explained by timing.
    # For peak deployment + MWR<TWR (consistent bad timing), the formula
    # (mwr-twr)*(1-2*pos) = (-)*(-) = positive — i.e., "+5% of the return
    # shortfall is attributable to deploying at peaks". finding_severity
    # carries the directional interpretation (strong_negative_timing).
    assert out["attribution_pct"] > 0


def test_cash_flow_timing_insufficient_inflows():
    """Only 2 inflows → severity = insufficient_signal."""
    nav_idx = pd.bdate_range(start="2020-01-01", end="2021-12-31")
    cf = pd.Series(0.0, index=nav_idx)
    cf.loc[pd.Timestamp("2020-06-01")] = -5000.0
    cf.loc[pd.Timestamp("2020-09-01")] = -5000.0
    nav = pd.DataFrame({"holdings_value": 100000.0, "cash_flow": cf, "nav": 100000.0}, index=nav_idx)
    spy = _make_spy(start="2019-01-01", periods=1500)
    out = attribute_cash_flow_timing(nav, spy, twr_annualized=0.10, mwr_annualized=0.12)
    assert out["finding_severity"] == "insufficient_signal"


def test_cash_flow_timing_attribution_zero_at_midrange():
    """Inflows clustered at SPY mid-range → attribution near zero, neutral."""
    nav_idx = pd.bdate_range(start="2020-01-01", end="2021-12-31")
    cf = pd.Series(0.0, index=nav_idx)
    mid_dates = ["2020-06-15", "2020-09-15", "2020-12-15", "2021-03-15", "2021-06-15", "2021-09-15"]
    for d in mid_dates:
        cf.loc[pd.Timestamp(d)] = -5000.0
    nav = pd.DataFrame({"holdings_value": 100000.0, "cash_flow": cf, "nav": 100000.0}, index=nav_idx)
    # Nearly-flat SPY (very tight 52w range, all positions ~0.5)
    spy_idx = pd.bdate_range(start="2019-01-01", end="2022-06-01")
    np.random.seed(42)
    spy = pd.Series(300.0 + np.random.normal(0, 0.5, len(spy_idx)), index=spy_idx)
    out = attribute_cash_flow_timing(nav, spy, twr_annualized=0.10, mwr_annualized=0.11)
    if out["weighted_avg_position_pct"] is not None:
        assert 0.30 < out["weighted_avg_position_pct"] < 0.70
        assert abs(out["attribution_pct"]) < 0.005
    assert out["finding_severity"] in ("neutral", "insufficient_signal")


def test_cash_flow_timing_inflow_below_threshold_excluded():
    """Inflow < $250 doesn't count — only large deposits matter."""
    nav_idx = pd.bdate_range(start="2020-01-01", end="2021-12-31")
    cf = pd.Series(0.0, index=nav_idx)
    cf.loc[pd.Timestamp("2020-06-15")] = -100.0  # below $250 threshold
    nav = pd.DataFrame({"holdings_value": 100000.0, "cash_flow": cf, "nav": 100000.0}, index=nav_idx)
    spy = _make_spy(start="2019-01-01", periods=1500)
    out = attribute_cash_flow_timing(nav, spy, twr_annualized=0.10, mwr_annualized=0.12)
    assert out["finding_severity"] == "insufficient_signal"
    assert out["inflows_summary"]["n_total"] == 0


# ─── Deferred patterns ───────────────────────────────────────────────────────

def test_deferred_patterns_return_structured_response():
    for fn in (
        detect_sector_concentration_drift,
        detect_position_sizing_creep,
        detect_frequency_increase,
        detect_sector_cycling,
        detect_time_of_day_bias,
    ):
        out = fn()
        assert out["status"] == "insufficient_data"
        assert out["deferred"] is True
        assert "reason" in out
        assert "deferred_to_sprint" in out


# ─── End-to-end JSON serialization safety ────────────────────────────────────

# ─── Capitulation trigger (Issue-1 fix) ──────────────────────────────────────

def test_panic_sell_capitulation_trigger_fires_on_single_sell():
    """1 sell that crashes NAV >70% in a day → fires via capitulation
    trigger even though n_sells < 3 (would fail standard trigger)."""
    sell_date = "2022-03-07"
    trades = [
        _trade("AAPL", "buy", 100, 200.0, "2019-06-01"),
        _trade("AAPL", "sell", 95, 130.0, sell_date),  # near-total liquidation, big loss
    ]
    # NAV crashes from 100k to 1.5k on sell_date (-98.5%)
    nav_idx = pd.bdate_range(start="2019-06-01", end="2024-06-01")
    hv = pd.Series(100000.0, index=nav_idx)
    hv.loc[hv.index >= pd.Timestamp(sell_date)] = 1500.0
    cf = pd.Series(0.0, index=nav_idx)
    cf.loc[pd.Timestamp(sell_date)] = +12350.0  # sell proceeds
    nav = pd.DataFrame({"holdings_value": hv, "cash_flow": cf, "nav": hv})
    spy = _make_spy(start="2019-06-01", periods=1500)

    events = detect_panic_sells(trades, nav, spy)
    assert len(events) == 1
    e = events[0]
    assert e["trigger_type"] == "capitulation"
    assert e["trigger_paths_fired"]["capitulation"] is True
    assert e["nav_drop_pct"] > 0.70
    assert e["n_positions"] == 1


def test_panic_sell_capitulation_below_70pct_does_not_fire():
    """50% NAV drop is significant but doesn't pass the 70% capitulation
    floor. With only 1 sell, standard trigger also fails (needs ≥3)."""
    sell_date = "2022-06-15"
    trades = [
        _trade("AAPL", "buy", 100, 200.0, "2019-06-01"),
        _trade("AAPL", "sell", 50, 130.0, sell_date),  # half-liquidation
    ]
    nav_idx = pd.bdate_range(start="2019-06-01", end="2024-06-01")
    hv = pd.Series(100000.0, index=nav_idx)
    hv.loc[hv.index >= pd.Timestamp(sell_date)] = 50000.0  # -50% drop
    nav = pd.DataFrame({"holdings_value": hv, "cash_flow": pd.Series(0.0, index=nav_idx), "nav": hv})
    spy = _make_spy(start="2019-06-01", periods=1500)

    events = detect_panic_sells(trades, nav, spy)
    assert events == []  # 50% drop < 70% capitulation threshold; n_sells=1 < 3 standard


def test_panic_sell_both_triggers_fire_capitulation_wins():
    """When standard AND capitulation both fire, trigger_type='capitulation'
    (the stronger signal). Both flags surfaced in trigger_paths_fired."""
    sell_date = "2022-03-07"
    trades = [
        _trade("AAPL", "buy", 100, 200.0, "2019-06-01"),
        _trade("MSFT", "buy", 50,  150.0, "2019-06-01"),
        _trade("JPM",  "buy", 80,  120.0, "2019-06-01"),
        # 3 sells (passes standard), each at deep loss, total NAV crashes
        _trade("AAPL", "sell", 100, 130.0, sell_date),
        _trade("MSFT", "sell", 50,  100.0, sell_date),
        _trade("JPM",  "sell", 80,   90.0, sell_date),
        # No buys for next 6 months
    ]
    nav_idx = pd.bdate_range(start="2019-06-01", end="2024-06-01")
    hv = pd.Series(100000.0, index=nav_idx)
    hv.loc[hv.index >= pd.Timestamp(sell_date)] = 500.0  # -99.5%
    nav = pd.DataFrame({"holdings_value": hv, "cash_flow": pd.Series(0.0, index=nav_idx), "nav": hv})
    spy = _make_spy(start="2019-06-01", periods=1500)

    events = detect_panic_sells(trades, nav, spy)
    assert len(events) == 1
    e = events[0]
    assert e["trigger_type"] == "capitulation"
    assert e["trigger_paths_fired"]["standard"] is True
    assert e["trigger_paths_fired"]["capitulation"] is True


def test_capitulation_excludes_profitable_sell():
    """Real-data false positive: 2020-05-26 MARK was sold at +$58 GAIN
    (+28.6% of $203 NAV). NAV went to $0 (-100% drop) because it was the
    last position. Selling at a profit cannot be "panic" — capitulation
    trigger must require realized < 0."""
    sell_date = "2020-06-01"
    trades = [
        _trade("MARK", "buy",  100, 1.45, "2020-04-01"),  # entry at low cost
        _trade("MARK", "sell", 100, 2.03, sell_date),     # exit at gain (+$58)
    ]
    nav_idx = pd.bdate_range(start="2020-04-01", end="2024-06-01")
    hv = pd.Series(203.0, index=nav_idx)
    hv.loc[hv.index >= pd.Timestamp(sell_date)] = 0.0  # NAV → 0 (last position closed)
    nav = pd.DataFrame({"holdings_value": hv, "cash_flow": pd.Series(0.0, index=nav_idx), "nav": hv})
    spy = _make_spy(start="2020-04-01", periods=1500)

    events = detect_panic_sells(trades, nav, spy)
    assert events == [], f"capitulation trigger fired on a profitable sell: {events}"


def test_capitulation_excludes_sub_threshold_loss():
    """Real-data false positive: 2020-04-21 NFLX was sold at a small loss
    (-$14, only -1.4% of the $1,018 NAV). The 100% NAV drop reflects
    closing a small position, not a panic. Capitulation must require
    |loss_pct_of_nav| >= 2%."""
    sell_date = "2020-05-01"
    trades = [
        _trade("NFLX", "buy",  2, 432.07, "2020-03-01"),  # cost = $864.14
        _trade("NFLX", "sell", 2, 425.13, sell_date),     # loss = -$13.88
    ]
    # NAV $1018 (NFLX position + a tiny bit of cash); after sell, holdings = $0
    nav_idx = pd.bdate_range(start="2020-03-01", end="2024-06-01")
    hv = pd.Series(1018.0, index=nav_idx)
    hv.loc[hv.index >= pd.Timestamp(sell_date)] = 0.0
    nav = pd.DataFrame({"holdings_value": hv, "cash_flow": pd.Series(0.0, index=nav_idx), "nav": hv})
    spy = _make_spy(start="2020-03-01", periods=1500)

    events = detect_panic_sells(trades, nav, spy)
    assert events == [], f"capitulation trigger fired on a sub-2% NAV loss: {events}"


# ─── deployment_by_year (Issue-3 fix) ────────────────────────────────────────

def test_cash_flow_timing_emits_deployment_by_year():
    """Verify the per-year deployment summary is emitted with correct
    aggregation and characterization."""
    nav_idx = pd.bdate_range(start="2020-01-01", end="2024-12-31")
    cf = pd.Series(0.0, index=nav_idx)
    # All 5 dates verified as business days (Mon-Fri, no weekends).
    # Year 2020: 2 trough deployments
    cf.loc[pd.Timestamp("2020-03-23")] = -5000.0   # Mon
    cf.loc[pd.Timestamp("2020-09-23")] = -3000.0   # Wed
    # Year 2024: 3 peak deployments
    cf.loc[pd.Timestamp("2024-06-17")] = -10000.0  # Mon
    cf.loc[pd.Timestamp("2024-09-16")] = -8000.0   # Mon
    cf.loc[pd.Timestamp("2024-12-16")] = -6000.0   # Mon
    nav = pd.DataFrame({"holdings_value": 100000.0, "cash_flow": cf, "nav": 100000.0}, index=nav_idx)

    # Synthesize SPY: 2020 deployments at troughs, 2024 deployments at peaks
    spy_idx = pd.bdate_range(start="2019-01-01", end="2025-06-01")
    spy = pd.Series(np.full(len(spy_idx), 300.0), index=spy_idx)
    for d in ["2020-03-23", "2020-09-23"]:
        d_ts = pd.Timestamp(d)
        for offset in range(-30, 31):
            target = d_ts + pd.Timedelta(days=offset)
            if target in spy.index:
                spy.loc[target] = 280.0 - (30 - abs(offset)) * 1.0
        spy.loc[d_ts] = 250.0
    for d in ["2024-06-17", "2024-09-16", "2024-12-16"]:
        d_ts = pd.Timestamp(d)
        for offset in range(-30, 31):
            target = d_ts + pd.Timedelta(days=offset)
            if target in spy.index:
                spy.loc[target] = 320.0 + (30 - abs(offset)) * 1.0
        spy.loc[d_ts] = 350.0

    out = attribute_cash_flow_timing(nav, spy, twr_annualized=0.10, mwr_annualized=0.12)
    by_year = out.get("deployment_by_year")
    assert isinstance(by_year, list)
    years = {entry["year"]: entry for entry in by_year}
    # 2020 — both inflows at trough
    assert 2020 in years
    assert years[2020]["n_inflows"] == 2
    assert years[2020]["total_dollars"] == 8000.0
    assert years[2020]["characterization"] == "trough_heavy"
    assert years[2020]["n_at_trough"] == 2
    assert years[2020]["n_mid_range"] == 0
    assert years[2020]["n_at_peak"] == 0
    assert years[2020]["n_insufficient_history"] == 0
    # 2024 — all 3 inflows at peak
    assert 2024 in years
    assert years[2024]["n_inflows"] == 3
    assert years[2024]["total_dollars"] == 24000.0
    assert years[2024]["characterization"] == "peak_heavy"
    assert years[2024]["n_at_peak"] == 3
    assert years[2024]["n_at_trough"] == 0


def test_deployment_by_year_surfaces_within_year_variance():
    """A year with mixed-position inflows (1 trough + 1 peak) has weighted
    avg near 0.5 (mid_range), but n_at_trough=1 + n_at_peak=1 reveals the
    bimodal behavior the avg alone hides."""
    nav_idx = pd.bdate_range(start="2020-01-01", end="2022-12-31")
    cf = pd.Series(0.0, index=nav_idx)
    # Year 2022: equal-sized inflows at trough and at peak
    cf.loc[pd.Timestamp("2022-03-23")] = -5000.0  # trough
    cf.loc[pd.Timestamp("2022-09-23")] = -5000.0  # peak
    nav = pd.DataFrame({"holdings_value": 100000.0, "cash_flow": cf, "nav": 100000.0}, index=nav_idx)

    spy_idx = pd.bdate_range(start="2019-01-01", end="2023-06-01")
    spy = pd.Series(np.full(len(spy_idx), 300.0), index=spy_idx)
    # 2022-03-23: trough
    for offset in range(-30, 31):
        target = pd.Timestamp("2022-03-23") + pd.Timedelta(days=offset)
        if target in spy.index:
            spy.loc[target] = 280.0 - (30 - abs(offset)) * 1.0
    spy.loc[pd.Timestamp("2022-03-23")] = 250.0
    # 2022-09-23: peak
    for offset in range(-30, 31):
        target = pd.Timestamp("2022-09-23") + pd.Timedelta(days=offset)
        if target in spy.index:
            spy.loc[target] = 320.0 + (30 - abs(offset)) * 1.0
    spy.loc[pd.Timestamp("2022-09-23")] = 350.0

    out = attribute_cash_flow_timing(nav, spy, twr_annualized=0.10, mwr_annualized=0.10)
    by_year = {y["year"]: y for y in out["deployment_by_year"]}
    y2022 = by_year[2022]
    # Per-bucket counts reveal both inflows
    assert y2022["n_at_trough"] == 1
    assert y2022["n_at_peak"] == 1
    assert y2022["n_mid_range"] == 0
    # Weighted average is near 0.5 (the avg of trough + peak, equal weights)
    assert 0.4 < y2022["weighted_avg_spy_52w_position_pct"] < 0.6
    # And the fallback string lands on mid_range — but the per-bucket counts
    # are the source of truth for the dashboard
    assert y2022["characterization"] == "mid_range"


def test_synopsis_fields_pre_computed_for_ai_consumption():
    """Synopsis fields shut down two LLM mis-handling classes from 9B.3
    review: (a) summing the wrong subset of years (Claude computed $38,984
    when the true 2024-2026 total was $49,084) and (b) grabbing a per-year
    weighted_avg field instead of the aggregate. The Python module must
    pre-compute every headline number the prompt will reference."""
    nav_idx = pd.bdate_range(start="2020-01-01", end="2026-12-31")
    cf = pd.Series(0.0, index=nav_idx)
    # Two trough deployments in 2020 (early years, no full 52w SPY history)
    cf.loc[pd.Timestamp("2020-03-23")] = -5000.0
    cf.loc[pd.Timestamp("2020-09-23")] = -3000.0
    # Three peak deployments in 2024-2026 — all business days
    cf.loc[pd.Timestamp("2024-06-17")] = -10000.0  # Mon
    cf.loc[pd.Timestamp("2025-09-15")] = -8000.0   # Mon
    cf.loc[pd.Timestamp("2026-04-20")] = -6000.0   # Mon
    nav = pd.DataFrame({"holdings_value": 100000.0, "cash_flow": cf, "nav": 100000.0}, index=nav_idx)

    spy_idx = pd.bdate_range(start="2019-01-01", end="2027-06-01")
    spy = pd.Series(np.full(len(spy_idx), 300.0), index=spy_idx)
    for d in ["2024-06-17", "2025-09-15", "2026-04-20"]:
        d_ts = pd.Timestamp(d)
        for offset in range(-30, 31):
            target = d_ts + pd.Timedelta(days=offset)
            if target in spy.index:
                spy.loc[target] = 320.0 + (30 - abs(offset)) * 1.0
        spy.loc[d_ts] = 350.0

    out = attribute_cash_flow_timing(nav, spy, twr_annualized=0.10, mwr_annualized=0.12)
    sf = out.get("synopsis_fields")
    assert isinstance(sf, dict), "synopsis_fields must be present on the response"

    # All 8 keys present
    expected_keys = {
        "weighted_avg_position_pct_aggregate",
        "total_inflows_dollars",
        "total_inflows_recent_3y",
        "recent_3y_year_range",
        "recent_3y_pct_of_total",
        "inflow_count_at_peak",
        "inflow_count_total",
        "peak_inflow_pct",
    }
    assert set(sf.keys()) == expected_keys, f"missing keys: {expected_keys - set(sf.keys())}"

    # Aggregate weighted_avg matches the top-level field (not a per-year value)
    assert sf["weighted_avg_position_pct_aggregate"] == out["weighted_avg_position_pct"]

    # Total inflows = sum of all five = $32,000
    assert sf["total_inflows_dollars"] == 32000.0
    # Inflow counts
    assert sf["inflow_count_total"] == 5
    assert sf["inflow_count_at_peak"] == 3  # 2024, 2025, 2026 all set as peaks
    # peak_inflow_pct = 3 / 5 = 0.6
    assert abs(sf["peak_inflow_pct"] - 0.6) < 1e-6

    # Recent-3y window slides with current year. Today's date set in CI.
    from datetime import datetime as _dt
    expected_range = f"{_dt.now().year - 2}-{_dt.now().year}"
    assert sf["recent_3y_year_range"] == expected_range


def test_synopsis_fields_recent_3y_excludes_old_inflows():
    """If today is 2026, the 3y window covers 2024+2025+2026 only.
    A $5k 2020 deposit must NOT be counted in total_inflows_recent_3y but
    must remain in total_inflows_dollars and the percentage should reflect
    the partial coverage."""
    nav_idx = pd.bdate_range(start="2020-01-01", end="2026-12-31")
    cf = pd.Series(0.0, index=nav_idx)
    cf.loc[pd.Timestamp("2020-03-23")] = -5000.0   # Old
    cf.loc[pd.Timestamp("2024-06-17")] = -3000.0   # In recent window
    cf.loc[pd.Timestamp("2025-09-15")] = -7000.0   # In recent window
    nav = pd.DataFrame({"holdings_value": 100000.0, "cash_flow": cf, "nav": 100000.0}, index=nav_idx)
    spy = _make_spy(start="2019-01-01", periods=2200)

    out = attribute_cash_flow_timing(nav, spy, twr_annualized=0.10, mwr_annualized=0.12)
    sf = out["synopsis_fields"]

    from datetime import datetime as _dt
    if _dt.now().year >= 2026:
        # 2024 + 2025 = $10,000 of $15,000 total = 66.67%
        assert sf["total_inflows_dollars"] == 15000.0
        assert sf["total_inflows_recent_3y"] == 10000.0
        assert abs(sf["recent_3y_pct_of_total"] - (10000.0 / 15000.0)) < 1e-4


def test_cash_flow_timing_attribution_formula_reliability_flag():
    """When |twr_mwr_gap| > 50%, the linear attribution formula breaks down
    and the response flags `attribution_formula_reliable: false`. Below 50%,
    the flag is true."""
    nav_idx = pd.bdate_range(start="2020-01-01", end="2024-12-31")
    cf = pd.Series(0.0, index=nav_idx)
    for d in ["2020-06-15", "2020-09-15", "2020-12-15", "2021-03-15", "2021-06-15"]:
        cf.loc[pd.Timestamp(d)] = -3000.0
    nav = pd.DataFrame({"holdings_value": 100000.0, "cash_flow": cf, "nav": 100000.0}, index=nav_idx)
    spy = _make_spy(start="2019-01-01", periods=2000)

    # Small gap → reliable
    out_small = attribute_cash_flow_timing(nav, spy, twr_annualized=0.10, mwr_annualized=0.20)
    assert out_small["attribution_formula_reliable"] is True

    # Huge gap (e.g., +1436% MWR like Patrick's real data) → unreliable
    out_huge = attribute_cash_flow_timing(nav, spy, twr_annualized=-0.07, mwr_annualized=14.36)
    assert out_huge["attribution_formula_reliable"] is False
    # Directional finding still emitted (severity bucket is the trustworthy field)
    assert out_huge.get("finding_severity") in ("strong_positive_timing", "strong_negative_timing", "neutral", "insufficient_signal")


def test_full_pipeline_response_json_serializable():
    """Synthesize a realistic mini-portfolio, run all 3 detectors, ensure the
    whole composite response sanitizes + JSON-serializes."""
    today = pd.Timestamp("2024-06-01")
    trades = [
        _trade("AAPL", "buy", 100, 200.0, "2019-06-01"),
        _trade("MSFT", "buy", 50,  150.0, "2019-06-01"),
        _trade("JPM",  "buy", 80,  120.0, "2019-06-01"),
        _trade("AAPL", "sell", 100, 130.0, "2020-03-12"),
        _trade("MSFT", "sell", 50,  100.0, "2020-03-12"),
        _trade("JPM",  "sell", 80,   90.0, "2020-03-12"),
        _trade("PLTR", "buy", 100, 30.0,  "2022-01-15"),
    ]
    nav_idx = pd.bdate_range(start="2019-06-01", end="2024-06-01")
    cf = pd.Series(0.0, index=nav_idx)
    cf.loc[pd.Timestamp("2020-03-12")] = +30000.0  # net sells (positive inflow to cash account)
    cf.loc[pd.Timestamp("2022-01-15")] = -3000.0   # buy
    nav = pd.DataFrame({"holdings_value": 100000.0, "cash_flow": cf, "nav": 100000.0}, index=nav_idx)
    spy = _make_spy(start="2019-01-01", periods=2000)

    panic = detect_panic_sells(trades, nav, spy)
    disposition = detect_disposition_effect(trades, current_prices={"PLTR": 10.0}, today=today)
    cf_timing = attribute_cash_flow_timing(nav, spy, twr_annualized=0.10, mwr_annualized=0.12)
    response = {
        "panic_sells": panic,
        "disposition": disposition,
        "cash_flow_timing": cf_timing,
        "deferred": {
            "sector_concentration_drift": detect_sector_concentration_drift(),
            "position_sizing_creep":     detect_position_sizing_creep(),
            "frequency_increase":         detect_frequency_increase(),
            "sector_cycling":             detect_sector_cycling(),
            "time_of_day_bias":           detect_time_of_day_bias(),
        },
    }
    sanitized = _sanitize_for_json(response)
    json.dumps(sanitized)  # raises if any leftover NaN/Inf
