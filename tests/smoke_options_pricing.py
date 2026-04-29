"""Smoke verification for lib/options_pricing.

Run on production (Railway) where scipy is installed:
    python tests/smoke_options_pricing.py

Verifies BS price + Greeks + payoff diagrams + IV solver on a textbook
ATM AAPL call so any drift from canonical values is obvious.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def hr(s=""):
    print()
    print("=" * 78)
    if s:
        print(f"  {s}")
        print("=" * 78)


def main():
    try:
        from lib.options_pricing import (
            bs_price,
            bs_greeks,
            implied_volatility,
            payoff_at_expiration,
            payoff_now,
            breakeven,
            prob_of_profit,
        )
    except ImportError as e:
        raise SystemExit(f"scipy/numpy missing — install requirements first: {e}")

    # Textbook fixture: AAPL at $200, 30-day ATM call, 25% IV, 4.5% rate.
    S = 200.0
    K = 200.0
    T_days = 30
    T = T_days / 365.0
    r = 0.045
    sigma = 0.25

    hr("BS PRICE — AAPL ATM 30-day call")
    call_price = bs_price(S, K, T, r, sigma, "call")
    put_price = bs_price(S, K, T, r, sigma, "put")
    print(f"  call:  ${call_price:.4f}")
    print(f"  put:   ${put_price:.4f}")
    # Put-call parity check: C - P should equal S - K*exp(-rT)
    import math
    parity_lhs = call_price - put_price
    parity_rhs = S - K * math.exp(-r * T)
    print(f"  parity LHS (C-P):    ${parity_lhs:.4f}")
    print(f"  parity RHS (S-Ke-rT): ${parity_rhs:.4f}")
    assert abs(parity_lhs - parity_rhs) < 0.01, "put-call parity violated"
    print(f"  parity OK")

    hr("GREEKS — AAPL ATM 30-day call")
    g = bs_greeks(S, K, T, r, sigma, "call")
    for k, v in g.items():
        print(f"  {k:<6}: {v:+.6f}")
    # ATM call delta should be ~0.50 (slightly above due to drift)
    assert 0.50 < g["delta"] < 0.55, f"unexpected ATM call delta: {g['delta']}"
    # Gamma > 0
    assert g["gamma"] > 0
    # Theta < 0 for long call (decay)
    assert g["theta"] < 0
    # Vega > 0
    assert g["vega"] > 0
    print("  sanity OK")

    hr("IMPLIED VOLATILITY — round trip")
    market = call_price
    iv = implied_volatility(market, S, K, T, r, "call")
    print(f"  market price input: ${market:.4f}")
    print(f"  recovered IV:       {iv:.4f} (target 0.2500)")
    assert iv is not None and abs(iv - sigma) < 0.001, f"IV recovery failed: {iv}"
    print("  recovery OK")

    hr("PAYOFF AT EXPIRATION — long call, 1 contract")
    s_range = [180, 190, 200, 210, 220]
    payoff = payoff_at_expiration(s_range, K, market, "call", "long", contracts=1)
    for s, p in zip(s_range, payoff):
        print(f"  S=${s:>6.2f}  P&L=${p:+9.2f}")
    # At S=K, P&L = -premium*100 (lose the entire premium)
    expected_loss = -market * 100
    assert abs(payoff[2] - expected_loss) < 0.5, "ATM expiration math off"
    # Above breakeven (~K + premium)
    assert payoff[-1] > 0
    print("  shape OK")

    hr("BREAKEVEN")
    bes = breakeven(K, market, "call")
    print(f"  call breakeven: ${bes[0]:.2f} (K={K} + premium={market:.2f})")
    assert abs(bes[0] - (K + market)) < 0.01

    hr("PROBABILITY OF PROFIT — long call")
    pop = prob_of_profit(S, K, T, r, sigma, "call", "long", market)
    print(f"  PoP: {pop:.4f}  (lognormal terminal-distribution)")
    assert pop is not None and 0 < pop < 1

    hr("PAYOFF NOW vs AT EXPIRATION (S=K)")
    pn = payoff_now([K], K, T, r, sigma, market, "call", "long", contracts=1)
    pe = payoff_at_expiration([K], K, market, "call", "long", contracts=1)
    print(f"  P&L now @ K:       ${pn[0]:+.2f}  (BS theoretical)")
    print(f"  P&L at exp @ K:    ${pe[0]:+.2f}  (intrinsic)")
    assert pn[0] > pe[0], "now-curve should be above at-exp at the strike (time value)"
    print("  time-value envelope OK")

    hr("SHORT POSITION — long call inverted")
    short_pe = payoff_at_expiration([K], K, market, "call", "short", contracts=1)
    print(f"  short call P&L @ K: ${short_pe[0]:+.2f}  (should be +premium*100)")
    assert abs(short_pe[0] - market * 100) < 0.5

    hr("SMOKE OK")


if __name__ == "__main__":
    main()
