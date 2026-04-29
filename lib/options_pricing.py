"""Black-Scholes pricing + Greeks + payoff diagrams.

Pure-numpy/scipy implementation. We deliberately avoid py_vollib to keep the
dependency surface small — scipy.stats.norm gives us the normal CDF/PDF, and
scipy.optimize.brentq solves for implied volatility.

All functions assume:
  - S          = current underlying price ($)
  - K          = strike ($)
  - T          = time to expiration (YEARS — convert from days/365)
  - r          = continuously-compounded risk-free rate (decimal, e.g. 0.045)
  - sigma      = annualized volatility (decimal, e.g. 0.30 for 30%)
  - option_type ∈ {"call", "put"}
  - position_side ∈ {"long", "short"}

Premium is per-share quoted (NOT per-contract). The payoff helpers multiply
by 100 internally to map per-share BS output → per-contract dollar P&L for
1 contract; multiply by `contracts` for sized positions.

Architectural rule #2: LLMs never calculate financial math. Everything here
is closed-form Black-Scholes — deterministic, verifiable.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


# ─── Black-Scholes core ──────────────────────────────────────────────────

def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        # Degenerate cases handled by the caller — return placeholders that
        # won't be used because bs_price/bs_greeks short-circuit on T<=0.
        return 0.0, 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bs_price(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> float:
    """Theoretical Black-Scholes price (per share, not per contract).

    At expiration (T<=0) returns the intrinsic value. With sigma<=0, treats as
    deterministic forward (intrinsic discounted)."""
    opt = (option_type or "").lower()
    if opt not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    if T <= 0:
        return max(S - K, 0.0) if opt == "call" else max(K - S, 0.0)

    if sigma <= 0:
        # Sigma=0 collapses BS to discounted intrinsic on the forward.
        forward = S * math.exp(-0.0 * T)  # no dividend yield assumed in v1
        if opt == "call":
            return max(forward - K * math.exp(-r * T), 0.0)
        return max(K * math.exp(-r * T) - forward, 0.0)

    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if opt == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> dict:
    """Greeks at the given (S, T, sigma).

    Returns dict with:
      delta — change in option price per $1 in underlying
      gamma — change in delta per $1 in underlying (per-share, identical for call/put)
      theta — daily time decay (NOT per-year — divided by 365 for trader-friendliness)
      vega  — change in option price per 1% (absolute) change in IV (e.g. 0.20 → 0.21)
      rho   — change in option price per 1% (absolute) change in r
    """
    opt = (option_type or "").lower()
    if opt not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

    d1, d2 = _d1_d2(S, K, T, r, sigma)
    sqrt_T = math.sqrt(T)
    pdf_d1 = float(norm.pdf(d1))

    gamma = pdf_d1 / (S * sigma * sqrt_T)
    # Vega is reported per 1% change (0.01 absolute), not per 100% change.
    vega = S * pdf_d1 * sqrt_T / 100.0

    if opt == "call":
        delta = float(norm.cdf(d1))
        # Theta per-year, then convert to per-day:
        theta_yr = (
            -S * pdf_d1 * sigma / (2 * sqrt_T)
            - r * K * math.exp(-r * T) * float(norm.cdf(d2))
        )
        rho = K * T * math.exp(-r * T) * float(norm.cdf(d2)) / 100.0
    else:
        delta = float(norm.cdf(d1)) - 1.0
        theta_yr = (
            -S * pdf_d1 * sigma / (2 * sqrt_T)
            + r * K * math.exp(-r * T) * float(norm.cdf(-d2))
        )
        rho = -K * T * math.exp(-r * T) * float(norm.cdf(-d2)) / 100.0

    theta_day = theta_yr / 365.0

    return {"delta": delta, "gamma": gamma, "theta": theta_day, "vega": vega, "rho": rho}


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
) -> Optional[float]:
    """Solve BS for sigma given a market price. Returns None on failure
    (e.g. price below intrinsic, T<=0, root-finder doesn't converge)."""
    if market_price is None or market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None
    intrinsic = (
        max(S - K, 0.0) if (option_type or "").lower() == "call" else max(K - S, 0.0)
    )
    if market_price < intrinsic - 1e-6:
        return None  # Arbitrage — no real-valued IV.

    def objective(sigma: float) -> float:
        return bs_price(S, K, T, r, sigma, option_type) - market_price

    try:
        # Bracket: 0.01% to 500% annualized vol covers anything realistic.
        return float(brentq(objective, 1e-4, 5.0, xtol=1e-5, maxiter=100))
    except (ValueError, RuntimeError):
        return None


# ─── Payoff diagrams ─────────────────────────────────────────────────────

# Multiplier — standard equity options are 100 shares per contract.
CONTRACT_MULTIPLIER = 100


def _signed_premium_per_contract(
    premium_paid: float, position_side: str
) -> float:
    """For a long position, premium is paid (negative cash flow at open).
    For a short position, premium is received (positive cash flow at open).
    Returns a signed dollar amount per contract."""
    side = (position_side or "").lower()
    if side not in ("long", "short"):
        raise ValueError(f"position_side must be 'long' or 'short', got {position_side!r}")
    cost = float(premium_paid) * CONTRACT_MULTIPLIER
    return -cost if side == "long" else cost  # long: paid out; short: received


def payoff_at_expiration(
    S_range: list[float] | np.ndarray,
    K: float,
    premium_paid: float,
    option_type: str,
    position_side: str,
    contracts: int = 1,
) -> list[float]:
    """P&L per terminal underlying price S, AT expiration.

    Long call: max(S-K, 0) * 100 - premium*100      per contract
    Short call: -max(S-K, 0) * 100 + premium*100   per contract
    Long put:  max(K-S, 0) * 100 - premium*100
    Short put: -max(K-S, 0) * 100 + premium*100
    """
    arr = np.asarray(S_range, dtype=float)
    opt = (option_type or "").lower()
    side = (position_side or "").lower()
    if opt == "call":
        intrinsic = np.maximum(arr - K, 0.0)
    else:
        intrinsic = np.maximum(K - arr, 0.0)
    intrinsic_dollars = intrinsic * CONTRACT_MULTIPLIER
    open_cash = _signed_premium_per_contract(premium_paid, side)
    if side == "long":
        pnl = intrinsic_dollars + open_cash  # open_cash is negative
    else:
        pnl = -intrinsic_dollars + open_cash  # open_cash is positive
    pnl = pnl * contracts
    return [float(x) for x in pnl]


def payoff_now(
    S_range: list[float] | np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma: float,
    premium_paid: float,
    option_type: str,
    position_side: str,
    contracts: int = 1,
) -> list[float]:
    """P&L per current underlying price S, BEFORE expiration.

    Marks each S to the BS theoretical value at the given (T, sigma) and
    subtracts/adds the premium paid/received."""
    arr = np.asarray(S_range, dtype=float)
    side = (position_side or "").lower()
    open_cash = _signed_premium_per_contract(premium_paid, side)
    bs_values = np.array(
        [bs_price(float(s), K, T, r, sigma, option_type) for s in arr],
        dtype=float,
    )
    bs_dollars = bs_values * CONTRACT_MULTIPLIER
    if side == "long":
        pnl = bs_dollars + open_cash  # marking to market — current value MINUS what we paid
    else:
        pnl = -bs_dollars + open_cash  # short: we received premium, owe BS value back
    pnl = pnl * contracts
    return [float(x) for x in pnl]


def breakeven(
    K: float, premium_paid: float, option_type: str
) -> list[float]:
    """Underlying price(s) at which P&L crosses zero AT EXPIRATION.

    Single-leg single-strike options have one breakeven:
      Call: K + premium
      Put:  K - premium
    """
    opt = (option_type or "").lower()
    p = float(premium_paid)
    if opt == "call":
        return [K + p]
    return [max(K - p, 0.0)]


def prob_of_profit(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    position_side: str,
    premium_paid: float,
) -> Optional[float]:
    """Approximate probability of finishing profitable AT EXPIRATION.

    Uses the lognormal terminal-distribution closed form:
        S_T = S * exp((r - 0.5σ²)T + σ√T·Z),   Z ~ N(0,1)

    For each side/type, the breakeven defines the threshold:
      Long call:   PoP = P(S_T > BE) where BE = K + premium
      Long put:    PoP = P(S_T < BE) where BE = K - premium
      Short call:  PoP = P(S_T < BE) where BE = K + premium
      Short put:   PoP = P(S_T > BE) where BE = K - premium
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None
    opt = (option_type or "").lower()
    side = (position_side or "").lower()
    if opt not in ("call", "put") or side not in ("long", "short"):
        return None

    BE = (K + float(premium_paid)) if opt == "call" else (K - float(premium_paid))
    if BE <= 0:
        # Degenerate — premium exceeds put strike; floor at 0 and treat as
        # always profitable on the upside / never on the downside.
        return 1.0 if (opt == "put" and side == "long") else 0.0

    # log(S_T / S) is normal with mean (r - 0.5σ²)T and stdev σ√T
    mu = (r - 0.5 * sigma * sigma) * T
    stdev = sigma * math.sqrt(T)
    z = (math.log(BE / S) - mu) / stdev

    # P(S_T > BE) = P(Z > z) = 1 - N(z)
    p_above = 1.0 - float(norm.cdf(z))
    p_below = 1.0 - p_above

    if opt == "call" and side == "long":
        return p_above
    if opt == "call" and side == "short":
        return p_below
    if opt == "put" and side == "long":
        return p_below
    return p_above  # put short
