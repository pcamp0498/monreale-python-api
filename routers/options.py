"""Options FIFO matching endpoints (Sprint 9C.1).

Three POST endpoints, each accepting `{options_trades: [...]}` in the
request body. The Python microservice does NOT query Supabase directly —
the Next.js side fetches options_trades for the authenticated user and
posts them here. This mirrors the convention used by /performance/calculate
and /extract/robinhood-csv.

   POST /options/positions  → open_positions list (still alive)
   POST /options/history    → closed_positions list (matched + expired)
   POST /options/summary    → aggregates: realized P&L, win rate, etc.

All numerics flow through _sanitize_for_json so any NaN/Inf becomes null.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

import os
import time
from datetime import datetime

import requests

from lib.auth import verify_api_key
from lib.options_fifo import match_options_positions
from lib.options_spreads import detect_spreads
from lib.performance_math import _sanitize_for_json
# lib.options_pricing pulls in scipy. Import is deferred into the forecast
# handler below so this router loads cleanly in environments where scipy is
# not installed (e.g. test runners that only exercise /health and /).

router = APIRouter()


def _extract_trades(body: dict) -> list[dict]:
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    trades = body.get("options_trades")
    if trades is None:
        raise HTTPException(status_code=400, detail="options_trades field required")
    if not isinstance(trades, list):
        raise HTTPException(status_code=400, detail="options_trades must be a list")
    return trades


@router.post("/positions", dependencies=[Depends(verify_api_key)])
async def options_positions(body: dict):
    """Run FIFO matching, return only the OPEN positions slice."""
    try:
        trades = _extract_trades(body)
        result = match_options_positions(trades)
        return _sanitize_for_json({
            "open_positions": result["open_positions"],
            "n_open": len(result["open_positions"]),
            "match_warnings": result["match_warnings"],
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"options match failed: {e}")


@router.post("/history", dependencies=[Depends(verify_api_key)])
async def options_history(body: dict):
    """Run FIFO matching, return only the CLOSED positions slice."""
    try:
        trades = _extract_trades(body)
        result = match_options_positions(trades)
        # Sort closed positions by close_date DESC (newest first) for the UI
        closed = sorted(
            result["closed_positions"],
            key=lambda p: str(p.get("close_date") or ""),
            reverse=True,
        )
        return _sanitize_for_json({
            "closed_positions": closed,
            "n_closed": len(closed),
            "match_warnings": result["match_warnings"],
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"options match failed: {e}")


@router.post("/match", dependencies=[Depends(verify_api_key)])
async def options_match(body: dict):
    """Full FIFO output — used by the Next.js commit endpoint to persist
    matched closed_positions back into Supabase. Unlike /positions and
    /history (which are UI slices), this returns the complete result
    including manual_review_required (CONV rows that need human triage)
    and the closed_positions list with all provenance arrays."""
    try:
        trades = _extract_trades(body)
        result = match_options_positions(trades)
        return _sanitize_for_json({
            "closed_positions": result["closed_positions"],
            "open_positions": result["open_positions"],
            "match_warnings": result["match_warnings"],
            "manual_review_required": result["manual_review_required"],
            "n_closed": len(result["closed_positions"]),
            "n_open": len(result["open_positions"]),
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"options match failed: {e}")


@router.post("/summary", dependencies=[Depends(verify_api_key)])
async def options_summary(body: dict):
    """Aggregates over the closed-positions list:
       - total_realized_pnl
       - n_closed / n_winners / n_losers / win_rate
       - avg_pnl_per_contract / avg_hold_days
       - best / worst single closed-position record
       - n_open / n_expired_worthless
    """
    try:
        trades = _extract_trades(body)
        result = match_options_positions(trades)
        closed = result["closed_positions"]
        opened = result["open_positions"]

        if not closed:
            return _sanitize_for_json({
                "n_closed": 0,
                "n_open": len(opened),
                "n_expired_worthless": 0,
                "total_realized_pnl": 0.0,
                "win_rate": None,
                "n_winners": 0,
                "n_losers": 0,
                "avg_pnl_per_contract": None,
                "avg_hold_days": None,
                "best_trade": None,
                "worst_trade": None,
                "match_warnings": result["match_warnings"],
            })

        winners = [p for p in closed if (p.get("realized_pnl") or 0) > 0]
        losers = [p for p in closed if (p.get("realized_pnl") or 0) < 0]
        expired = [p for p in closed if p.get("outcome") == "expired_worthless"]

        total_pnl = sum(float(p.get("realized_pnl") or 0) for p in closed)
        total_contracts = sum(float(p.get("contracts") or 0) for p in closed)
        avg_pnl_per_contract = (total_pnl / total_contracts) if total_contracts > 0 else None
        avg_hold_days = (
            sum(int(p.get("days_held") or 0) for p in closed) / len(closed)
            if closed else None
        )
        win_rate = (len(winners) / len(closed)) if closed else None

        best = max(closed, key=lambda p: float(p.get("realized_pnl") or 0))
        worst = min(closed, key=lambda p: float(p.get("realized_pnl") or 0))

        return _sanitize_for_json({
            "n_closed": len(closed),
            "n_open": len(opened),
            "n_expired_worthless": len(expired),
            "n_winners": len(winners),
            "n_losers": len(losers),
            "win_rate": round(win_rate, 4) if win_rate is not None else None,
            "total_realized_pnl": round(total_pnl, 2),
            "avg_pnl_per_contract": round(avg_pnl_per_contract, 2) if avg_pnl_per_contract is not None else None,
            "avg_hold_days": round(avg_hold_days, 1) if avg_hold_days is not None else None,
            "best_trade": best,
            "worst_trade": worst,
            "match_warnings": result["match_warnings"],
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"options summary failed: {e}")


@router.post("/detect-spreads", dependencies=[Depends(verify_api_key)])
async def options_detect_spreads(body: dict):
    """Cluster closed_positions into multi-leg spread records (Sprint 9C.2).

    Body shape: {closed_positions: [...]}. Each closed_position should be the
    full options_closed_positions row (id + math + dates + position-key).
    Returns the detected spreads ready for UPSERT, plus diagnostic counts."""
    try:
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="body must be a JSON object")
        positions = body.get("closed_positions")
        if positions is None:
            raise HTTPException(status_code=400, detail="closed_positions field required")
        if not isinstance(positions, list):
            raise HTTPException(status_code=400, detail="closed_positions must be a list")
        spreads = detect_spreads(positions)
        n_legs = sum(len(s.get("leg_position_ids") or []) for s in spreads)
        return _sanitize_for_json({
            "spreads": spreads,
            "n_spreads_detected": len(spreads),
            "n_legs_clustered": n_legs,
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"spread detection failed: {e}")


# ─── Sprint 9C.3.5 — pre-trade forecasting (Polygon chain + Black-Scholes) ───

POLYGON_BASE = "https://api.polygon.io"


def _polygon_underlying_price(ticker: str, api_key: str) -> float | None:
    """Fetch the most recent traded price for an underlying."""
    try:
        r = requests.get(
            f"{POLYGON_BASE}/v2/last/trade/{ticker.upper()}",
            params={"apiKey": api_key},
            timeout=8,
        )
        if r.status_code != 200:
            return None
        return float(r.json().get("results", {}).get("p") or 0) or None
    except Exception:
        return None


@router.get("/chain/{ticker}", dependencies=[Depends(verify_api_key)])
async def options_chain(ticker: str):
    """Pull the active option chain for a ticker from Polygon.

    Returns expirations within the next 365 days, each contract carrying
    bid/ask/last/volume/open_interest/iv/greeks if Polygon has them.

    Pagination is capped at 3 pages × 250 contracts/page = 750 raw entries;
    after the date-window filter that's plenty for any equity options chain
    we care about. Each Polygon GET has a 5s timeout so a single slow
    response never holds the gunicorn worker past its 30s ceiling.
    """
    t0 = time.perf_counter()
    sym = (ticker or "").upper().strip()
    print(f"[options/chain] start ticker={sym}")
    try:
        api_key = os.getenv("POLYGON_API_KEY", "")
        if not api_key:
            raise HTTPException(status_code=500, detail="POLYGON_API_KEY not configured")
        if not sym:
            raise HTTPException(status_code=400, detail="ticker required")

        today = datetime.utcnow().date()
        cutoff = today.replace(year=today.year + 1)

        contracts: list[dict] = []
        url = f"{POLYGON_BASE}/v3/snapshot/options/{sym}"
        params = {"apiKey": api_key, "limit": 250}
        pages = 0
        MAX_PAGES = 3            # 3 × 250 = 750 raw entries — enough post-filter
        ENOUGH_IN_WINDOW = 100   # if we already have this many in [today, today+365], stop

        while url and pages < MAX_PAGES:
            r = requests.get(
                url,
                params=params if pages == 0 else {"apiKey": api_key},
                timeout=5,
            )
            if r.status_code == 404:
                # No options chain for this ticker — common for illiquid names.
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                print(f"[options/chain] {sym} no chain (404) in {elapsed_ms}ms")
                return _sanitize_for_json({
                    "ticker": sym,
                    "contracts": [],
                    "underlying_price": None,
                    "n_expirations": 0,
                    "n_contracts": 0,
                    "expirations": [],
                    "available": False,
                })
            if r.status_code != 200:
                raise HTTPException(
                    status_code=502,
                    detail=f"Polygon options snapshot {r.status_code}: {r.text[:200]}",
                )
            payload = r.json()
            for entry in payload.get("results", []) or []:
                details = entry.get("details") or {}
                day = entry.get("day") or {}
                last_q = entry.get("last_quote") or {}
                greeks = entry.get("greeks") or {}
                exp_str = details.get("expiration_date")
                if not exp_str:
                    continue
                try:
                    exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                except ValueError:
                    continue
                if exp_date <= today or exp_date > cutoff:
                    continue
                contracts.append({
                    "ticker": details.get("ticker"),
                    "strike": details.get("strike_price"),
                    "expiration": exp_str,
                    "option_type": (details.get("contract_type") or "").lower(),
                    "bid": last_q.get("bid"),
                    "ask": last_q.get("ask"),
                    "last": day.get("close"),
                    "volume": day.get("volume"),
                    "open_interest": entry.get("open_interest"),
                    "iv": entry.get("implied_volatility"),
                    "delta": greeks.get("delta"),
                    "gamma": greeks.get("gamma"),
                    "theta": greeks.get("theta"),
                    "vega": greeks.get("vega"),
                })
            pages += 1
            # Early break when we already have enough contracts inside the
            # 365-day window — saves a Polygon roundtrip on liquid names.
            if len(contracts) >= ENOUGH_IN_WINDOW:
                print(f"[options/chain] {sym} early-break after page {pages} ({len(contracts)} in-window)")
                break
            url = payload.get("next_url")

        underlying_price = _polygon_underlying_price(sym, api_key)

        contracts.sort(
            key=lambda c: (c["expiration"], c.get("option_type") or "", c.get("strike") or 0)
        )
        expirations = sorted({c["expiration"] for c in contracts})

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        print(
            f"[options/chain] {sym} fetched in {elapsed_ms}ms "
            f"with {len(contracts)} contracts ({len(expirations)} expirations, {pages} pages)"
        )

        return _sanitize_for_json({
            "ticker": sym,
            "underlying_price": underlying_price,
            "contracts": contracts,
            "n_contracts": len(contracts),
            "n_expirations": len(expirations),
            "expirations": expirations,
            "available": len(contracts) > 0,
        })
    except HTTPException:
        raise
    except requests.exceptions.Timeout:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        print(f"[options/chain] {sym} TIMEOUT after {elapsed_ms}ms")
        raise HTTPException(
            status_code=504,
            detail=f"Polygon options chain request timed out for {sym} after {elapsed_ms}ms",
        )
    except Exception as e:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        print(f"[options/chain] {sym} FAILED after {elapsed_ms}ms: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"options chain fetch failed for {sym}: {e}",
        )


@router.post("/forecast", dependencies=[Depends(verify_api_key)])
async def options_forecast(body: dict):
    """Pre-trade forecast for a single-leg option position.

    Body:
        ticker, strike, expiration ("YYYY-MM-DD"), option_type, position_side,
        premium (per share, NOT per contract), contracts (default 1),
        risk_free_rate (default 0.045)

    Returns payoff_at_expiration + payoff_now arrays, greeks at current spot,
    breakeven(s), max_profit / max_loss, prob_of_profit, current spot price,
    and the IV used for the BS curve.
    """
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")

    try:
        ticker = str(body["ticker"]).upper().strip()
        strike = float(body["strike"])
        expiration = str(body["expiration"])
        option_type = str(body["option_type"]).lower()
        position_side = str(body["position_side"]).lower()
        premium = float(body["premium"])
        contracts = int(body.get("contracts") or 1)
        r = float(body.get("risk_free_rate") or 0.045)
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"invalid forecast body: {e}")

    if option_type not in ("call", "put"):
        raise HTTPException(status_code=400, detail="option_type must be 'call' or 'put'")
    if position_side not in ("long", "short"):
        raise HTTPException(status_code=400, detail="position_side must be 'long' or 'short'")
    if contracts < 1:
        raise HTTPException(status_code=400, detail="contracts must be >= 1")

    # Time to expiration in years (act/365).
    try:
        exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="expiration must be YYYY-MM-DD")
    today = datetime.utcnow().date()
    days_to_exp = (exp_date - today).days
    if days_to_exp <= 0:
        raise HTTPException(status_code=400, detail="expiration must be in the future")
    T = days_to_exp / 365.0

    # Lazy import — keeps the router loadable in scipy-less environments.
    try:
        from lib.options_pricing import (
            bs_greeks,
            breakeven,
            implied_volatility,
            payoff_at_expiration,
            payoff_now,
            prob_of_profit,
        )
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"options pricing dependency missing (scipy/numpy): {e}",
        )

    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="POLYGON_API_KEY not configured")

    S = _polygon_underlying_price(ticker, api_key)
    if S is None or S <= 0:
        raise HTTPException(status_code=502, detail=f"could not fetch underlying price for {ticker}")

    # Try to pull the contract's IV from the snapshot. Fall back to computing
    # it from the user-provided premium if Polygon doesn't have it.
    sigma = None
    market_price = None
    try:
        # Polygon contract ticker format: O:AAPL241220C00200000
        strike_str = f"{int(round(strike * 1000)):08d}"
        exp_compact = exp_date.strftime("%y%m%d")
        opt_letter = "C" if option_type == "call" else "P"
        contract_ticker = f"O:{ticker}{exp_compact}{opt_letter}{strike_str}"
        r_snap = requests.get(
            f"{POLYGON_BASE}/v3/snapshot/options/{ticker}/{contract_ticker}",
            params={"apiKey": api_key},
            timeout=8,
        )
        if r_snap.status_code == 200:
            snap = r_snap.json().get("results") or {}
            sigma = snap.get("implied_volatility")
            day = snap.get("day") or {}
            last_q = snap.get("last_quote") or {}
            mid = None
            if last_q.get("bid") is not None and last_q.get("ask") is not None:
                mid = (float(last_q["bid"]) + float(last_q["ask"])) / 2.0
            market_price = mid if mid is not None else day.get("close")
    except Exception:
        pass  # fall through to computed IV

    if sigma is None or sigma <= 0:
        # Solve for IV from the user-supplied premium.
        sigma = implied_volatility(premium, S, strike, T, r, option_type)
    if sigma is None or sigma <= 0:
        raise HTTPException(
            status_code=422,
            detail="implied volatility unavailable from Polygon and could not be solved from premium",
        )

    # S range for the chart: strike ± 30%, 100 points.
    s_min = strike * 0.7
    s_max = strike * 1.3
    s_range = [s_min + (s_max - s_min) * (i / 99) for i in range(100)]

    payoff_exp = payoff_at_expiration(s_range, strike, premium, option_type, position_side, contracts)
    payoff_t0 = payoff_now(s_range, strike, T, r, sigma, premium, option_type, position_side, contracts)

    greeks = bs_greeks(S, strike, T, r, sigma, option_type)
    # Apply position-side sign and contract-multiplier conventions to greeks
    # so the UI shows what the position *actually* moves by.
    if position_side == "short":
        greeks = {k: -v for k, v in greeks.items()}
    greeks_position = {k: v * contracts * 100 for k, v in greeks.items()}

    bes = breakeven(strike, premium, option_type)
    pop = prob_of_profit(S, strike, T, r, sigma, option_type, position_side, premium)

    # Max profit / loss — single-leg conventions
    premium_total = premium * 100 * contracts  # dollars per position
    if option_type == "call" and position_side == "long":
        max_profit = None  # unbounded
        max_loss = -premium_total
    elif option_type == "call" and position_side == "short":
        max_profit = premium_total
        max_loss = None  # unbounded (naked short call)
    elif option_type == "put" and position_side == "long":
        # Max profit when underlying → 0
        max_profit = (strike - premium) * 100 * contracts
        max_loss = -premium_total
    else:  # put short
        max_profit = premium_total
        max_loss = -((strike - premium) * 100 * contracts)

    return _sanitize_for_json({
        "ticker": ticker,
        "underlying_price": S,
        "strike": strike,
        "expiration": expiration,
        "days_to_expiration": days_to_exp,
        "option_type": option_type,
        "position_side": position_side,
        "premium_per_share": premium,
        "contracts": contracts,
        "risk_free_rate": r,
        "implied_volatility": sigma,
        "iv_source": "polygon" if market_price is not None and (market_price or 0) > 0 else "computed_from_premium",
        "market_price_per_share": market_price,
        "s_range": s_range,
        "payoff_at_expiration": payoff_exp,
        "payoff_now": payoff_t0,
        "greeks_per_share": greeks,            # signed, per-share
        "greeks_position": greeks_position,    # signed, full-position dollar greek
        "breakeven": bes,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "prob_of_profit": pop,
    })
