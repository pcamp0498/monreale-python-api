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

from lib.auth import verify_api_key
from lib.options_fifo import match_options_positions
from lib.options_spreads import detect_spreads
from lib.performance_math import _sanitize_for_json

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
