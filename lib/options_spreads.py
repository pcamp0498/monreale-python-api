"""Multi-leg option spread detection — Sprint 9C.2.

Post-FIFO clustering. Takes a list of `closed_positions` records (one per
matched / expired option contract chunk emitted by lib/options_fifo.py) and
groups them into spread records when they look like the legs of a single
multi-leg structure (verticals, straddles, strangles, irons, calendars,
diagonals, stacked longs/shorts).

Grouping key: (user_id, underlying_ticker, open_date.date()) — same broker
day on the same underlying. Within each group we examine leg count, types,
sides, strikes, and expiration to classify the structure.

The classifier is intentionally conservative: anything ambiguous is tagged
'custom' with confidence='low' so the UI can flag it for human review
rather than mis-attribute the math.

This module never mutates input. It returns a list of spread dicts ready
for UPSERT into the option_spreads Supabase table.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from typing import Optional


def _to_date(v) -> Optional[date]:
    """Coerce a TIMESTAMPTZ/ISO string/date/datetime to a date for grouping."""
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    s = str(v)
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d").date()
        except ValueError:
            return None


def _grouping_key(p: dict) -> tuple:
    """Same broker day on the same underlying — expiration deliberately
    OMITTED so calendar/diagonal spreads (different leg expirations) can
    cluster."""
    return (
        p.get("user_id"),
        p.get("underlying_ticker"),
        _to_date(p.get("open_date")),
    )


def _classify_2leg(legs: list[dict]) -> tuple[str, str]:
    """Return (spread_type, confidence) for a 2-leg group."""
    a, b = legs
    types = {a["option_type"], b["option_type"]}
    sides = {a["position_side"], b["position_side"]}
    strikes = {float(a["strike"]), float(b["strike"])}
    expirations = {_to_date(a.get("expiration_date")), _to_date(b.get("expiration_date"))}

    # Different expirations → calendar (same strike) or diagonal (different
    # strikes). Calendars/diagonals always use the same option type.
    if len(expirations) > 1:
        if a["option_type"] == b["option_type"]:
            if len(strikes) == 1:
                return "calendar", "medium"
            return "diagonal", "medium"
        return "custom", "low"

    # Same expiration, different option types (one call + one put)
    if len(types) == 2:
        if len(sides) == 1:
            # both long → straddle (same strike) / strangle (different strikes)
            # both short → short straddle / short strangle
            side = a["position_side"]
            same_strike = len(strikes) == 1
            if side == "long":
                return ("long_straddle" if same_strike else "long_strangle"), "high"
            return ("short_straddle" if same_strike else "short_strangle"), "high"
        # Mixed sides + different types is uncommon (synthetic stock,
        # collars, etc.) — flag for review.
        return "custom", "low"

    # Same expiration + same option type
    opt_type = a["option_type"]
    if len(sides) == 2:
        # Vertical spread: one long + one short on the same underlying/expiration.
        long_leg = a if a["position_side"] == "long" else b
        short_leg = b if a["position_side"] == "long" else a
        long_strike = float(long_leg["strike"])
        short_strike = float(short_leg["strike"])
        if opt_type == "call":
            # bull call (debit): long lower, short higher
            # bear call (credit): short lower, long higher
            return ("bull_call_spread" if long_strike < short_strike else "bear_call_spread"), "high"
        # puts:
        # bull put (credit): short higher, long lower
        # bear put (debit): long higher, short lower
        return ("bear_put_spread" if long_strike > short_strike else "bull_put_spread"), "high"

    # Same expiration + same option type + same side → stacked
    side = a["position_side"]
    if opt_type == "call":
        return ("stacked_long_calls" if side == "long" else "stacked_short_calls"), "medium"
    return ("stacked_long_puts" if side == "long" else "stacked_short_puts"), "medium"


def _classify_4leg(legs: list[dict]) -> tuple[str, str]:
    """Iron condor/butterfly only — 2 calls + 2 puts with one long + one
    short on each side. Anything else → custom."""
    calls = [l for l in legs if l["option_type"] == "call"]
    puts = [l for l in legs if l["option_type"] == "put"]
    if len(calls) != 2 or len(puts) != 2:
        return "custom", "low"
    call_sides = sorted(l["position_side"] for l in calls)
    put_sides = sorted(l["position_side"] for l in puts)
    if call_sides != ["long", "short"] or put_sides != ["long", "short"]:
        return "custom", "low"
    strikes = {float(l["strike"]) for l in legs}
    if len(strikes) == 4:
        return "iron_condor", "high"
    if len(strikes) == 3:
        return "iron_butterfly", "high"
    return "custom", "low"


def detect_spreads(closed_positions: list[dict]) -> list[dict]:
    """Walk closed_positions and emit spread records for clustered legs.

    Single-leg groups are NOT emitted — they remain naked positions and the
    spread_id back-link stays NULL.

    CONV-placeholder closed_positions (outcome='conversion_unhandled') are
    skipped — they have no real math and would corrupt aggregations.
    """
    if not closed_positions:
        return []

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for p in closed_positions:
        if p.get("outcome") == "conversion_unhandled":
            continue
        groups[_grouping_key(p)].append(p)

    spreads: list[dict] = []
    for key, legs in groups.items():
        if len(legs) < 2:
            continue
        if len(legs) == 2:
            spread_type, confidence = _classify_2leg(legs)
        elif len(legs) == 4:
            spread_type, confidence = _classify_4leg(legs)
        else:
            spread_type, confidence = "custom", "low"

        # At-open net math:
        #   net_cost     = sum(long-side total_cost)     ← gross debit at open
        #   net_proceeds = sum(short-side total_proceeds) ← gross credit at open
        # Mixing total_cost across longs+shorts is meaningless because for a
        # short leg total_cost = closing BTC cost, not opening premium.
        net_cost = sum(
            float(l.get("total_cost") or 0)
            for l in legs if l.get("position_side") == "long"
        )
        net_proceeds = sum(
            float(l.get("total_proceeds") or 0)
            for l in legs if l.get("position_side") == "short"
        )
        realized_pnl = sum(float(l.get("realized_pnl") or 0) for l in legs)

        # Return basis: net debit (debit spread) or net credit (credit spread).
        basis = abs(net_cost - net_proceeds)
        realized_pnl_pct = (realized_pnl / basis * 100.0) if basis > 0 else None

        # Lifecycle: spread opens with the earliest leg, closes with the latest.
        open_dates = [_to_date(l.get("open_date")) for l in legs if _to_date(l.get("open_date"))]
        close_dates = [_to_date(l.get("close_date")) for l in legs if _to_date(l.get("close_date"))]
        open_date_min = min(open_dates) if open_dates else _to_date(legs[0].get("open_date"))
        close_date_max = max(close_dates) if close_dates else _to_date(legs[0].get("close_date"))
        days_held = (close_date_max - open_date_min).days if (close_date_max and open_date_min) else 0

        # Earliest leg expiration as the primary expiration (deterministic
        # for multi-expiration calendars/diagonals so the UNIQUE key fires).
        leg_exps = [_to_date(l.get("expiration_date")) for l in legs if _to_date(l.get("expiration_date"))]
        primary_exp = min(leg_exps) if leg_exps else None

        legs_summary = []
        leg_position_ids: list = []
        for l in legs:
            legs_summary.append({
                "id": str(l["id"]) if l.get("id") is not None else None,
                "option_type": l.get("option_type"),
                "position_side": l.get("position_side"),
                "strike": float(l.get("strike") or 0),
                "expiration_date": (
                    _to_date(l.get("expiration_date")).isoformat()
                    if _to_date(l.get("expiration_date")) else None
                ),
                "contracts": float(l.get("contracts") or 0),
                "total_cost": float(l.get("total_cost") or 0) if l.get("total_cost") is not None else None,
                "total_proceeds": float(l.get("total_proceeds") or 0) if l.get("total_proceeds") is not None else None,
                "realized_pnl": float(l.get("realized_pnl") or 0),
                "outcome": l.get("outcome"),
            })
            if l.get("id") is not None:
                leg_position_ids.append(l["id"])

        spreads.append({
            "user_id": key[0],
            "spread_type": spread_type,
            "detection_confidence": confidence,
            "underlying_ticker": key[1],
            "expiration_date": primary_exp.isoformat() if primary_exp else None,
            "open_date": open_date_min.isoformat() if open_date_min else None,
            "close_date": close_date_max.isoformat() if close_date_max else None,
            "days_held": days_held,
            "legs": legs_summary,
            "net_cost": round(net_cost, 2),
            "net_proceeds": round(net_proceeds, 2),
            "realized_pnl": round(realized_pnl, 2),
            "realized_pnl_pct": round(realized_pnl_pct, 4) if realized_pnl_pct is not None else None,
            "leg_position_ids": leg_position_ids,
        })

    return spreads
