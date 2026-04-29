"""FIFO matching for naked-long and naked-short option positions.

Patrick's data is 99% long-only (BTO + STC + OEXP). Multi-leg spread
detection is deferred to Sprint 9C.2 — this module handles each leg as
an independent position keyed on (underlying, expiration, strike, type)
and emits closed_positions on STC/BTC matches and on OEXP for any
remaining open queue contents.

Trans codes:
    BTO  — Buy To Open    (open long)
    STC  — Sell To Close  (close long)
    STO  — Sell To Open   (open short)
    BTC  — Buy To Close   (close short)
    OEXP — Option Expired (closes any remaining open contracts at $0)
"""
from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Optional


def _position_key(t: dict) -> tuple:
    """A position is uniquely identified by its underlying + contract spec.
    NVDA $850 call ≠ NVDA $900 call ≠ NVDA $850 put."""
    return (
        t.get("underlying_ticker"),
        t.get("expiration_date"),
        float(t.get("strike") or 0),
        t.get("option_type"),
    )


def _to_dt(v) -> Optional[datetime]:
    """Coerce executed_at (str | datetime | None) to a datetime for date
    arithmetic. Returns None for anything unparseable."""
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    s = str(v)
    # Strip TZ for arithmetic — tz-naive UTC convention
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d")
        except ValueError:
            return None


def _days_held(open_dt: Optional[datetime], close_dt: Optional[datetime]) -> int:
    if not open_dt or not close_dt:
        return 0
    return max(0, (close_dt - open_dt).days)


def match_options_positions(options_trades: list[dict]) -> dict:
    """Walk a list of options trades and emit:
       - closed_positions: list of FIFO-matched close events + expirations
       - open_positions:   contracts still alive after the walk
       - match_warnings:   data-integrity issues encountered

    Each closed_position record carries the math (cost, proceeds, realized_pnl,
    days_held, position_side) and provenance (open_trade_id, close_trade_id
    where available).

    The dual-queue design (long_open / short_open per position key) keeps
    BTO/STC and STO/BTC streams isolated — a BTC against an empty short
    queue won't accidentally close a long; an STC against an empty long
    queue is logged as a warning instead.
    """
    closed_positions: list[dict] = []
    open_positions: list[dict] = []
    match_warnings: list[str] = []
    # Trades the matcher cannot reason about (CONV = contract conversion from
    # corporate action; deferred to Sprint 9C.6). They surface here so the
    # frontend can prompt the user to manually review and split / re-cost.
    manual_review_required: list[dict] = []

    if not options_trades:
        return {
            "closed_positions": [],
            "open_positions": [],
            "match_warnings": [],
            "manual_review_required": [],
        }

    # Group by position key
    grouped: dict[tuple, list[dict]] = {}
    for t in options_trades:
        if t.get("cancellation_status", "normal") != "normal":
            continue
        grouped.setdefault(_position_key(t), []).append(t)

    for key, group in grouped.items():
        # Sort chronologically (executed_at ascending) with a tiebreaker that
        # processes OPENS before CLOSES on the same date. Robinhood's activity
        # CSV lists rows newest-first; without the tiebreaker, a same-day STC
        # listed before its matching BTO would process first and emit a
        # phantom "no open long" warning. Sprint 9C.1 audit caught 20 such
        # cases on Patrick's data — all eliminated by the secondary key.
        group.sort(key=lambda t: (
            str(t.get("executed_at") or ""),
            0 if t.get("trans_code") in ("BTO", "STO") else 1,
        ))

        # FIFO queues — items are dicts {contracts, premium_per, total_amount, executed_at, trade_id}
        long_open: deque = deque()
        short_open: deque = deque()

        for t in group:
            tc = t.get("trans_code")
            contracts = float(t.get("contracts") or 0)
            premium_per = t.get("premium_per_contract")
            total_amount = t.get("total_amount")
            executed_at = t.get("executed_at")
            trade_id = t.get("id")

            if tc == "BTO":
                # Open long: enqueue
                cost = abs(float(total_amount or 0))
                long_open.append({
                    "contracts": contracts,
                    "premium_per": premium_per,
                    "total_amount": cost,  # absolute cost
                    "executed_at": executed_at,
                    "trade_id": trade_id,
                })

            elif tc == "STC":
                # Close long: dequeue FIFO until contracts satisfied
                remaining = contracts
                proceeds_total = abs(float(total_amount or 0))
                if remaining <= 0:
                    continue
                if not long_open:
                    match_warnings.append(
                        f"STC with no open long for {key} on {executed_at} ({contracts} contracts)"
                    )
                    continue
                while remaining > 1e-9 and long_open:
                    lot = long_open[0]
                    take = min(remaining, lot["contracts"])
                    # Pro-rata both legs of the math
                    chunk_cost = lot["total_amount"] * (take / lot["contracts"]) if lot["contracts"] > 0 else 0.0
                    chunk_proceeds = proceeds_total * (take / contracts) if contracts > 0 else 0.0
                    realized = chunk_proceeds - chunk_cost
                    pnl_pct = (realized / chunk_cost * 100.0) if chunk_cost > 0 else None
                    closed_positions.append({
                        "underlying_ticker": key[0],
                        "expiration_date": key[1],
                        "strike": key[2],
                        "option_type": key[3],
                        "position_side": "long",
                        "contracts": round(take, 4),
                        "total_cost": round(chunk_cost, 2),
                        "total_proceeds": round(chunk_proceeds, 2),
                        "realized_pnl": round(realized, 2),
                        "realized_pnl_pct": round(pnl_pct, 4) if pnl_pct is not None else None,
                        "outcome": "closed",
                        "open_date": lot["executed_at"],
                        "close_date": executed_at,
                        "days_held": _days_held(_to_dt(lot["executed_at"]), _to_dt(executed_at)),
                        "open_trade_ids": [lot.get("trade_id")] if lot.get("trade_id") else [],
                        "close_trade_ids": [trade_id] if trade_id else [],
                    })
                    lot["contracts"] -= take
                    lot["total_amount"] -= chunk_cost
                    remaining -= take
                    if lot["contracts"] <= 1e-9:
                        long_open.popleft()
                if remaining > 1e-9:
                    match_warnings.append(
                        f"STC oversold long for {key} on {executed_at}: "
                        f"{contracts} requested, {contracts - remaining} matched"
                    )

            elif tc == "STO":
                # Open short: enqueue
                proceeds = abs(float(total_amount or 0))
                short_open.append({
                    "contracts": contracts,
                    "premium_per": premium_per,
                    "total_amount": proceeds,  # absolute proceeds (premium received)
                    "executed_at": executed_at,
                    "trade_id": trade_id,
                })

            elif tc == "BTC":
                # Close short: dequeue FIFO
                remaining = contracts
                cost_total = abs(float(total_amount or 0))
                if remaining <= 0:
                    continue
                if not short_open:
                    match_warnings.append(
                        f"BTC with no open short for {key} on {executed_at} ({contracts} contracts)"
                    )
                    continue
                while remaining > 1e-9 and short_open:
                    lot = short_open[0]
                    take = min(remaining, lot["contracts"])
                    chunk_proceeds = lot["total_amount"] * (take / lot["contracts"]) if lot["contracts"] > 0 else 0.0
                    chunk_cost = cost_total * (take / contracts) if contracts > 0 else 0.0
                    realized = chunk_proceeds - chunk_cost  # short P&L: kept premium minus close cost
                    # Short return uses premium received (proceeds) as the basis
                    pnl_pct = (realized / chunk_proceeds * 100.0) if chunk_proceeds > 0 else None
                    closed_positions.append({
                        "underlying_ticker": key[0],
                        "expiration_date": key[1],
                        "strike": key[2],
                        "option_type": key[3],
                        "position_side": "short",
                        "contracts": round(take, 4),
                        "total_cost": round(chunk_cost, 2),
                        "total_proceeds": round(chunk_proceeds, 2),
                        "realized_pnl": round(realized, 2),
                        "realized_pnl_pct": round(pnl_pct, 4) if pnl_pct is not None else None,
                        "outcome": "closed",
                        "open_date": lot["executed_at"],
                        "close_date": executed_at,
                        "days_held": _days_held(_to_dt(lot["executed_at"]), _to_dt(executed_at)),
                        "open_trade_ids": [lot.get("trade_id")] if lot.get("trade_id") else [],
                        "close_trade_ids": [trade_id] if trade_id else [],
                    })
                    lot["contracts"] -= take
                    lot["total_amount"] -= chunk_proceeds
                    remaining -= take
                    if lot["contracts"] <= 1e-9:
                        short_open.popleft()
                if remaining > 1e-9:
                    match_warnings.append(
                        f"BTC oversold short for {key} on {executed_at}: "
                        f"{contracts} requested, {contracts - remaining} matched"
                    )

            elif tc == "OEXP":
                # Expiration: any remaining open contracts (long OR short)
                # close at $0. Long: full premium loss. Short: full premium kept.
                # Patrick's data only has long OEXPs (no covered calls/CSPs),
                # but the symmetric handling is correct for either case.
                long_remaining = sum(l["contracts"] for l in long_open)
                short_remaining = sum(s["contracts"] for s in short_open)

                if long_remaining > 0:
                    while long_open:
                        lot = long_open.popleft()
                        if lot["contracts"] <= 0:
                            continue
                        closed_positions.append({
                            "underlying_ticker": key[0],
                            "expiration_date": key[1],
                            "strike": key[2],
                            "option_type": key[3],
                            "position_side": "long",
                            "contracts": round(lot["contracts"], 4),
                            "total_cost": round(lot["total_amount"], 2),
                            "total_proceeds": 0.0,
                            "realized_pnl": round(-lot["total_amount"], 2),
                            # Expired-worthless long always loses 100% of premium paid
                            "realized_pnl_pct": -100.0 if lot["total_amount"] > 0 else None,
                            "outcome": "expired_worthless",
                            "open_date": lot["executed_at"],
                            "close_date": executed_at,
                            "days_held": _days_held(_to_dt(lot["executed_at"]), _to_dt(executed_at)),
                            "open_trade_ids": [lot.get("trade_id")] if lot.get("trade_id") else [],
                            "close_trade_ids": [trade_id] if trade_id else [],
                        })
                if short_remaining > 0:
                    while short_open:
                        lot = short_open.popleft()
                        if lot["contracts"] <= 0:
                            continue
                        closed_positions.append({
                            "underlying_ticker": key[0],
                            "expiration_date": key[1],
                            "strike": key[2],
                            "option_type": key[3],
                            "position_side": "short",
                            "contracts": round(lot["contracts"], 4),
                            "total_cost": 0.0,
                            "total_proceeds": round(lot["total_amount"], 2),
                            "realized_pnl": round(lot["total_amount"], 2),
                            # Expired-worthless short keeps 100% of premium received
                            "realized_pnl_pct": 100.0 if lot["total_amount"] > 0 else None,
                            "outcome": "expired_worthless",
                            "open_date": lot["executed_at"],
                            "close_date": executed_at,
                            "days_held": _days_held(_to_dt(lot["executed_at"]), _to_dt(executed_at)),
                            "open_trade_ids": [lot.get("trade_id")] if lot.get("trade_id") else [],
                            "close_trade_ids": [trade_id] if trade_id else [],
                        })
                if long_remaining == 0 and short_remaining == 0:
                    match_warnings.append(
                        f"OEXP with no matching open position for {key} on {executed_at}"
                    )

            elif tc == "CONV":
                # Contract conversion from corporate action (e.g., merger,
                # ticker change). Full handling deferred to Sprint 9C.6 —
                # for now emit a deterministic conversion_unhandled record
                # so it shows up in the dashboard's "needs review" tray
                # AND surface the raw trade in manual_review_required so
                # the UI can render the source row verbatim.
                closed_positions.append({
                    "underlying_ticker": key[0],
                    "expiration_date": key[1],
                    "strike": key[2],
                    "option_type": key[3],
                    "position_side": "long",  # placeholder; CONV semantics need 9C.6 to disambiguate
                    "contracts": round(contracts, 4),
                    "total_cost": None,
                    "total_proceeds": None,
                    "realized_pnl": 0.0,    # excluded from P&L totals downstream
                    "realized_pnl_pct": None,
                    "outcome": "conversion_unhandled",
                    "open_date": executed_at,
                    "close_date": executed_at,
                    "days_held": 0,
                    "open_trade_ids": [trade_id] if trade_id else [],
                    "close_trade_ids": [trade_id] if trade_id else [],
                })
                manual_review_required.append({
                    "reason": "conversion_unhandled",
                    "trans_code": tc,
                    "underlying_ticker": key[0],
                    "expiration_date": key[1],
                    "strike": key[2],
                    "option_type": key[3],
                    "contracts": contracts,
                    "executed_at": executed_at,
                    "trade_id": trade_id,
                    "deferred_to_sprint": "9C.6",
                    "note": "Contract conversion from corporate action — Sprint 9C.6 will add proper position-translation logic.",
                })

            else:
                match_warnings.append(
                    f"Unknown trans_code '{tc}' for {key} on {executed_at}"
                )

        # Anything left in either queue at end of group is an open position
        for lot in long_open:
            if lot["contracts"] <= 1e-9:
                continue
            open_positions.append({
                "underlying_ticker": key[0],
                "expiration_date": key[1],
                "strike": key[2],
                "option_type": key[3],
                "position_side": "long",
                "contracts": round(lot["contracts"], 4),
                "total_cost": round(lot["total_amount"], 2),
                "open_date": lot["executed_at"],
                "open_trade_ids": [lot.get("trade_id")] if lot.get("trade_id") else [],
            })
        for lot in short_open:
            if lot["contracts"] <= 1e-9:
                continue
            open_positions.append({
                "underlying_ticker": key[0],
                "expiration_date": key[1],
                "strike": key[2],
                "option_type": key[3],
                "position_side": "short",
                "contracts": round(lot["contracts"], 4),
                "total_proceeds": round(lot["total_amount"], 2),
                "open_date": lot["executed_at"],
                "open_trade_ids": [lot.get("trade_id")] if lot.get("trade_id") else [],
            })

    return {
        "closed_positions": closed_positions,
        "open_positions": open_positions,
        "match_warnings": match_warnings,
        "manual_review_required": manual_review_required,
    }
