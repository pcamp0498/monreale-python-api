"""Behavioral bias detection on a personal trade history.

Three active patterns in v1:
  1. PANIC SELL — multi-position liquidation followed by an extended cash
     period, contextualized against subsequent SPY return.
  2. HOLD-TOO-LONG / DISPOSITION EFFECT — open positions at deep losses
     held longer than winners, plus a population statistic comparing average
     holding periods of closed winners vs. losers.
  3. CASH-FLOW TIMING ATTRIBUTION — refines the TWR/MWR gap by checking
     where SPY was in its 52-week range at each significant deposit moment.

Five additional patterns return structured "insufficient_data" placeholders.

Reuses helpers from lib.performance_math (build_daily_nav, match_fifo_lots,
_to_utc_naive, _sanitize_for_json, _drop_cancelled). All datetimes flow
through tz-naive UTC. The whole response is JSON-serializable via the
existing _sanitize_for_json wrapper at the router boundary.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Optional

import math
import numpy as np
import pandas as pd

from lib.performance_math import (
    match_fifo_lots,
    _drop_cancelled,
    _to_utc_naive,
)


# ─────────────────────────────────────────────────────────────────────────────
# Named constants — every threshold and breakpoint that drives a finding
# lives here so the dashboard can render "Why severity = high?" with full
# math visible from a single API response.
# ─────────────────────────────────────────────────────────────────────────────

# Panic-Sell — STANDARD trigger thresholds
PANIC_SELL_NPOS_TRIGGER             = 3       # min sells on same day to qualify
PANIC_SELL_LOSS_TRIGGER_PCT         = 0.02    # min |realized loss / NAV|
PANIC_SELL_DAYS_TRIGGER             = 30      # idle window after panic (trading days)
PANIC_SELL_BUYBACK_THRESHOLD_PCT    = 0.10    # idle if subsequent buys < 10% of liquidated $

# Panic-Sell — CAPITULATION trigger (alternative independent path)
# Fires on near-total liquidations regardless of subsequent buyback behavior.
# Catches events like 2022-03-07 where Patrick sold 5 positions taking NAV
# from $953.98 → $4.57 (99.5% drop) but re-deployed ~18% within 14 days,
# which the standard-trigger 10% buyback floor would otherwise reject.
PANIC_SELL_NAV_CAPITULATION_THRESHOLD = 0.70  # NAV drop > 70% on a sell day

# Panic-Sell severity saturation breakpoints
PANIC_SELL_LOSS_SATURATION_PCT      = 0.10    # -10% NAV loss → loss_norm = 1.0
PANIC_SELL_NPOS_SATURATION          = 10      # 10 positions sold → n_pos_norm = 1.0
PANIC_SELL_DAYS_SATURATION          = 90      # 90 trading days idle → days_norm = 1.0
PANIC_SELL_RECOVERY_SATURATION_PCT  = 0.20    # +20% SPY 6m return → recovery_norm = 1.0

# Panic-Sell severity weights (must sum to 1.0)
PANIC_SELL_WEIGHT_LOSS              = 0.30
PANIC_SELL_WEIGHT_NPOS              = 0.20
PANIC_SELL_WEIGHT_DAYS              = 0.20
PANIC_SELL_WEIGHT_RECOVERY          = 0.30

# Panic-Sell severity bucket cutoffs (over [0, 1] severity_score)
PANIC_SELL_SEVERITY_HIGH_CUTOFF     = 0.60
PANIC_SELL_SEVERITY_MED_CUTOFF      = 0.30

# Disposition-Effect thresholds
DISPOSITION_DEEP_LOSS_PCT           = -0.50   # unrealized loss < -50%
DISPOSITION_LONG_HOLD_DAYS          = 365     # held > 1 year
DISPOSITION_RATIO_TRIGGER           = 1.5     # avg_loser_hold / avg_winner_hold > 1.5
DISPOSITION_N_FLOOR                 = 5       # need ≥5 winners AND ≥5 losers
DISPOSITION_MATERIAL_REDUCTION_PCT  = 0.10    # ≥10% of position sold = material reduction

# Cash-Flow Timing Attribution
CFT_INFLOW_DOLLAR_THRESHOLD         = 250.00  # |cash_flow| ≥ $250 to count as inflow
CFT_TROUGH_CUTOFF                   = 0.33    # SPY 52w pos < 0.33 → near-trough
CFT_PEAK_CUTOFF                     = 0.67    # SPY 52w pos > 0.67 → near-peak
CFT_MIN_INFLOWS                     = 5       # < 5 inflows → insufficient_signal
CFT_ATTRIBUTION_THRESHOLD_PCT       = 0.01    # |attribution| > 1% to call timing strong
CFT_SPY_LOOKBACK_DAYS               = 252     # 52-week trailing window
CFT_SPY_MIN_HISTORY_DAYS            = 100     # min trailing rows for valid 52w pos


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_spy_index(spy_daily: pd.Series | pd.DataFrame) -> pd.Series:
    """Coerce a SPY price input (Series or single-column DataFrame) to a
    tz-naive Series indexed by date, sorted ascending."""
    if spy_daily is None:
        return pd.Series(dtype=float)
    s = spy_daily["SPY"] if isinstance(spy_daily, pd.DataFrame) and "SPY" in spy_daily.columns else (
        spy_daily.iloc[:, 0] if isinstance(spy_daily, pd.DataFrame) else spy_daily
    )
    s = s.copy()
    s.index = _to_utc_naive(pd.to_datetime(s.index, utc=True, errors="coerce")).normalize()
    s = s[~s.index.isna()].sort_index()
    return s.dropna()


def _spy_return(spy: pd.Series, start: pd.Timestamp, days_forward: int) -> Optional[float]:
    """SPY return from `start` to `start + days_forward` calendar days, or None
    if either anchor is missing from the price index."""
    if spy is None or spy.empty:
        return None
    end = start + pd.Timedelta(days=days_forward)
    # Find the first SPY day on or after each anchor
    after_start = spy.loc[spy.index >= start]
    after_end = spy.loc[spy.index >= end]
    if after_start.empty or after_end.empty:
        return None
    p0 = float(after_start.iloc[0])
    p1 = float(after_end.iloc[0])
    if not (np.isfinite(p0) and np.isfinite(p1)) or p0 <= 0:
        return None
    return p1 / p0 - 1.0


def _trading_days_between(nav_index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Number of NAV-index entries strictly between (start, end]. Patrick's
    clarification: NAV index is SPY-aligned business days, so this is the
    canonical trading-day reference."""
    if nav_index is None or len(nav_index) == 0:
        return 0
    mask = (nav_index > start) & (nav_index <= end)
    return int(mask.sum())


# ─────────────────────────────────────────────────────────────────────────────
# 1. PANIC SELL DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _compute_panic_severity(
    loss_pct_of_nav: float,
    n_positions: int,
    days_idle_after: int,
    spy_return_6m: Optional[float],
) -> dict:
    """Pure function — returns the full severity dict from the four inputs.

    See spec: each input normalized to [0,1] via clip(value/saturation, 0, 1),
    weighted sum to a [0,1] score, bucketed via two cutoffs. When SPY 6m data
    is unavailable, the recovery weight is dropped and the remaining three
    weights are renormalized to sum to 1.0.
    """
    loss_pct_norm = min(max(abs(loss_pct_of_nav) / PANIC_SELL_LOSS_SATURATION_PCT, 0.0), 1.0)
    n_pos_norm = min(max((n_positions - PANIC_SELL_NPOS_TRIGGER) / max(PANIC_SELL_NPOS_SATURATION - PANIC_SELL_NPOS_TRIGGER, 1), 0.0), 1.0)
    days_norm = min(max(days_idle_after / PANIC_SELL_DAYS_SATURATION, 0.0), 1.0)

    if spy_return_6m is None:
        recovery_norm: Optional[float] = None
        # Reweight: drop recovery weight, redistribute proportionally.
        total_remaining = PANIC_SELL_WEIGHT_LOSS + PANIC_SELL_WEIGHT_NPOS + PANIC_SELL_WEIGHT_DAYS
        w_loss = PANIC_SELL_WEIGHT_LOSS / total_remaining
        w_npos = PANIC_SELL_WEIGHT_NPOS / total_remaining
        w_days = PANIC_SELL_WEIGHT_DAYS / total_remaining
        score = w_loss * loss_pct_norm + w_npos * n_pos_norm + w_days * days_norm
        basis = "partial_no_market_data"
        weights_used = {"loss_pct": round(w_loss, 4), "n_positions": round(w_npos, 4), "days_idle": round(w_days, 4), "market_recovery": None}
    else:
        recovery_norm = min(max(spy_return_6m / PANIC_SELL_RECOVERY_SATURATION_PCT, 0.0), 1.0)
        score = (
            PANIC_SELL_WEIGHT_LOSS * loss_pct_norm
            + PANIC_SELL_WEIGHT_NPOS * n_pos_norm
            + PANIC_SELL_WEIGHT_DAYS * days_norm
            + PANIC_SELL_WEIGHT_RECOVERY * recovery_norm
        )
        basis = "full"
        weights_used = {
            "loss_pct": PANIC_SELL_WEIGHT_LOSS,
            "n_positions": PANIC_SELL_WEIGHT_NPOS,
            "days_idle": PANIC_SELL_WEIGHT_DAYS,
            "market_recovery": PANIC_SELL_WEIGHT_RECOVERY,
        }

    if score >= PANIC_SELL_SEVERITY_HIGH_CUTOFF:
        bucket = "high"
    elif score >= PANIC_SELL_SEVERITY_MED_CUTOFF:
        bucket = "medium"
    else:
        bucket = "low"

    return {
        "severity": bucket,
        "severity_score": round(score, 4),
        "severity_basis": basis,
        "severity_factors": {
            "loss_pct_norm": round(loss_pct_norm, 4),
            "n_positions_norm": round(n_pos_norm, 4),
            "days_idle_norm": round(days_norm, 4),
            "market_recovery_norm": round(recovery_norm, 4) if recovery_norm is not None else None,
        },
        "severity_weights": weights_used,
        "severity_breakpoints": {
            "loss_pct_trigger": PANIC_SELL_LOSS_TRIGGER_PCT,
            "loss_pct_saturation": PANIC_SELL_LOSS_SATURATION_PCT,
            "n_positions_trigger": PANIC_SELL_NPOS_TRIGGER,
            "n_positions_saturation": PANIC_SELL_NPOS_SATURATION,
            "days_idle_trigger": PANIC_SELL_DAYS_TRIGGER,
            "days_idle_saturation": PANIC_SELL_DAYS_SATURATION,
            "market_recovery_saturation": PANIC_SELL_RECOVERY_SATURATION_PCT,
        },
    }


def detect_panic_sells(
    trades: list[dict],
    daily_nav: pd.DataFrame,
    spy_daily: pd.Series | pd.DataFrame | None = None,
) -> list[dict]:
    """Scan the trade history for days that fire EITHER panic-sell trigger.

    Two independent trigger paths share the same severity formula:

      STANDARD trigger — n_sells ≥ 3 AND |realized loss / NAV| > 2% AND
        subsequent buys in next 30 trading days < 10% of liquidated value.
        Captures the classic "sell-and-stay-in-cash" pattern.

      CAPITULATION trigger — n_sells ≥ 1 AND realized loss < 0 AND
        |realized loss / NAV| > 2% AND NAV drops > 70% on the same calendar
        day. Captures near-total liquidations regardless of subsequent
        buyback behavior. The realized-loss + 2%-of-NAV gates suppress
        false positives that would otherwise fire on profitable sells (you
        can't "panic" out of a position at a profit) and on micro-positions
        where 100% NAV drop reflects closing a sub-2% sleeve, not a
        meaningful capitulation. Each event includes a `trigger_type`
        field ("standard" | "capitulation"). When both triggers fire on
        the same day, "capitulation" wins (it's the stronger signal).

    Cancelled trades are filtered upstream.
    """
    trades = _drop_cancelled(trades or [])
    if not trades or daily_nav is None or daily_nav.empty:
        return []

    spy = _normalize_spy_index(spy_daily) if spy_daily is not None else pd.Series(dtype=float)
    nav_idx = daily_nav.index
    nav_series = daily_nav.get("nav") if "nav" in daily_nav else daily_nav.iloc[:, 0]

    # Index trades by calendar date (yyyy-mm-dd string) for grouping.
    df = pd.DataFrame(trades).dropna(subset=["executed_at", "ticker", "action"])
    df["date"] = _to_utc_naive(pd.to_datetime(df["executed_at"], utc=True, errors="coerce")).dt.normalize()
    df = df.dropna(subset=["date"])
    df["amount"] = pd.to_numeric(df.get("amount"), errors="coerce")

    sells_only = df[df["action"] == "sell"]
    sell_counts = sells_only.groupby("date").size()

    # Pre-compute FIFO closed lots once across the full history. Each closed
    # lot has exit_date, ticker, pnl_dollars — exactly what we need to look
    # up realized loss for any given calendar day.
    closed_all = match_fifo_lots(trades)
    closed_by_date: dict[str, list[dict]] = {}
    for c in closed_all:
        closed_by_date.setdefault(c.get("exit_date", ""), []).append(c)

    events: list[dict] = []
    for date_ts, n_sells in sell_counts.items():
        # Both triggers require at least 1 sell — sell_counts is already
        # restricted to days with ≥1 sell, so no early-exit needed here.
        date_str = date_ts.strftime("%Y-%m-%d")

        # ── Realized loss for this day (from FIFO closed lots) ──
        day_lots = closed_by_date.get(date_str, [])
        realized = sum(float(l.get("pnl_dollars") or 0) for l in day_lots)

        # ── NAV at start-of-day = NAV on most recent prior business day ──
        prior = nav_idx[nav_idx < date_ts]
        if len(prior) == 0:
            continue  # no prior NAV reference; can't size loss
        nav_start = float(nav_series.loc[prior[-1]])
        if nav_start <= 0:
            continue

        loss_pct_of_nav = realized / nav_start  # negative number when realized<0

        # ── NAV at end-of-day for capitulation check ──
        same_day_idx = nav_idx[nav_idx == date_ts]
        if len(same_day_idx) > 0:
            nav_end = float(nav_series.loc[same_day_idx[0]])
            nav_drop_pct = (nav_start - nav_end) / nav_start
        else:
            nav_end = nav_start
            nav_drop_pct = 0.0

        # ── Liquidated value (used by standard buyback check) ──
        day_sells = sells_only[sells_only["date"] == date_ts]
        liquidated_value = float(day_sells["amount"].abs().sum()) if not day_sells.empty else 0.0

        # ── Idle-window data (only computed when forward NAV is available) ──
        nav_after = nav_idx[nav_idx > date_ts]
        if len(nav_after) >= PANIC_SELL_DAYS_TRIGGER:
            idle_window_end = nav_after[PANIC_SELL_DAYS_TRIGGER - 1]
            buys_in_window = df[
                (df["action"] == "buy")
                & (df["date"] > date_ts)
                & (df["date"] <= idle_window_end)
            ]
            subsequent_buy_value = float(buys_in_window["amount"].abs().sum())
            forward_window_available = True
        else:
            idle_window_end = None
            subsequent_buy_value = 0.0
            forward_window_available = False

        # ── Evaluate both trigger paths ──
        standard_fires = (
            n_sells >= PANIC_SELL_NPOS_TRIGGER
            and realized < 0
            and abs(loss_pct_of_nav) >= PANIC_SELL_LOSS_TRIGGER_PCT
            and forward_window_available
            and (
                liquidated_value <= 0
                or subsequent_buy_value < PANIC_SELL_BUYBACK_THRESHOLD_PCT * liquidated_value
            )
        )
        capitulation_fires = (
            n_sells >= 1
            and realized < 0
            and abs(loss_pct_of_nav) >= PANIC_SELL_LOSS_TRIGGER_PCT
            and nav_drop_pct > PANIC_SELL_NAV_CAPITULATION_THRESHOLD
        )

        if not (standard_fires or capitulation_fires):
            continue

        # Capitulation wins ties — NAV-collapse is the stronger signal.
        trigger_type = "capitulation" if capitulation_fires else "standard"

        # ── days_idle_after: trading days from panic_date to next buy
        # (or end-of-data, whichever comes first) ──
        next_buy_rows = df[(df["action"] == "buy") & (df["date"] > date_ts)]
        if next_buy_rows.empty:
            next_buy_date = nav_idx[-1]
        else:
            next_buy_date = next_buy_rows["date"].min()
        days_idle = _trading_days_between(nav_idx, date_ts, next_buy_date)

        # ── Market context ──
        spy_6m = _spy_return(spy, date_ts, days_forward=180) if not spy.empty else None
        spy_12m = _spy_return(spy, date_ts, days_forward=365) if not spy.empty else None

        # ── Severity (shared formula across both trigger paths) ──
        severity = _compute_panic_severity(loss_pct_of_nav, int(n_sells), days_idle, spy_6m)
        # The breakpoints dict in severity reports STANDARD trigger thresholds;
        # add the capitulation threshold so the dashboard can render both.
        severity["severity_breakpoints"]["nav_capitulation_threshold"] = PANIC_SELL_NAV_CAPITULATION_THRESHOLD

        tickers_today = sorted(day_sells["ticker"].dropna().unique().tolist())

        event = {
            "date": date_str,
            "tickers": tickers_today,
            "n_positions": int(n_sells),
            "trigger_type": trigger_type,
            "trigger_paths_fired": {
                "standard": bool(standard_fires),
                "capitulation": bool(capitulation_fires),
            },
            "total_loss_dollars": round(realized, 2),
            "total_loss_pct_of_nav": round(loss_pct_of_nav, 6),
            "nav_start_of_day": round(nav_start, 2),
            "nav_end_of_day": round(nav_end, 2),
            "nav_drop_pct": round(nav_drop_pct, 6),
            "days_idle_after": int(days_idle),
            "liquidated_value_dollars": round(liquidated_value, 2),
            "subsequent_buy_value_dollars": round(subsequent_buy_value, 2),
            # Field names carry temporal direction explicitly. "_forward"
            # = looking forward from panic_date. If a backward-looking variant
            # is added later it will be named spy_return_6m_prior — symmetric.
            "market_context": {
                "spy_return_6m_forward": round(spy_6m, 6) if spy_6m is not None else None,
                "spy_return_12m_forward": round(spy_12m, 6) if spy_12m is not None else None,
            },
            "n_observations": 1,
        }
        event.update(severity)
        events.append(event)

    # Sort by severity_score descending so the strongest signals come first
    events.sort(key=lambda e: e.get("severity_score") or 0, reverse=True)
    return events


# ─────────────────────────────────────────────────────────────────────────────
# 2. HOLD-TOO-LONG / DISPOSITION EFFECT
# ─────────────────────────────────────────────────────────────────────────────

def _earliest_material_reduction_date(ticker_trades: list[dict]) -> Optional[str]:
    """Walk a single ticker's trades chronologically and return the date of
    the most recent sell that was ≥ DISPOSITION_MATERIAL_REDUCTION_PCT of the
    then-current position. Returns None if no material reduction ever
    occurred (i.e., a never-sold-meaningfully position)."""
    rows = sorted(
        [t for t in ticker_trades if t.get("action") in ("buy", "sell")],
        key=lambda t: str(t.get("executed_at") or "")
    )
    held = 0.0
    last_material: Optional[str] = None
    for t in rows:
        shares = float(t.get("shares") or 0)
        if t["action"] == "buy":
            held += shares
        elif t["action"] == "sell":
            if held > 0 and shares / held >= DISPOSITION_MATERIAL_REDUCTION_PCT:
                last_material = str(t.get("executed_at") or "")[:10]
            held -= shares
    return last_material


def detect_disposition_effect(
    trades: list[dict],
    current_prices: dict[str, float] | None = None,
    today: Optional[pd.Timestamp] = None,
) -> dict:
    """Return both flagged open-position list and the population disposition
    statistic comparing average winner vs. loser holding periods."""
    trades = _drop_cancelled(trades or [])
    if today is None:
        today = pd.Timestamp.now(tz="UTC").tz_convert(None).normalize()
    current_prices = current_prices or {}

    closed = match_fifo_lots(trades)

    # ── (A) Flagged open positions ─────────────────────────────────────────
    # Group trades by ticker
    by_ticker: dict[str, list[dict]] = {}
    for t in trades:
        if t.get("action") in ("buy", "sell") and t.get("ticker"):
            by_ticker.setdefault(t["ticker"], []).append(t)

    flagged: list[dict] = []
    for ticker, tx in by_ticker.items():
        buys = [t for t in tx if t["action"] == "buy"]
        sells = [t for t in tx if t["action"] == "sell"]

        # Trigger: position not reduced (no sells of this ticker) AND has open shares
        if sells:
            continue
        shares_open = sum(float(t.get("shares") or 0) for t in buys)
        if shares_open <= 0:
            continue

        cost_basis_total = sum(float(t.get("shares") or 0) * float(t.get("price") or 0) for t in buys)
        if cost_basis_total <= 0:
            continue

        current_price = current_prices.get(ticker)
        if current_price is None or not np.isfinite(current_price):
            current_value = 0.0
            unrealized_loss_pct = None
        else:
            current_value = shares_open * float(current_price)
            unrealized_loss_pct = (current_value - cost_basis_total) / cost_basis_total

        # Earliest entry date for this ticker
        earliest_str = min(str(t.get("executed_at") or "")[:10] for t in buys if t.get("executed_at"))
        try:
            earliest_dt = pd.Timestamp(earliest_str)
        except Exception:
            continue
        days_held = int((today - earliest_dt).days)

        # Trigger conditions: deep loss + long hold
        if unrealized_loss_pct is None or unrealized_loss_pct >= DISPOSITION_DEEP_LOSS_PCT:
            continue
        if days_held <= DISPOSITION_LONG_HOLD_DAYS:
            continue

        # n_subsequent_buys_same_ticker:
        # "after the most recent material reduction (≥ 10% of position sold).
        # For positions with no sells, this is just total buys since inception."
        # Since we already filtered out positions WITH sells, this equals len(buys).
        n_subsequent_buys = len(buys)

        flagged.append({
            "ticker": ticker,
            "earliest_entry_date": earliest_str,
            "days_held": days_held,
            "shares_open": round(shares_open, 8),
            "cost_basis_total": round(cost_basis_total, 2),
            "current_price": round(float(current_price), 4) if current_price is not None else None,
            "current_value": round(current_value, 2),
            "unrealized_loss_dollars": round(current_value - cost_basis_total, 2),
            "unrealized_loss_pct": round(unrealized_loss_pct, 6),
            "n_subsequent_buys_same_ticker": int(n_subsequent_buys),
        })

    flagged.sort(key=lambda f: f["unrealized_loss_pct"])  # ascending (worst first)

    # ── (B) Disposition stats over closed positions ────────────────────────
    winners = [c for c in closed if (c.get("pnl_dollars") or 0) > 0]
    losers = [c for c in closed if (c.get("pnl_dollars") or 0) < 0]
    n_w, n_l = len(winners), len(losers)

    if n_w < DISPOSITION_N_FLOOR or n_l < DISPOSITION_N_FLOOR:
        stats: Optional[dict] = None
        stats_reason = (
            f"insufficient closed positions in one or both categories "
            f"(n_winners={n_w}, n_losers={n_l}, floor={DISPOSITION_N_FLOOR})"
        )
    else:
        avg_w = float(np.mean([c.get("holding_period_days") or 0 for c in winners]))
        avg_l = float(np.mean([c.get("holding_period_days") or 0 for c in losers]))
        ratio = avg_l / avg_w if avg_w > 0 else None
        stats = {
            "avg_winner_holding_days": round(avg_w, 2),
            "avg_loser_holding_days": round(avg_l, 2),
            "disposition_ratio": round(ratio, 4) if ratio is not None else None,
            "n_winners": n_w,
            "n_losers": n_l,
            "disposition_effect_detected": bool(ratio is not None and ratio > DISPOSITION_RATIO_TRIGGER),
            "threshold_ratio": DISPOSITION_RATIO_TRIGGER,
            "n_observations_total": n_w + n_l,
        }
        stats_reason = None

    return {
        "flagged_positions": flagged,
        "disposition_stats": stats,
        "disposition_stats_reason": stats_reason,
        "n_flagged": len(flagged),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. CASH-FLOW TIMING ATTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def attribute_cash_flow_timing(
    daily_nav: pd.DataFrame,
    spy_daily: pd.Series | pd.DataFrame | None,
    twr_annualized: float,
    mwr_annualized: float,
) -> dict:
    """Compute the SPY 52-week position at each significant cash inflow and
    estimate how much of the TWR/MWR gap is attributable to deployment timing.

    v1 uses a linear heuristic:
        attribution_pct = (mwr - twr) * (1 - 2 * weighted_avg_position_pct)
    Deployment at the trough (avg_pos = 0) maps the full gap to "good timing";
    deployment at the peak (avg_pos = 1) maps it to "bad timing"; mid-range
    (0.5) maps to zero attribution. This is intentionally simple and
    transparent — a counterfactual-MWR rebuild (recompute MWR with each cash
    flow shifted to the SPY mid-range price for that period) would be more
    rigorous but materially bigger; it's marked as the v2 path.

    ATTRIBUTION FORMULA LIMIT: the linear formula is calibrated for
    twr_mwr_gap values in the 0–50% range. At extreme gaps (e.g., the
    +1436% MWR observed for a high-IRR cashflow profile), the attribution
    magnitude becomes mathematically meaningful but numerically silly
    (multiples of 100%). Above ~50% gap the response includes
    `attribution_formula_reliable: false` so the dashboard can de-emphasize
    the raw number while still surfacing the directional finding (peak vs
    trough deployment) which IS valid at any gap magnitude.

    DEFERRED TO SPRINT 9B.4: a counterfactual-MWR attribution that recomputes
    MWR with each cash flow shifted to that period's SPY mid-range price.
    The difference (true_mwr − counterfactual_mwr) gives the dollar-anchored
    timing attribution that holds at any gap magnitude. Estimated 1–2 hours
    of additional build (price-aware cash-flow shift + per-flow XIRR
    rebuilds). Tracking ticket TBD; until then v1 falls back to the
    weighted_avg_position_pct + finding_severity + deployment_by_year
    triad for any case where attribution_formula_reliable is false.
    """
    twr_mwr_gap = float(mwr_annualized) - float(twr_annualized)
    base_response = {
        "twr_annualized": round(float(twr_annualized), 6),
        "mwr_annualized": round(float(mwr_annualized), 6),
        "twr_mwr_gap_pct": round(twr_mwr_gap, 6),
    }

    if daily_nav is None or daily_nav.empty or "cash_flow" not in daily_nav:
        return {**base_response, "finding_severity": "insufficient_signal", "reason": "no daily NAV / cash_flow data"}

    spy = _normalize_spy_index(spy_daily) if spy_daily is not None else pd.Series(dtype=float)

    # Inflows = days where cash flowed INTO holdings (Robinhood convention:
    # buy = negative cash_flow because cash leaves account into stock).
    cf = daily_nav["cash_flow"].astype(float)
    inflow_days = cf[(cf < 0) & (cf.abs() >= CFT_INFLOW_DOLLAR_THRESHOLD)]
    if inflow_days.empty:
        return {**base_response, "cash_inflows": [], "inflows_summary": {"n_total": 0, "n_at_trough": 0, "n_at_peak": 0, "n_in_middle": 0, "n_with_insufficient_history": 0}, "finding_severity": "insufficient_signal", "reason": "no cash inflows above threshold"}

    inflow_records: list[dict] = []
    weighted_num = 0.0
    weighted_den = 0.0
    n_trough = n_peak = n_middle = n_insufficient = 0

    for date_ts, amount in inflow_days.items():
        rec: dict = {
            "date": date_ts.strftime("%Y-%m-%d"),
            "amount": round(float(amount), 2),
            "spy_close": None,
            "spy_52w_low": None,
            "spy_52w_high": None,
            "spy_52w_position_pct": None,
            "category": None,
        }

        if not spy.empty:
            window_start = date_ts - pd.Timedelta(days=CFT_SPY_LOOKBACK_DAYS * 2)  # generous window so we have enough rows
            window = spy.loc[(spy.index >= window_start) & (spy.index <= date_ts)]
            # Limit to the most recent CFT_SPY_LOOKBACK_DAYS rows
            if len(window) > CFT_SPY_LOOKBACK_DAYS:
                window = window.iloc[-CFT_SPY_LOOKBACK_DAYS:]

            if len(window) >= CFT_SPY_MIN_HISTORY_DAYS:
                spy_close = float(window.iloc[-1])
                lo = float(window.min())
                hi = float(window.max())
                rec["spy_close"] = round(spy_close, 4)
                rec["spy_52w_low"] = round(lo, 4)
                rec["spy_52w_high"] = round(hi, 4)
                if hi > lo:
                    pos = (spy_close - lo) / (hi - lo)
                    rec["spy_52w_position_pct"] = round(pos, 4)
                    if pos < CFT_TROUGH_CUTOFF:
                        rec["category"] = "near-trough"
                        n_trough += 1
                    elif pos > CFT_PEAK_CUTOFF:
                        rec["category"] = "near-peak"
                        n_peak += 1
                    else:
                        rec["category"] = "mid-range"
                        n_middle += 1
                    # Weighted average over absolute amount
                    weight = abs(float(amount))
                    weighted_num += weight * pos
                    weighted_den += weight
            else:
                n_insufficient += 1
        else:
            n_insufficient += 1

        inflow_records.append(rec)

    n_total = len(inflow_records)
    n_with_pos = n_total - n_insufficient

    # ── Severity ──
    if n_total < CFT_MIN_INFLOWS or n_with_pos < (n_total / 2):
        severity = "insufficient_signal"
        weighted_avg = None
        attribution_pct = None
    else:
        weighted_avg = weighted_num / weighted_den if weighted_den > 0 else None
        attribution_pct = twr_mwr_gap * (1 - 2 * weighted_avg) if weighted_avg is not None else None

        if attribution_pct is None or weighted_avg is None:
            severity = "insufficient_signal"
        elif weighted_avg < CFT_TROUGH_CUTOFF and abs(attribution_pct) > CFT_ATTRIBUTION_THRESHOLD_PCT:
            severity = "strong_positive_timing"
        elif weighted_avg > CFT_PEAK_CUTOFF and abs(attribution_pct) > CFT_ATTRIBUTION_THRESHOLD_PCT:
            severity = "strong_negative_timing"
        else:
            severity = "neutral"

    # ── Per-year deployment summary ──────────────────────────────────────
    # Patrick's 2026-04-28 review revealed that the per-event timing finding
    # alone misses a structural pattern: cash deployment scaled up over time
    # and clustered in years when SPY was already at peaks. This block
    # surfaces that as a structural-context finding alongside the per-event
    # numbers.
    #
    # SOURCE OF TRUTH: the per-bucket counts (n_at_trough / n_mid_range /
    # n_at_peak / n_insufficient_history) are the granular signal — they
    # show within-year variance that the single weighted_avg field can hide
    # (e.g., a year with one trough deposit + one peak deposit may average
    # to "trough_heavy" without telling you about the contradicting inflow).
    # The `characterization` string is a fallback summary built from the
    # weighted average, kept for backward-compat / quick rendering.
    by_year_acc: dict[int, dict] = {}
    for r in inflow_records:
        try:
            year = int(r["date"][:4])
        except (KeyError, ValueError, TypeError):
            continue
        a = by_year_acc.setdefault(year, {
            "year": year,
            "n_inflows": 0,
            "total_dollars": 0.0,
            "n_at_trough": 0,
            "n_mid_range": 0,
            "n_at_peak": 0,
            "n_insufficient_history": 0,
            "_pos_weighted_num": 0.0,
            "_pos_weighted_den": 0.0,
        })
        a["n_inflows"] += 1
        a["total_dollars"] += abs(float(r.get("amount") or 0))
        pos = r.get("spy_52w_position_pct")
        if pos is None:
            a["n_insufficient_history"] += 1
        else:
            w = abs(float(r.get("amount") or 0))
            a["_pos_weighted_num"] += w * float(pos)
            a["_pos_weighted_den"] += w
            if pos < CFT_TROUGH_CUTOFF:
                a["n_at_trough"] += 1
            elif pos > CFT_PEAK_CUTOFF:
                a["n_at_peak"] += 1
            else:
                a["n_mid_range"] += 1

    deployment_by_year: list[dict] = []
    for year in sorted(by_year_acc.keys()):
        a = by_year_acc[year]
        wd = a["_pos_weighted_den"]
        avg = (a["_pos_weighted_num"] / wd) if wd > 0 else None
        if avg is None:
            char = "insufficient_history"
        elif avg < CFT_TROUGH_CUTOFF:
            char = "trough_heavy"
        elif avg > CFT_PEAK_CUTOFF:
            char = "peak_heavy"
        else:
            char = "mid_range"
        deployment_by_year.append({
            "year": year,
            "n_inflows": a["n_inflows"],
            "total_dollars": round(a["total_dollars"], 2),
            "weighted_avg_spy_52w_position_pct": round(avg, 4) if avg is not None else None,
            # Per-bucket counts — source of truth for within-year variance
            "n_at_trough": a["n_at_trough"],
            "n_mid_range": a["n_mid_range"],
            "n_at_peak": a["n_at_peak"],
            "n_insufficient_history": a["n_insufficient_history"],
            # Fallback summary string built from the weighted_avg
            "characterization": char,
        })

    # Attribution-formula reliability: linear heuristic breaks down above ~50%
    # absolute gap. Flag this so consumers can de-emphasize the raw attribution
    # number without losing the directional finding (severity bucket).
    attribution_formula_reliable = abs(twr_mwr_gap) <= 0.50

    # ── Synopsis fields — pre-computed headline numbers ──────────────────
    # Architectural Rule #2: LLMs never calculate financial math. The frontend
    # AI-synopsis prompt should reference these fields by NAME — never derive
    # totals or percentages from the underlying inflow_records or deployment
    # rollup. Patrick's 9B.3 review caught Claude computing $38,984 instead of
    # the true $49,084 (year-sum mistake) and quoting 0.971 (a per-year value)
    # instead of 0.691 (the aggregate). All eight fields below shut down both
    # error classes.
    total_inflows_dollars = sum(
        abs(float(r.get("amount") or 0)) for r in inflow_records
    )
    inflow_count_total_acc = sum(int(y["n_inflows"]) for y in deployment_by_year)
    inflow_count_at_peak_acc = sum(int(y["n_at_peak"]) for y in deployment_by_year)

    # Recent-3y rolling window: current year + the two prior calendar years.
    # At end of 2026 → "2024-2026"; on 1 Jan 2027 → "2025-2027". Window slides
    # automatically as `datetime.now()` ticks forward — no Sprint-level edits.
    current_year = datetime.now().year
    window_start_year = current_year - 2
    recent_3y_year_range = f"{window_start_year}-{current_year}"
    total_inflows_recent_3y = sum(
        float(y["total_dollars"])
        for y in deployment_by_year
        if int(y["year"]) >= window_start_year
    )
    recent_3y_pct_of_total = (
        total_inflows_recent_3y / total_inflows_dollars
        if total_inflows_dollars > 0
        else 0.0
    )
    peak_inflow_pct = (
        inflow_count_at_peak_acc / inflow_count_total_acc
        if inflow_count_total_acc > 0
        else 0.0
    )

    synopsis_fields = {
        "weighted_avg_position_pct_aggregate": (
            round(weighted_avg, 4) if weighted_avg is not None else None
        ),
        "total_inflows_dollars": round(total_inflows_dollars, 2),
        "total_inflows_recent_3y": round(total_inflows_recent_3y, 2),
        "recent_3y_year_range": recent_3y_year_range,
        "recent_3y_pct_of_total": round(recent_3y_pct_of_total, 4),
        "inflow_count_at_peak": int(inflow_count_at_peak_acc),
        "inflow_count_total": int(inflow_count_total_acc),
        "peak_inflow_pct": round(peak_inflow_pct, 4),
    }

    return {
        **base_response,
        "weighted_avg_position_pct": round(weighted_avg, 4) if weighted_avg is not None else None,
        "attribution_pct": round(attribution_pct, 6) if attribution_pct is not None else None,
        "attribution_formula_reliable": bool(attribution_formula_reliable),
        "cash_inflows": inflow_records,
        "inflows_summary": {
            "n_total": n_total,
            "n_at_trough": n_trough,
            "n_at_peak": n_peak,
            "n_in_middle": n_middle,
            "n_with_insufficient_history": n_insufficient,
        },
        "deployment_by_year": deployment_by_year,
        "synopsis_fields": synopsis_fields,
        "finding_severity": severity,
        "n_observations": n_total,
        "thresholds": {
            "inflow_dollar_threshold": CFT_INFLOW_DOLLAR_THRESHOLD,
            "trough_cutoff": CFT_TROUGH_CUTOFF,
            "peak_cutoff": CFT_PEAK_CUTOFF,
            "min_inflows": CFT_MIN_INFLOWS,
            "attribution_threshold_pct": CFT_ATTRIBUTION_THRESHOLD_PCT,
            "attribution_reliable_max_abs_gap_pct": 0.50,
            "spy_lookback_days": CFT_SPY_LOOKBACK_DAYS,
            "spy_min_history_days": CFT_SPY_MIN_HISTORY_DAYS,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4–8. Deferred patterns — structured insufficient_data placeholders
# ─────────────────────────────────────────────────────────────────────────────

def _deferred(reason: str, sprint: str = "9C") -> dict:
    return {
        "status": "insufficient_data",
        "reason": reason,
        "deferred_to_sprint": sprint,
        "deferred": True,
    }


def detect_sector_concentration_drift(*_args, **_kwargs) -> dict:
    return _deferred(
        "Sector data not yet populated on trades; Sprint 9C will backfill from Polygon ticker_details and enable the rolling-sector-share calculation."
    )


def detect_position_sizing_creep(*_args, **_kwargs) -> dict:
    return _deferred(
        "Need a normalized position-size series across the whole history; deferred until NAV-relative sizing is added in 9C."
    )


def detect_frequency_increase(*_args, **_kwargs) -> dict:
    return _deferred(
        "Trade-frequency rolling-window analysis deferred to 9C; needs explicit period bucketing to distinguish episodic dips from a true increase trend."
    )


def detect_sector_cycling(*_args, **_kwargs) -> dict:
    return _deferred(
        "Same blocker as sector concentration drift — requires per-trade sector classification scheduled for 9C."
    )


def detect_time_of_day_bias(*_args, **_kwargs) -> dict:
    return _deferred(
        "Robinhood activity CSV does not include execution timestamps with hour-of-day precision; deferred until a brokerage source with intraday timestamps is integrated."
    )
