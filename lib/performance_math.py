"""Pure performance-attribution math. No FastAPI deps — only pandas/numpy/pyxirr.

Inputs are plain dicts (parser output shape). Outputs are dicts/lists ready for
JSON serialization. All arithmetic happens here; the frontend just renders.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Iterable, Optional

import math
import numpy as np
import pandas as pd

# pyxirr is available on Railway via requirements.txt
try:
    from pyxirr import xirr
except ImportError:
    xirr = None  # tests can monkey-patch if needed


# ─────────────────────────────────────────────────────────────────────────────
# Timezone normalization barrier
#
# Real Robinhood imports produce a mix of tz-naive (CSV-parsed) and tz-aware
# (Supabase TIMESTAMPTZ-fetched) ISO strings. Pandas 2.2+ refuses to compare
# them ("Cannot compare tz-naive and tz-aware timestamps"). Every entry point
# that produces a Timestamp / DatetimeIndex / Series passes through this
# helper so all downstream comparisons happen between tz-naive UTC values.
#
# The pattern: parse with `pd.to_datetime(..., utc=True)` first to coerce
# everything to tz-aware UTC, then strip the tz with this helper to get
# tz-naive UTC.
# ─────────────────────────────────────────────────────────────────────────────

def _sanitize_for_json(value):
    """Convert NaN / +Inf / -Inf to None so FastAPI's strict JSON serializer
    doesn't 500. Recurses into dicts and lists. Everything else passes through.

    Strict JSON has no NaN literal — Python's json module emits 'NaN' which is
    invalid JSON, and FastAPI's strict mode raises 'Out of range float values
    are not JSON compliant: nan'. This helper is the response-boundary
    barrier: every numeric field flows through it before the dict is returned.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, np.floating):
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(v) for v in value]
    return value


def _to_utc_naive(dt_series_or_value):
    """Convert any pd.Timestamp / Series of timestamps / DatetimeIndex to
    tz-naive UTC. If already tz-naive, return as-is.
    """
    # pandas Series — uses .dt accessor
    if hasattr(dt_series_or_value, "dt"):
        if getattr(dt_series_or_value.dt, "tz", None) is not None:
            return dt_series_or_value.dt.tz_convert("UTC").dt.tz_localize(None)
        return dt_series_or_value
    # DatetimeIndex / scalar Timestamp — both expose .tz directly
    if hasattr(dt_series_or_value, "tz") and dt_series_or_value.tz is not None:
        return dt_series_or_value.tz_convert("UTC").tz_localize(None)
    return dt_series_or_value


# ─────────────────────────────────────────────────────────────────────────────
# Defensive filter: every public function that consumes raw trades drops
# rows where cancellation_status != 'normal' so a broker-cancelled fat-finger
# (BCXL match) never feeds into FIFO, NAV, MWR, or count metrics. Default to
# 'normal' when the field is absent so callers from older code paths still
# work safely.
# ─────────────────────────────────────────────────────────────────────────────

def _drop_cancelled(trades: list[dict]) -> list[dict]:
    return [t for t in trades if t.get("cancellation_status", "normal") == "normal"]


# ─────────────────────────────────────────────────────────────────────────────
# FIFO matching
# ─────────────────────────────────────────────────────────────────────────────

def match_fifo_lots(trades: list[dict]) -> list[dict]:
    """FIFO match buys against sells per ticker.

    Cancelled trades (cancellation_status != 'normal') are filtered out
    before matching so a broker-cancelled fat-finger never closes against
    a real lot.

    Each closed position represents ONE matched chunk: the ENTIRE sell is split
    into chunks, each chunk consuming from the oldest open buy lot. A 10-share
    sell against two 6-share lots produces TWO closed positions.

    Returns list of:
        ticker, entry_date, exit_date, shares, cost_basis, proceeds,
        pnl_dollars, pnl_pct, holding_period_days, is_long_term,
        entry_trade_id, exit_trade_id  (when ids are present in input).
    """
    trades = _drop_cancelled(trades)
    # Sort once by executed_at then group per ticker
    typed = [t for t in trades if t.get("ticker") and t.get("action") in ("buy", "sell")]
    typed.sort(key=lambda t: (t.get("ticker"), str(t.get("executed_at") or "")))

    closed: list[dict] = []
    by_ticker: dict[str, list[dict]] = {}
    for t in typed:
        by_ticker.setdefault(t["ticker"], []).append(t)

    for ticker, ticker_trades in by_ticker.items():
        # Open lots: deque of [shares_remaining, price, executed_at, trade_id]
        lots: deque = deque()
        for t in ticker_trades:
            shares = float(t.get("shares") or 0)
            price = float(t.get("price") or 0)
            ex_at = str(t.get("executed_at") or "")[:10]
            tid = t.get("id")
            if shares <= 0 or not ex_at:
                continue
            if t["action"] == "buy":
                lots.append({"shares": shares, "price": price, "date": ex_at, "id": tid})
            else:  # sell
                remaining = shares
                while remaining > 0 and lots:
                    lot = lots[0]
                    take = min(remaining, lot["shares"])
                    cost_basis = take * lot["price"]
                    proceeds = take * price
                    pnl = proceeds - cost_basis
                    pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0.0
                    try:
                        d_entry = datetime.strptime(lot["date"], "%Y-%m-%d")
                        d_exit = datetime.strptime(ex_at, "%Y-%m-%d")
                        hold_days = (d_exit - d_entry).days
                    except ValueError:
                        hold_days = 0
                    closed.append({
                        "ticker": ticker,
                        "entry_date": lot["date"],
                        "exit_date": ex_at,
                        "shares": round(take, 8),
                        "cost_basis": round(cost_basis, 4),
                        "proceeds": round(proceeds, 4),
                        "pnl_dollars": round(pnl, 4),
                        "pnl_pct": round(pnl_pct, 4),
                        "holding_period_days": hold_days,
                        "is_long_term": hold_days >= 365,
                        "entry_trade_id": lot["id"],
                        "exit_trade_id": tid,
                    })
                    lot["shares"] -= take
                    remaining -= take
                    if lot["shares"] <= 1e-9:
                        lots.popleft()
                # If remaining > 0 the user oversold (short / data gap); ignore
    return closed


# ─────────────────────────────────────────────────────────────────────────────
# Daily NAV
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Sprint 9C.4 — options cash flows + inventory series
# ─────────────────────────────────────────────────────────────────────────────

def build_options_cash_flows(options_trades: list[dict]) -> list[tuple[str, float]]:
    """Sprint 9C.4 spec helper. Returns [(executed_at_iso, signed_dollar_amount)]
    sorted ascending by date.

    Sign convention follows Robinhood: BTO/BTC = negative (cash out),
    STC/STO = positive (cash in). OEXP and CONV emit no cash flow row —
    the premium was counted at open and OEXP just realizes worthlessness.

    This is the simple list-of-tuples shape; the engine helper below uses
    a richer DataFrame with FIFO-matched holdings deltas, but downstream
    consumers (e.g. MWR XIRR cash-flow series) can use this directly.
    """
    out: list[tuple[str, float]] = []
    if not options_trades:
        return out
    for t in options_trades:
        if t.get("cancellation_status", "normal") != "normal":
            continue
        tc = t.get("trans_code")
        if tc in ("OEXP", "CONV"):
            continue
        ex = t.get("executed_at")
        ta = t.get("total_amount")
        if not ex or ta is None:
            continue
        try:
            amt = float(ta)
        except (TypeError, ValueError):
            continue
        # Force the sign per trans_code. Robinhood usually stores buy=neg
        # already, but defend against parsers that flip the sign.
        if tc in ("BTO", "BTC"):
            amt = -abs(amt)
        elif tc in ("STC", "STO"):
            amt = abs(amt)
        else:
            continue  # unknown trans code — skip
        out.append((str(ex), amt))
    out.sort(key=lambda r: r[0])
    return out


def _build_options_inventory_series(
    options_trades: list[dict],
    idx: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series]:
    """Walk options trades chronologically with FIFO matching, emitting
    (cumulative_inventory_at_cost, daily_cash_flow) series indexed by `idx`.

    Long lots carry +cost; short lots carry -proceeds. So:
      - BTO @ $500: cf=-500, inventory +=500 → NAV unchanged
      - STC @ $700 closing $500 lot: cf=+700, inventory -=500 → NAV +$200
      - STO @ $400: cf=+400, inventory -=400 → NAV unchanged
      - BTC @ $200 closing $400 short: cf=-200, inventory +=400 → NAV +$200
      - OEXP long: cf=0, inventory -=lot_cost → NAV drops by full premium
      - OEXP short: cf=0, inventory +=lot_proceeds → NAV rises by kept premium
      - CONV: skipped (manual_review_required)
    """
    cf_zero = pd.Series(0.0, index=idx)
    inv_zero = pd.Series(0.0, index=idx)
    if not options_trades:
        return inv_zero, cf_zero

    # Group by position key.
    grouped: dict[tuple, list[dict]] = {}
    for t in options_trades:
        if t.get("cancellation_status", "normal") != "normal":
            continue
        if t.get("trans_code") == "CONV":
            continue
        key = (
            t.get("underlying_ticker"),
            t.get("expiration_date"),
            float(t.get("strike") or 0),
            t.get("option_type"),
        )
        grouped.setdefault(key, []).append(t)

    # Per-trade events (timestamp, cash_flow, holdings_delta).
    events: list[tuple] = []

    for key, group in grouped.items():
        # Same opens-before-closes tiebreaker as lib/options_fifo so a
        # same-day BTO+STC pair processes BTO first, not the descending-
        # CSV order.
        group.sort(key=lambda t: (
            str(t.get("executed_at") or ""),
            0 if t.get("trans_code") in ("BTO", "STO") else 1,
        ))
        long_q: list[dict] = []   # entries: {"contracts": float, "cost": float}
        short_q: list[dict] = []  # entries: {"contracts": float, "proceeds": float}

        for t in group:
            tc = t.get("trans_code")
            contracts = float(t.get("contracts") or 0)
            try:
                ta = float(t.get("total_amount") or 0)
            except (TypeError, ValueError):
                continue
            ts = t.get("executed_at")
            if not ts:
                continue

            if tc == "BTO":
                cost = abs(ta)
                if contracts > 0:
                    long_q.append({"contracts": contracts, "cost": cost})
                # cf = -cost (cash out); hv_delta = +cost (long inventory at cost)
                events.append((ts, -cost, cost))

            elif tc == "STC":
                proceeds = abs(ta)
                # FIFO close — dequeue from long_q. Track lot cost closed
                # so hv_delta drops the right amount.
                remaining = contracts
                cost_closed = 0.0
                while remaining > 1e-9 and long_q:
                    lot = long_q[0]
                    take = min(remaining, lot["contracts"])
                    chunk_cost = (
                        lot["cost"] * (take / lot["contracts"])
                        if lot["contracts"] > 0 else 0.0
                    )
                    cost_closed += chunk_cost
                    lot["contracts"] -= take
                    lot["cost"] -= chunk_cost
                    remaining -= take
                    if lot["contracts"] <= 1e-9:
                        long_q.pop(0)
                # cf = +proceeds (cash in); hv_delta = -cost_closed (drop closed lot)
                events.append((ts, proceeds, -cost_closed))

            elif tc == "STO":
                proc = abs(ta)
                if contracts > 0:
                    short_q.append({"contracts": contracts, "proceeds": proc})
                # cf = +proc (cash in); hv_delta = -proc (short = negative inv)
                events.append((ts, proc, -proc))

            elif tc == "BTC":
                cost = abs(ta)
                remaining = contracts
                proc_closed = 0.0
                while remaining > 1e-9 and short_q:
                    lot = short_q[0]
                    take = min(remaining, lot["contracts"])
                    chunk_proc = (
                        lot["proceeds"] * (take / lot["contracts"])
                        if lot["contracts"] > 0 else 0.0
                    )
                    proc_closed += chunk_proc
                    lot["contracts"] -= take
                    lot["proceeds"] -= chunk_proc
                    remaining -= take
                    if lot["contracts"] <= 1e-9:
                        short_q.pop(0)
                # cf = -cost (cash out); hv_delta = +proc_closed (release liability)
                events.append((ts, -cost, proc_closed))

            elif tc == "OEXP":
                long_total = sum(lot["cost"] for lot in long_q)
                short_total = sum(lot["proceeds"] for lot in short_q)
                long_q.clear()
                short_q.clear()
                # cf = 0 (no cash on expiration); hv_delta liquidates both queues
                events.append((ts, 0.0, -long_total + short_total))

            # CONV / unknown: skipped.

    if not events:
        return inv_zero, cf_zero

    df = pd.DataFrame(events, columns=["date", "cash_flow", "holdings_delta"])
    df["date"] = _to_utc_naive(
        pd.to_datetime(df["date"], utc=True, errors="coerce")
    ).dt.normalize()
    df = df.dropna(subset=["date"])
    if df.empty:
        return inv_zero, cf_zero

    daily = df.groupby("date").agg({"cash_flow": "sum", "holdings_delta": "sum"})
    cf_series = daily["cash_flow"].reindex(idx, fill_value=0.0)
    # Cumulative inventory — the running options-at-cost line. Reindex with
    # 0 fill, THEN cumsum, so days with no events carry the last total.
    inv_series = daily["holdings_delta"].reindex(idx, fill_value=0.0).cumsum()
    return inv_series, cf_series


def _build_options_only_nav(options_trades: list[dict]) -> pd.DataFrame:
    """Options-only NAV (scope='options'). NAV = options inventory at cost.
    Returns a DataFrame indexed by business days from earliest options trade
    to today, with columns: nav, cash_flow, holdings_value, dividends_cum.
    Empty DataFrame if no usable trades."""
    if not options_trades:
        return pd.DataFrame()

    # Gather trade dates (skip CONV / cancelled — they don't enter the math).
    dates: list[pd.Timestamp] = []
    for t in options_trades:
        if t.get("cancellation_status", "normal") != "normal":
            continue
        if t.get("trans_code") == "CONV":
            continue
        ts = t.get("executed_at")
        if not ts:
            continue
        try:
            d = pd.Timestamp(ts)
            dates.append(d)
        except (ValueError, TypeError):
            continue
    if not dates:
        return pd.DataFrame()

    start = min(dates).normalize()
    end = pd.Timestamp.now().normalize()
    if end < start:
        end = start
    idx = pd.bdate_range(start=start, end=end)
    if len(idx) == 0:
        return pd.DataFrame()
    # Strip tz to match downstream tz-naive convention.
    idx = _to_utc_naive(idx)

    inv_series, cf_series = _build_options_inventory_series(options_trades, idx)
    nav = inv_series.copy()
    out = pd.DataFrame({
        "nav": nav,
        "cash_flow": cf_series,
        "holdings_value": inv_series,
        "dividends_cum": pd.Series(0.0, index=idx),
    })
    out.attrs["tickers_skipped"] = []
    return out


def build_daily_nav(
    trades: list[dict],
    dividends: list[dict],
    options_trades: Optional[list[dict]] = None,
    scope: str = "combined",
) -> pd.DataFrame:
    """Daily portfolio NAV from trades + dividends (+ optionally options),
    valuing open positions with Polygon close prices and adding cumulative
    dividends.

    Sprint 9C.4 — `options_trades` and `scope` parameters added. Options
    enter the NAV time series via `_build_options_inventory_series`, which
    walks trades chronologically with FIFO matching and emits a parallel
    "options inventory at cost" line plus a cash_flow line. Both layer
    additively on the equity components, so the TWR formula sees a clean
    (holdings + cf) / prev_holdings - 1 ratio.

    scope ∈ {"combined" (default), "equity", "options"}:
        - combined: equity holdings + options inventory in one NAV
        - equity:   options ignored even if passed (preserves prior behavior)
        - options:  equity ignored; NAV = options inventory only

    Cancelled equity trades are filtered out before NAV construction.
    Skipped-ticker trades (Polygon couldn't price them) are also fully
    excluded — both their holdings AND their cash_flow are dropped — so
    the TWR formula doesn't see a cash flow without a corresponding
    holdings change (which would generate spurious daily returns and
    explode the chained product).

    Returns a DataFrame indexed by date with columns: nav, cash_flow,
    holdings_value, dividends_cum. Empty DataFrame if no priceable data.
    tickers_skipped surfaced via .attrs.
    """
    eq_active = scope in ("combined", "equity")
    opt_active = scope in ("combined", "options") and bool(options_trades)

    trades = _drop_cancelled(trades) if eq_active else []
    if not eq_active:
        dividends = []
    if not trades and not opt_active:
        return pd.DataFrame()
    if not trades:
        # Options-only branch — skip the equity pipeline entirely.
        return _build_options_only_nav(options_trades or [])

    df = pd.DataFrame(trades)
    df = df.dropna(subset=["executed_at", "ticker"])
    # utc=True coerces mixed tz-aware/tz-naive strings to a single tz-aware UTC
    # Series; _to_utc_naive then strips the tz so all downstream comparisons
    # against tz-naive bdate_range / Polygon indices are consistent.
    df["date"] = _to_utc_naive(pd.to_datetime(df["executed_at"], utc=True, errors="coerce")).dt.normalize()
    df = df.dropna(subset=["date"])
    df["shares_signed"] = df.apply(
        lambda r: float(r["shares"] or 0) * (1 if r["action"] == "buy" else -1 if r["action"] == "sell" else 0),
        axis=1,
    )
    df["amount"] = df.apply(
        lambda r: float(r["amount"] or 0)
        if r.get("amount") is not None
        else float(r.get("shares") or 0) * float(r.get("price") or 0) * (-1 if r["action"] == "buy" else 1),
        axis=1,
    )

    if df.empty:
        return pd.DataFrame()

    start = df["date"].min()
    end = pd.Timestamp.now().normalize()
    if end < start:
        end = start

    # Business-day index
    idx = pd.bdate_range(start=start, end=end)
    if len(idx) == 0:
        return pd.DataFrame()

    tickers = sorted([t for t in df["ticker"].dropna().unique() if t])
    if not tickers:
        return pd.DataFrame()

    # Fetch prices first, BEFORE deciding which trades feed into NAV. Skipped
    # tickers are then fully excluded from both positions AND cash_flow — this
    # is the fix for the spurious-daily-return issue (when cash_flow included
    # SPXU buys but holdings_value didn't, TWR formula treated the unmatched
    # cash flow as a price move and produced wild daily returns).
    prices_df, tickers_skipped = _fetch_prices(tickers, start, end)
    if prices_df.empty:
        empty = pd.DataFrame()
        empty.attrs["tickers_skipped"] = tickers_skipped
        return empty
    prices_df = prices_df.reindex(idx).ffill().bfill()

    skipped_set = {s.get("ticker") for s in tickers_skipped if s.get("ticker")}
    priced_tickers = [t for t in tickers if t in prices_df.columns and t not in skipped_set]
    if not priced_tickers:
        empty = pd.DataFrame()
        empty.attrs["tickers_skipped"] = tickers_skipped
        return empty

    # NAV-relevant slice: trades whose ticker we actually got prices for.
    # FIFO matching upstream still uses the unfiltered trades list — we only
    # exclude here so cash_flow stays in sync with holdings_value.
    df_nav = df[df["ticker"].isin(priced_tickers)].copy()
    if df_nav.empty:
        empty = pd.DataFrame()
        empty.attrs["tickers_skipped"] = tickers_skipped
        return empty

    # Cumulative position per (priced) ticker on each business day
    positions = pd.DataFrame(0.0, index=idx, columns=priced_tickers)
    for ticker in priced_tickers:
        td = df_nav[df_nav["ticker"] == ticker].sort_values("date")
        cum = 0.0
        for _, r in td.iterrows():
            cum += float(r["shares_signed"] or 0)
            mask = positions.index >= r["date"]
            positions.loc[mask, ticker] = cum

    # Holdings value — multiplied/summed only over priced tickers, in lockstep
    # with the cash-flow filter below.
    holdings_value = (positions[priced_tickers] * prices_df[priced_tickers]).sum(axis=1, skipna=True).fillna(0)

    # Cumulative dividends as cash drag
    div_cum = pd.Series(0.0, index=idx)
    if dividends:
        ddf = pd.DataFrame(dividends).dropna(subset=["paid_at"])
        if not ddf.empty:
            ddf["date"] = _to_utc_naive(pd.to_datetime(ddf["paid_at"], utc=True, errors="coerce")).dt.normalize()
            ddf = ddf.dropna(subset=["date"])
            if not ddf.empty:
                ddf["amount"] = ddf["amount"].astype(float)
                daily_div = ddf.groupby("date")["amount"].sum()
                div_cum = daily_div.reindex(idx, fill_value=0).cumsum()

    # Cash flow on a given day = signed amount of trades that day, restricted
    # to priced-ticker trades only (df_nav, not df) so cash_flow stays
    # symmetric with holdings_value. This is the core of the chained-product
    # explosion fix.
    daily_cf = df_nav.groupby("date")["amount"].sum().reindex(idx, fill_value=0)

    # Sprint 9C.4 — options layer. When scope ∈ {"combined", "options"} and
    # options_trades is non-empty, fold the options inventory line into
    # holdings_value and the options cash-flow line into daily_cf. The
    # holdings line carries +cost for long lots and -proceeds for short
    # lots so OEXP / STC / BTC realize P&L correctly via the same TWR
    # formula. Cash flow follows Robinhood convention (buy=neg, sell=pos)
    # so the (hv + cf) / prev_hv strip-out works identically.
    if opt_active:
        opt_inv, opt_cf = _build_options_inventory_series(options_trades or [], idx)
        holdings_value = holdings_value + opt_inv
        daily_cf = daily_cf + opt_cf

    # Forward-fill any leftover NaN in NAV from price gaps; drop leading
    # rows that are still NaN (no positions yet on those days).
    nav = (holdings_value + div_cum).ffill()
    if nav.isna().any():
        first_valid = nav.first_valid_index()
        if first_valid is None:
            empty = pd.DataFrame()
            empty.attrs["tickers_skipped"] = tickers_skipped
            return empty
        nav = nav.loc[first_valid:]
        holdings_value = holdings_value.loc[first_valid:]
        daily_cf = daily_cf.loc[first_valid:]
        div_cum = div_cum.loc[first_valid:]

    # NAV-near-zero diagnostic: log any day where NAV briefly dropped below
    # a small threshold (between trades, after a full sell, before next buy).
    # The TWR formula divides by previous NAV — values near zero produce
    # exploding daily returns. The outlier zapper catches the final symptom
    # but we want to see WHICH dates trigger so we can decide if NAV-gap
    # protection is needed.
    near_zero_mask = (nav > 0) & (nav < 10.0)
    if near_zero_mask.any():
        near_zero = nav[near_zero_mask]
        prev_nav = nav.shift(1)
        print(f"[perf] WARNING: NAV near zero on {len(near_zero)} day(s); "
              f"first 5 instances:")
        for date, val in near_zero.head(5).items():
            prev_val = prev_nav.loc[date] if date in prev_nav.index else float("nan")
            cf_val = daily_cf.loc[date] if date in daily_cf.index else 0.0
            print(f"  [perf] {date.date()}: NAV=${val:.2f}  prev_NAV=${prev_val:.2f}  cash_flow=${cf_val:.2f}")

    out = pd.DataFrame({
        "nav": nav,
        "cash_flow": daily_cf,
        "holdings_value": holdings_value,
        "dividends_cum": div_cum,
    })
    out.attrs["tickers_skipped"] = tickers_skipped
    return out


def _fetch_prices(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, list[dict]]:
    """Fetch daily closes for a list of tickers between start and end.

    Per-ticker fetch with a 30s timeout (lifted from the previous 10s — Polygon
    can be slow on multi-year fetches for low-volume tickers like SPXU). Each
    ticker that errors or times out is recorded in `skipped` and excluded from
    the returned DataFrame; the rest still flow through. Index is tz-naive UTC.

    Returns:
        (df, skipped) where df has columns for the successfully-fetched
        tickers only and skipped is a list of {ticker, reason} dicts.
    """
    import os
    import requests
    from datetime import datetime, timedelta

    skipped: list[dict] = []
    if not tickers:
        return pd.DataFrame(), skipped

    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        return pd.DataFrame(), [{"ticker": t, "reason": "no_polygon_api_key"} for t in tickers]

    days = max((end - start).days + 60, 30)
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    price_data: dict[str, dict] = {}
    for ticker in tickers:
        if not ticker:
            continue
        try:
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}"
            response = requests.get(
                url,
                params={"adjusted": "true", "sort": "asc", "limit": 5000, "apiKey": api_key},
                timeout=30,
            )
            if response.status_code != 200:
                print(f"[perf] {ticker} skipped: polygon_status_{response.status_code}")
                skipped.append({"ticker": ticker, "reason": f"polygon_status_{response.status_code}"})
                continue
            results = response.json().get("results") or []
            if not results:
                print(f"[perf] {ticker} skipped: no_price_data")
                skipped.append({"ticker": ticker, "reason": "no_price_data"})
                continue
            closes = {
                datetime.fromtimestamp(r["t"] / 1000).strftime("%Y-%m-%d"): r["c"]
                for r in results
            }
            price_data[ticker] = closes
        except requests.Timeout:
            print(f"[perf] {ticker} skipped: polygon_timeout_30s")
            skipped.append({"ticker": ticker, "reason": "polygon_timeout_30s"})
        except Exception as e:
            print(f"[perf] {ticker} skipped: {str(e)[:120]}")
            skipped.append({"ticker": ticker, "reason": f"fetch_error: {str(e)[:80]}"})

    if not price_data:
        return pd.DataFrame(), skipped

    df = pd.DataFrame(price_data)
    df.index = _to_utc_naive(pd.to_datetime(df.index))
    df = df.sort_index().dropna(how="all")
    return df, skipped


# ─────────────────────────────────────────────────────────────────────────────
# Returns
# ─────────────────────────────────────────────────────────────────────────────

# A real diversified portfolio's worst single-day drop is roughly Black Monday
# 1987 (-22%) and the largest single-day gain is similar in magnitude. Anything
# beyond ±30% is virtually certainly a data anomaly (price gap, slippage
# between Robinhood execution price and Polygon close, missed corporate action,
# transient cash_flow / holdings_value desync). We replace those with 0 — no
# return that day — instead of clipping, because a clipped -30% would still
# distort the chained product. Logged so anomalies are visible in Railway logs.
_DAILY_RET_OUTLIER_ABS = 0.30


def _zap_outliers(returns: pd.Series, label: str = "returns") -> pd.Series:
    """Replace any |return| > 30% with 0. Used for portfolio AND benchmark
    series before they enter Sharpe / Sortino / alpha / beta / max_drawdown.
    Logs the count + range to Railway so anomalies are visible in production
    logs (diagnostic print kept until we confirm clean numbers in prod)."""
    if returns is None or returns.empty:
        return returns
    cleaned = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        return cleaned
    outliers = cleaned[cleaned.abs() > _DAILY_RET_OUTLIER_ABS]
    if not outliers.empty:
        print(f"[perf] {label}: zeroed {len(outliers)} outliers > {_DAILY_RET_OUTLIER_ABS*100:.0f}% "
              f"(min={outliers.min():.4f}, max={outliers.max():.4f}, total_len={len(cleaned)})")
        cleaned = cleaned.where(cleaned.abs() <= _DAILY_RET_OUTLIER_ABS, 0.0)
    return cleaned


def _build_clean_daily_returns(daily_nav: pd.DataFrame) -> pd.Series:
    """Cash-flow-adjusted daily returns with the same outlier zap that
    compute_twr uses. Single source of truth for Sharpe / Sortino / alpha
    / beta / max_drawdown so they all agree on what "the return series" is.

    Sign convention (CRITICAL): Robinhood's parser preserves the source CSV
    convention where buy = NEGATIVE amount (cash leaves account) and
    sell = POSITIVE amount. So `cash_flow` here is signed FROM-the-portfolio:
    -1500 means $1500 of cash flowed OUT and into stock (a buy).

    The TWR contribution adjustment subtracts the contribution INTO the
    holdings basket. A buy is +1500 INTO holdings, which equals -cf when cf
    follows the Robinhood convention. So the formula is (V_t + cf_t) / V_{t-1}
    - 1, which strips out the buy/sell from the return series. The previous
    `(V_t - cf_t) / V_{t-1} - 1` had a sign flip that ADDED 2*|amt|/prev to
    every buy day, inflating mean daily return and the OLS intercept (alpha)
    for accumulating investors.
    """
    if daily_nav is None or daily_nav.empty or "holdings_value" not in daily_nav:
        return pd.Series(dtype=float)
    hv = daily_nav["holdings_value"].astype(float)
    cf = daily_nav.get("cash_flow", pd.Series(0.0, index=hv.index)).astype(float)
    prev = hv.shift(1).replace(0, np.nan)
    daily_ret = ((hv + cf) / prev) - 1
    return _zap_outliers(daily_ret, label="port_returns")


def _build_clean_nav_index(daily_returns: pd.Series, base: float = 100.0) -> pd.Series:
    """Cumulative-return equity curve (start at 100, multiply by 1+r each
    day). This is what max_drawdown should consume — the strategy's
    price-driven trajectory, NOT the raw NAV which spikes on trade days
    from capital additions."""
    if daily_returns is None or daily_returns.empty:
        return pd.Series(dtype=float)
    return base * (1 + daily_returns).cumprod()


def compute_twr(daily_nav: pd.DataFrame, periods_per_year: int = 252) -> float:
    """Annualized time-weighted return.

    Chained daily returns of holdings_value (excludes cash flows from the
    return series). Outlier daily returns (|r| > 30%) are zeroed out to
    prevent a single bad data point from blowing up the chained product.
    Returns float (e.g., 0.124 for 12.4% annualized).
    """
    if daily_nav is None or daily_nav.empty or "holdings_value" not in daily_nav:
        return 0.0
    hv = daily_nav["holdings_value"].astype(float)
    cf = daily_nav.get("cash_flow", pd.Series(0.0, index=hv.index)).astype(float)
    # Daily return: (V_t + cf_t) / V_{t-1} - 1, ignoring days where V_{t-1} = 0.
    # cf follows Robinhood convention (buy = negative cash outflow), so adding
    # cf strips out the buy/sell contribution from the return — see
    # _build_clean_daily_returns for the full sign-convention rationale.
    prev = hv.shift(1).replace(0, np.nan)
    daily_ret = ((hv + cf) / prev) - 1
    daily_ret = daily_ret.replace([np.inf, -np.inf], np.nan).dropna()
    if daily_ret.empty:
        return 0.0

    # Outlier guardrail: zero out any daily return more extreme than ±30%.
    # Logs the count once so Railway runtime logs surface the data quality.
    outliers = daily_ret[daily_ret.abs() > _DAILY_RET_OUTLIER_ABS]
    if not outliers.empty:
        print(f"[perf] zeroed {len(outliers)} TWR daily-return outliers > {_DAILY_RET_OUTLIER_ABS*100:.0f}% "
              f"(min={outliers.min():.4f}, max={outliers.max():.4f})")
        daily_ret = daily_ret.where(daily_ret.abs() <= _DAILY_RET_OUTLIER_ABS, 0.0)

    cum = (1 + daily_ret).prod() - 1
    n = len(daily_ret)
    if n <= 0 or not np.isfinite(cum):
        return 0.0
    if cum <= -1:
        return float(cum)
    base = 1 + cum
    if base <= 0:
        return 0.0
    annualized = base ** (periods_per_year / n) - 1
    if not np.isfinite(annualized):
        return 0.0
    return float(annualized)


def compute_mwr(
    trades: list[dict],
    dividends: list[dict],
    current_value: float = 0.0,
    options_trades: Optional[list[dict]] = None,
) -> float:
    """Money-weighted return via XIRR. Buys are negative, sells/divs/current
    holdings positive. Cancelled trades are filtered out so a reversed
    fat-finger doesn't generate a phantom IRR cashflow pair.

    Sprint 9C.4: if `options_trades` is passed, options cash flows are
    appended (BTO/BTC negative, STC/STO positive). OEXP and CONV are
    excluded — premium was counted at open. Returns float (annualized)
    or 0 on failure.
    """
    if xirr is None:
        return 0.0
    trades = _drop_cancelled(trades)
    cashflows: list[tuple] = []
    for t in trades:
        ex = t.get("executed_at")
        amt = t.get("amount")
        action = t.get("action")
        if not ex or amt is None:
            continue
        # Force sign convention: buy outflow negative, sell inflow positive
        if action == "buy":
            cashflows.append((datetime.strptime(ex[:10], "%Y-%m-%d").date(), -abs(float(amt))))
        elif action == "sell":
            cashflows.append((datetime.strptime(ex[:10], "%Y-%m-%d").date(), abs(float(amt))))
    for d in dividends:
        pa = d.get("paid_at")
        amt = d.get("amount")
        if not pa or amt is None:
            continue
        cashflows.append((datetime.strptime(pa[:10], "%Y-%m-%d").date(), float(amt)))
    if options_trades:
        for ex_str, amt in build_options_cash_flows(options_trades):
            try:
                cashflows.append((datetime.strptime(ex_str[:10], "%Y-%m-%d").date(), float(amt)))
            except (ValueError, TypeError):
                continue
    if current_value:
        cashflows.append((datetime.now().date(), float(current_value)))
    if len(cashflows) < 2:
        return 0.0
    try:
        result = xirr([c[0] for c in cashflows], [c[1] for c in cashflows])
        return float(result) if result is not None else 0.0
    except Exception as e:
        print(f"[perf] xirr error: {e}")
        return 0.0


def compute_alpha_beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series, rf_rate: float = 0.04, periods_per_year: int = 252):
    """CAPM regression of excess returns. Returns (annualized_alpha, beta).

    Standard CAPM:  port_excess = α_daily + β × bm_excess + ε
                    where  port_excess = port_return − rf_daily
                           bm_excess   = bm_return   − rf_daily

    Annualization is GEOMETRIC: α_annualized = (1 + α_daily)^252 − 1.
    This matches how compute_twr annualizes — both compound consistently.
    Arithmetic α × 252 was previously used; close to geometric for small α
    but mathematically inconsistent.

    Outlier handling: align portfolio and benchmark FIRST, then drop rows
    where either side exceeds the outlier threshold. This is the fix for
    the production sample-selection bias bug:
      - Old: zap independently (port → 0 on trade days, bm stays real).
        Inner-join kept those rows. Regression saw "port flat while bm
        moved", pulling β toward 0 and inflating α to absorb the slack.
      - New: drop outlier rows ENTIRELY from both sides. Regression only
        sees days where both sides have clean data.

    Returns (None, None) when:
        - either input is None / empty
        - aligned post-drop overlap < 30 points
        - benchmark variance is zero
        - regression produces NaN/Inf
        - result is statistically implausible (|β| > 5 or |α| > 100%/yr)
    """
    # Diagnostic prints (kept temporarily; will strip after prod confirms clean)
    if portfolio_returns is not None and not portfolio_returns.empty:
        p_outliers = (portfolio_returns.abs() > _DAILY_RET_OUTLIER_ABS).sum()
        p_nan = portfolio_returns.isna().sum()
        print(f"[perf] alpha_beta input port: len={len(portfolio_returns)} "
              f"min={portfolio_returns.min():.4f} max={portfolio_returns.max():.4f} "
              f"outliers>{_DAILY_RET_OUTLIER_ABS}={p_outliers} nan={p_nan}")
    if benchmark_returns is not None and not benchmark_returns.empty:
        b_outliers = (benchmark_returns.abs() > _DAILY_RET_OUTLIER_ABS).sum()
        b_nan = benchmark_returns.isna().sum()
        print(f"[perf] alpha_beta input bm:   len={len(benchmark_returns)} "
              f"min={benchmark_returns.min():.4f} max={benchmark_returns.max():.4f} "
              f"outliers>{_DAILY_RET_OUTLIER_ABS}={b_outliers} nan={b_nan}")

    if portfolio_returns is None or benchmark_returns is None:
        return None, None

    # Strip Inf/NaN before alignment
    p = portfolio_returns.replace([np.inf, -np.inf], np.nan).dropna()
    b = benchmark_returns.replace([np.inf, -np.inf], np.nan).dropna()

    # Align first, then drop rows where EITHER series is an outlier. Avoids
    # the sample-selection bias of independent zapping.
    aligned = pd.concat([p, b], axis=1, join="inner").dropna()
    if aligned.empty:
        return None, None
    p_aligned = aligned.iloc[:, 0]
    b_aligned = aligned.iloc[:, 1]
    clean_mask = (p_aligned.abs() <= _DAILY_RET_OUTLIER_ABS) & (b_aligned.abs() <= _DAILY_RET_OUTLIER_ABS)
    n_dropped = (~clean_mask).sum()
    if n_dropped > 0:
        print(f"[perf] alpha_beta: dropped {n_dropped} aligned rows where "
              f"either series exceeded {_DAILY_RET_OUTLIER_ABS*100:.0f}% threshold")
    p_aligned = p_aligned[clean_mask]
    b_aligned = b_aligned[clean_mask]

    if len(p_aligned) < 30:
        print(f"[perf] alpha_beta: clean aligned len {len(p_aligned)} < 30, returning None")
        return None, None

    rf_daily = rf_rate / periods_per_year
    excess_p = p_aligned - rf_daily
    excess_b = b_aligned - rf_daily
    if excess_b.std() == 0:
        return None, None
    try:
        beta, alpha_daily = np.polyfit(excess_b.values, excess_p.values, 1)
    except Exception as e:
        print(f"[perf] alpha_beta polyfit failed: {e}")
        return None, None
    if not np.isfinite(alpha_daily) or not np.isfinite(beta):
        return None, None

    # GEOMETRIC annualization — consistent with how compute_twr annualizes.
    # Guards: alpha_daily must keep (1 + α_d) > 0 for the power to be real.
    base = 1.0 + alpha_daily
    if base <= 0:
        return None, None
    annual_alpha = base ** periods_per_year - 1
    annual_alpha_arith = alpha_daily * periods_per_year

    # Full-visibility diagnostic (kept temporarily for production debugging).
    # Compare CAPM expectation vs OLS recovery so we can see directly whether
    # the regression's intercept is in the right space.
    p_mean_d = p_aligned.mean()
    b_mean_d = b_aligned.mean()
    p_mean_geo = (1 + p_mean_d) ** periods_per_year - 1 if (1 + p_mean_d) > 0 else float("nan")
    b_mean_geo = (1 + b_mean_d) ** periods_per_year - 1 if (1 + b_mean_d) > 0 else float("nan")
    print(f"[perf] alpha_beta DETAIL:")
    print(f"  formula: y = port_returns - rf_daily; X = bm_returns - rf_daily; OLS(y~X) -> intercept = alpha_daily")
    print(f"  n_observations               = {len(p_aligned)}")
    print(f"  portfolio_mean_daily_return  = {p_mean_d:.6f}")
    print(f"  portfolio_mean_annualized_geo= {p_mean_geo:.4f}")
    print(f"  benchmark_mean_daily_return  = {b_mean_d:.6f}")
    print(f"  benchmark_mean_annualized_geo= {b_mean_geo:.4f}")
    print(f"  rf_daily_used                = {rf_daily:.6f}")
    print(f"  rf_annualized                = {rf_rate:.4f}")
    print(f"  OLS intercept (raw daily)    = {alpha_daily:.6f}")
    print(f"  OLS slope (beta)             = {beta:.4f}")
    print(f"  intercept annual geometric   = {annual_alpha:.4f}")
    print(f"  intercept annual arithmetic  = {annual_alpha_arith:.4f}")
    print(f"  CAPM identity check: port_geo = rf + beta*(bm_geo - rf) + alpha_implied")
    capm_alpha_implied = p_mean_geo - (rf_rate + beta * (b_mean_geo - rf_rate)) if not (np.isnan(p_mean_geo) or np.isnan(b_mean_geo)) else float("nan")
    print(f"  CAPM-implied alpha (annual)  = {capm_alpha_implied:.4f}  <- this is what the user computes by hand")
    if not np.isfinite(annual_alpha):
        return None, None

    # Plausibility bound: real-world equity beta typically lives in [0, 3].
    # Real alpha rarely exceeds ±50%/yr for a diversified portfolio. Anything
    # beyond |β|>5 or |α|>100% is statistical garbage.
    if abs(beta) > 5.0 or abs(annual_alpha) > 1.0:
        print(f"[perf] alpha_beta result out of plausible range: "
              f"alpha={annual_alpha:.4f} beta={beta:.4f} — returning None")
        return None, None

    return float(annual_alpha), float(beta)


def compute_sharpe(returns: pd.Series, rf_rate: float = 0.04, periods_per_year: int = 252):
    """Sharpe ratio. Returns None when std is zero/NaN or there's no data
    (rather than a synthetic 0.0 that would pretend to be a real signal)."""
    if returns is None or returns.empty:
        return None
    r = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return None
    std = r.std()
    if std is None or not np.isfinite(std) or std == 0:
        return None
    rf_daily = rf_rate / periods_per_year
    excess = r - rf_daily
    val = (excess.mean() * periods_per_year) / (std * math.sqrt(periods_per_year))
    if not np.isfinite(val):
        return None
    return float(val)


def compute_sortino(returns: pd.Series, rf_rate: float = 0.04, periods_per_year: int = 252):
    """Sortino ratio. Returns None when downside std is zero/NaN or there's
    no negative returns."""
    if returns is None or returns.empty:
        return None
    r = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return None
    rf_daily = rf_rate / periods_per_year
    excess = r - rf_daily
    downside = r[r < 0]
    if downside.empty:
        return None
    dstd = downside.std()
    if dstd is None or not np.isfinite(dstd) or dstd == 0:
        return None
    val = (excess.mean() * periods_per_year) / (dstd * math.sqrt(periods_per_year))
    if not np.isfinite(val):
        return None
    return float(val)


def compute_max_drawdown(nav_series: pd.Series) -> tuple[float, str]:
    """Max peak-to-trough decline. Returns (max_dd_pct, trough_date_iso).

    Should be called with the cumulative-return-index series (clean equity
    curve) — NOT raw NAV which spikes on trade days from buys/sells.
    """
    if nav_series is None or nav_series.empty:
        return 0.0, ""
    # Diagnostic print (kept temporarily until prod numbers confirmed clean)
    print(f"[perf] max_drawdown input: len={len(nav_series)} "
          f"min={nav_series.min():.4f} max={nav_series.max():.4f} "
          f"first={nav_series.iloc[0]:.4f} last={nav_series.iloc[-1]:.4f} "
          f"nan={nav_series.isna().sum()}")
    s = nav_series.dropna()
    if s.empty:
        return 0.0, ""
    rolling_max = s.cummax()
    # Avoid div-by-zero when nav starts at zero (happens when all opening
    # tickers had price-fetch failures). inf/-inf would otherwise propagate.
    rolling_max_safe = rolling_max.where(rolling_max != 0, np.nan)
    drawdown = ((s - rolling_max_safe) / rolling_max_safe)
    drawdown = drawdown.replace([np.inf, -np.inf], np.nan).dropna()
    if drawdown.empty:
        return 0.0, ""
    trough_idx = drawdown.idxmin()
    min_val = drawdown.min()
    if not np.isfinite(min_val):
        return 0.0, ""
    # Sanity bound: max DD is in [-1, 0]. A real portfolio cannot lose more
    # than 100% of value (it goes to zero, not negative). Anything outside
    # this range is a math artifact from a NAV anomaly.
    bounded = max(-1.0, min(0.0, float(min_val)))
    return bounded, str(trough_idx.date()) if hasattr(trough_idx, "date") else str(trough_idx)


# ─────────────────────────────────────────────────────────────────────────────
# Bucket attributions
# ─────────────────────────────────────────────────────────────────────────────

_HOLDING_BUCKETS = [
    ("intraday", 0, 1),
    ("<1w", 1, 7),
    ("1-4w", 7, 28),
    ("1-3m", 28, 91),
    ("3-12m", 91, 365),
    (">1y", 365, 100_000),
]


def bucket_by_holding_period(closed: list[dict]) -> dict:
    out = {label: {"count": 0, "total_pnl": 0.0, "returns": []} for label, _, _ in _HOLDING_BUCKETS}
    for c in closed:
        days = c.get("holding_period_days") or 0
        pnl = float(c.get("pnl_dollars") or 0)
        ret = float(c.get("pnl_pct") or 0)
        for label, lo, hi in _HOLDING_BUCKETS:
            if lo <= days < hi:
                out[label]["count"] += 1
                out[label]["total_pnl"] += pnl
                out[label]["returns"].append(ret)
                break
    return {
        k: {
            "count": v["count"],
            "total_pnl": round(v["total_pnl"], 2),
            "avg_return_pct": round(sum(v["returns"]) / len(v["returns"]), 4) if v["returns"] else 0.0,
        }
        for k, v in out.items()
    }


def bucket_by_sector(closed: list[dict], sector_lookup: dict[str, str]) -> dict:
    """Bucket P&L by sector. sector_lookup: {ticker: sector_name}. Unknown → 'Unknown'."""
    out: dict[str, dict] = {}
    for c in closed:
        sector = sector_lookup.get(c.get("ticker", ""), "Unknown") or "Unknown"
        bucket = out.setdefault(sector, {"count": 0, "total_pnl": 0.0, "returns": []})
        bucket["count"] += 1
        bucket["total_pnl"] += float(c.get("pnl_dollars") or 0)
        bucket["returns"].append(float(c.get("pnl_pct") or 0))
    return {
        k: {
            "count": v["count"],
            "total_pnl": round(v["total_pnl"], 2),
            "avg_return_pct": round(sum(v["returns"]) / len(v["returns"]), 4) if v["returns"] else 0.0,
        }
        for k, v in out.items()
    }


def bucket_by_year(closed: list[dict], benchmark_annual_returns: dict[int, float] | None = None) -> dict:
    out: dict[str, dict] = {}
    for c in closed:
        ed = c.get("exit_date") or ""
        if len(ed) < 4:
            continue
        year = int(ed[:4])
        bucket = out.setdefault(str(year), {"count": 0, "total_pnl": 0.0, "returns": [], "benchmark": None})
        bucket["count"] += 1
        bucket["total_pnl"] += float(c.get("pnl_dollars") or 0)
        bucket["returns"].append(float(c.get("pnl_pct") or 0))
    bm = benchmark_annual_returns or {}
    return {
        k: {
            "count": v["count"],
            "total_pnl": round(v["total_pnl"], 2),
            "avg_return_pct": round(sum(v["returns"]) / len(v["returns"]), 4) if v["returns"] else 0.0,
            "benchmark_return_pct": bm.get(int(k)),
        }
        for k, v in out.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Headline stats
# ─────────────────────────────────────────────────────────────────────────────

def compute_headline_stats(
    trades: list[dict],
    dividends: list[dict],
    benchmark_ticker: str = "SPY",
    rf_rate: float = 0.04,
    options_trades: Optional[list[dict]] = None,
    scope: str = "combined",
) -> dict:
    """Top-level dashboard stats. All numbers round-tripped through pandas/numpy.

    Sprint 9C.4 — `options_trades` and `scope` ∈ {"combined", "equity",
    "options"}. NAV / TWR / MWR / Sharpe / max_drawdown all derive from
    the matching scope's NAV+CF series; closed_positions / win rate /
    best/worst pull from equity FIFO (match_fifo_lots), options FIFO
    (lib/options_fifo.match_options_positions), or both concatenated.

    Cancelled trades are excluded from every metric (n_trades, FIFO matching,
    NAV-driven returns, MWR, win/loss counts) via _drop_cancelled at each
    consumer's entry point.
    """
    if scope not in ("combined", "equity", "options"):
        scope = "combined"

    eq_active = scope in ("combined", "equity")
    opt_active = scope in ("combined", "options") and bool(options_trades)

    eq_trades_clean = _drop_cancelled(trades) if eq_active else []
    n_eq_trades = len([t for t in eq_trades_clean if t.get("action") in ("buy", "sell")])

    # Equity FIFO closed lots (only when equity is in-scope)
    eq_closed = match_fifo_lots(eq_trades_clean) if eq_active else []

    # Options closed positions (only when options is in-scope). lib/options_fifo
    # produces both closed + open; we take closed for win-rate / best-worst,
    # but we EXCLUDE outcome='conversion_unhandled' from P&L aggregations
    # because those carry realized_pnl=0 placeholders.
    opt_closed_raw: list[dict] = []
    n_options_trades = 0
    if opt_active:
        try:
            from lib.options_fifo import match_options_positions
            res = match_options_positions(options_trades or [])
            opt_closed_raw = res.get("closed_positions") or []
            n_options_trades = len([
                t for t in (options_trades or [])
                if t.get("cancellation_status", "normal") == "normal"
                and t.get("trans_code") != "CONV"
            ])
        except Exception as e:
            print(f"[perf] options FIFO failed: {e}")
    opt_closed_real = [c for c in opt_closed_raw if c.get("outcome") != "conversion_unhandled"]

    # Combined PnL list for win rate / best-worst aggregation. Equity uses
    # `pnl_dollars`; options closed_positions use `realized_pnl`. Normalize.
    pnls: list[float] = []
    pnls.extend(float(c.get("pnl_dollars") or 0) for c in eq_closed)
    pnls.extend(float(c.get("realized_pnl") or 0) for c in opt_closed_real)

    n_closed = len(eq_closed) + len(opt_closed_real)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    win_rate = (len(wins) / len(pnls)) if pnls else 0.0
    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss = (sum(losses) / len(losses)) if losses else 0.0
    sum_loss_abs = abs(sum(losses)) if losses else 0.0
    profit_factor = (sum(wins) / sum_loss_abs) if sum_loss_abs > 0 else (float("inf") if wins else 0.0)
    expectancy = (sum(pnls) / len(pnls)) if pnls else 0.0
    best_trade = max(pnls) if pnls else 0.0
    worst_trade = min(pnls) if pnls else 0.0

    # Total realized P&L per scope. Equity FIFO sum + options closed sum.
    total_realized_pnl = sum(pnls)

    # NAV-driven metrics. build_daily_nav handles scope routing.
    daily_nav = build_daily_nav(
        trades,
        dividends,
        options_trades=options_trades if opt_active else None,
        scope=scope,
    )
    tickers_skipped = list(daily_nav.attrs.get("tickers_skipped", []) or [])
    twr = compute_twr(daily_nav)
    current_holdings_value = (
        float(daily_nav["holdings_value"].iloc[-1])
        if not daily_nav.empty else 0.0
    )
    mwr = compute_mwr(
        eq_trades_clean if eq_active else [],
        dividends if eq_active else [],
        current_value=current_holdings_value,
        options_trades=options_trades if opt_active else None,
    )

    max_dd, max_dd_date = (0.0, "")
    sharpe = sortino = None  # None when std is zero/NaN — surfaces honestly
    alpha = None
    beta = None
    benchmark_total_return = 0.0
    benchmark_annualized_return = 0.0
    # Coverage range — non-null only when the benchmark series has data, so the
    # frontend can render "(since YYYY-MM-DD)" subtitles when SPY history is
    # truncated relative to portfolio history.
    benchmark_coverage_start: Optional[str] = None
    benchmark_coverage_end: Optional[str] = None
    if not daily_nav.empty:
        nav_series = daily_nav["nav"]
        # Single source of truth: cash-flow-adjusted returns with the >30%
        # outlier zap applied. Sharpe / Sortino / alpha / beta and the equity
        # curve for max_drawdown all derive from THIS series so they can't
        # disagree about what "the return that day" means.
        port_returns = _build_clean_daily_returns(daily_nav)
        # Equity-curve index for max_drawdown — cumulative product of
        # (1 + clean_return), so trade-day NAV jumps from buys/sells don't
        # poison the rolling max.
        clean_nav_index = _build_clean_nav_index(port_returns)
        max_dd, max_dd_date = compute_max_drawdown(clean_nav_index)
        sharpe = compute_sharpe(port_returns, rf_rate)
        sortino = compute_sortino(port_returns, rf_rate)
        # Benchmark returns
        try:
            from lib.polygon_client import get_prices_dataframe
            days = (nav_series.index[-1] - nav_series.index[0]).days + 60
            bm_df = get_prices_dataframe([benchmark_ticker], days=days)
            if not bm_df.empty and benchmark_ticker in bm_df.columns:
                bm_df.index = _to_utc_naive(bm_df.index)
                bm_series = bm_df[benchmark_ticker].reindex(nav_series.index, method="ffill")
                bm_returns_raw = bm_series.pct_change()
                # Zap benchmark outliers too — defensive, in case of a
                # corporate action / split that didn't clean adjust.
                bm_returns = _zap_outliers(bm_returns_raw, label="bm_returns")
                alpha, beta = compute_alpha_beta(port_returns, bm_returns, rf_rate)
                # Drop leading NaN values from bm_series — happens when
                # portfolio history extends before benchmark data availability
                # (Polygon Starter has ~5yr SPY history, but Patrick's portfolio
                # dates back to 2020-01-08). Without this, bm_series.iloc[0] is
                # NaN and benchmark_total_return / annualized return both come
                # out as 0.00% in the response.
                bm_series_clean = bm_series.dropna()
                if len(bm_series_clean) >= 2 and bm_series_clean.iloc[0] > 0 and pd.notna(bm_series_clean.iloc[-1]):
                    benchmark_total_return = float(bm_series_clean.iloc[-1] / bm_series_clean.iloc[0] - 1)
                    n = len(bm_series_clean)
                    if n > 0 and (1 + benchmark_total_return) > 0:
                        benchmark_annualized_return = float((1 + benchmark_total_return) ** (252 / n) - 1)
                    benchmark_coverage_start = bm_series_clean.index[0].strftime("%Y-%m-%d")
                    benchmark_coverage_end = bm_series_clean.index[-1].strftime("%Y-%m-%d")
        except Exception as e:
            print(f"[perf] benchmark fetch failed: {e}")

    # Calmar = annualized return / |max drawdown|
    calmar = (twr / abs(max_dd)) if max_dd != 0 else 0.0

    # Date range — pulled from whichever in-scope source has dated rows.
    all_dates: list[str] = []
    if eq_active:
        all_dates.extend(t.get("executed_at") for t in eq_trades_clean if t.get("executed_at"))
    if opt_active:
        all_dates.extend(
            t.get("executed_at") for t in (options_trades or [])
            if t.get("executed_at")
            and t.get("cancellation_status", "normal") == "normal"
            and t.get("trans_code") != "CONV"
        )
    date_range = {
        "start": min(all_dates) if all_dates else None,
        "end": max(all_dates) if all_dates else None,
    }

    # alpha/beta need explicit None handling because round(None, ...) raises;
    # every other numeric field gets cleaned by _sanitize_for_json below.
    response = {
        "twr": round(twr, 6),
        "mwr": round(mwr, 6),
        "twr_vs_mwr_gap": round(mwr - twr, 6),
        "alpha": round(alpha, 6) if alpha is not None else None,
        "beta": round(beta, 4) if beta is not None else None,
        "sharpe": round(sharpe, 4) if sharpe is not None else None,
        "sortino": round(sortino, 4) if sortino is not None else None,
        "calmar": round(calmar, 4),
        "max_drawdown": round(max_dd, 6),
        "max_drawdown_date": max_dd_date,
        "win_rate": round(win_rate, 4),
        "wins": len(wins),
        "losses": len(losses),
        "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else None,
        "expectancy": round(expectancy, 4),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "best_trade": round(best_trade, 4),
        "worst_trade": round(worst_trade, 4),
        "total_realized_pnl": round(total_realized_pnl, 2),
        "n_trades": n_eq_trades + n_options_trades,
        "n_equity_trades": n_eq_trades,
        "n_options_trades": n_options_trades,
        "n_closed_positions": n_closed,
        "n_equity_closed": len(eq_closed),
        "n_options_closed": len(opt_closed_real),
        "scope": scope,
        "date_range": date_range,
        "benchmark_ticker": benchmark_ticker,
        "benchmark_total_return": round(benchmark_total_return, 6),
        "benchmark_annualized_return": round(benchmark_annualized_return, 6),
        "benchmark_coverage_start": benchmark_coverage_start,
        "benchmark_coverage_end": benchmark_coverage_end,
        "tickers_skipped": tickers_skipped,
    }

    # Response-boundary barrier: NaN / +Inf / -Inf → None so FastAPI's strict
    # JSON serializer doesn't 500 with "Out of range float values are not JSON
    # compliant: nan". Any function above can produce nan via div-by-zero, lost
    # alignment, or partial Polygon data — we catch them all here.
    return _sanitize_for_json(response)
