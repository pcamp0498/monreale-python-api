"""Behavioral bias analysis endpoints (Sprint 9B.1).

POST /bias/analyze accepts trades + dividends and runs:
  - panic-sell detection
  - hold-too-long / disposition effect
  - cash-flow timing attribution

Plus structured "insufficient_data" placeholders for the 5 deferred patterns
(sector drift, sizing creep, frequency, sector cycling, time-of-day bias).

All response numerics flow through _sanitize_for_json so any NaN/Inf becomes
null at the response boundary.
"""
from fastapi import APIRouter, Depends, HTTPException

from lib.auth import verify_api_key
from lib.bias_detection import (
    detect_panic_sells,
    detect_disposition_effect,
    attribute_cash_flow_timing,
    detect_sector_concentration_drift,
    detect_position_sizing_creep,
    detect_frequency_increase,
    detect_sector_cycling,
    detect_time_of_day_bias,
)
from lib.performance_math import (
    build_daily_nav,
    compute_twr,
    compute_mwr,
    _sanitize_for_json,
    _drop_cancelled,
)

router = APIRouter()


@router.post("/analyze", dependencies=[Depends(verify_api_key)])
async def analyze_biases(body: dict):
    """Run the full bias-detection pipeline on a trades + dividends payload.

    Body: {
      trades:    [{ ticker, action, shares, price, amount, executed_at,
                    cancellation_status, ... }, ...],
      dividends: [{ ticker, amount, paid_at, ... }, ...]
    }

    Returns a dict with: panic_sells, disposition, cash_flow_timing,
    deferred (5 stub findings), and a top-level meta with input counts.
    """
    try:
        trades = body.get("trades") or []
        dividends = body.get("dividends") or []
        if not isinstance(trades, list) or not isinstance(dividends, list):
            raise HTTPException(status_code=400, detail="trades and dividends must be lists")

        # Build the daily NAV. This already drops cancelled trades and
        # filters skipped (non-priced) tickers under the hood.
        daily_nav = build_daily_nav(trades, dividends)
        twr = compute_twr(daily_nav)
        mwr = compute_mwr(trades, dividends, current_value=float(daily_nav["holdings_value"].iloc[-1]) if not daily_nav.empty else 0.0)

        # SPY history covering the full trade range plus 12 months forward
        # for market-context lookups. Use a wide net so the panic-sell detector
        # can compute 6m/12m forward returns even for events near the start
        # of the window.
        spy_daily = None
        try:
            from lib.polygon_client import get_prices_dataframe
            spy_df = get_prices_dataframe(["SPY"], days=2920)  # ~8 years
            if spy_df is not None and not spy_df.empty:
                spy_daily = spy_df["SPY"] if "SPY" in spy_df.columns else spy_df.iloc[:, 0]
        except Exception as e:
            print(f"[bias] SPY fetch failed: {e}")

        # Current prices for open positions (disposition effect needs them)
        current_prices: dict[str, float] = {}
        try:
            from lib.polygon_client import get_prices_dataframe
            clean = _drop_cancelled(trades)
            open_tickers = sorted({t.get("ticker") for t in clean if t.get("ticker") and t.get("action") == "buy"})
            # Fetch a thin slice of prices and take the last value per ticker
            if open_tickers:
                price_df = get_prices_dataframe(open_tickers, days=10)
                if price_df is not None and not price_df.empty:
                    last_row = price_df.iloc[-1]
                    for tk in open_tickers:
                        if tk in price_df.columns:
                            v = last_row.get(tk)
                            if v is not None:
                                try:
                                    fv = float(v)
                                    if fv == fv and fv > 0:  # not NaN, positive
                                        current_prices[tk] = fv
                                except (TypeError, ValueError):
                                    pass
        except Exception as e:
            print(f"[bias] current-price fetch failed: {e}")

        # ── Run the three active detectors ──
        panic_events = detect_panic_sells(trades, daily_nav, spy_daily)
        disposition = detect_disposition_effect(trades, current_prices=current_prices)
        cf_timing = attribute_cash_flow_timing(daily_nav, spy_daily, twr, mwr)

        response = {
            "panic_sells": panic_events,
            "disposition": disposition,
            "cash_flow_timing": cf_timing,
            "deferred": {
                "sector_concentration_drift": detect_sector_concentration_drift(),
                "position_sizing_creep":     detect_position_sizing_creep(),
                "frequency_increase":         detect_frequency_increase(),
                "sector_cycling":             detect_sector_cycling(),
                "time_of_day_bias":           detect_time_of_day_bias(),
            },
            "meta": {
                "n_trades_input": len(trades),
                "n_trades_after_cancellation_filter": len(_drop_cancelled(trades)),
                "n_dividends_input": len(dividends),
                "twr_annualized": twr,
                "mwr_annualized": mwr,
                "nav_index_size": int(len(daily_nav.index)) if not daily_nav.empty else 0,
                "spy_history_available": spy_daily is not None and not (spy_daily.empty if hasattr(spy_daily, "empty") else False),
                "current_prices_fetched": len(current_prices),
            },
        }

        return _sanitize_for_json(response)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"bias analysis failed: {e}")
