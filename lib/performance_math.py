"""Pure performance-attribution math. No FastAPI deps — only pandas/numpy/pyxirr.

Inputs are plain dicts (parser output shape). Outputs are dicts/lists ready for
JSON serialization. All arithmetic happens here; the frontend just renders.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Iterable

import math
import numpy as np
import pandas as pd

# pyxirr is available on Railway via requirements.txt
try:
    from pyxirr import xirr
except ImportError:
    xirr = None  # tests can monkey-patch if needed


# ─────────────────────────────────────────────────────────────────────────────
# FIFO matching
# ─────────────────────────────────────────────────────────────────────────────

def match_fifo_lots(trades: list[dict]) -> list[dict]:
    """FIFO match buys against sells per ticker.

    Each closed position represents ONE matched chunk: the ENTIRE sell is split
    into chunks, each chunk consuming from the oldest open buy lot. A 10-share
    sell against two 6-share lots produces TWO closed positions.

    Returns list of:
        ticker, entry_date, exit_date, shares, cost_basis, proceeds,
        pnl_dollars, pnl_pct, holding_period_days, is_long_term,
        entry_trade_id, exit_trade_id  (when ids are present in input).
    """
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

def build_daily_nav(trades: list[dict], dividends: list[dict]) -> pd.DataFrame:
    """Daily portfolio NAV from trades + dividends, valuing open positions
    with Polygon close prices and adding cumulative dividends.

    Returns a DataFrame indexed by date with columns: nav, cash_flow, holdings_value, dividends_cum.
    Empty DataFrame if no priceable data.
    """
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    df = df.dropna(subset=["executed_at", "ticker"])
    df["date"] = pd.to_datetime(df["executed_at"]).dt.normalize()
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

    # Cumulative position per ticker on each business day
    positions = pd.DataFrame(0.0, index=idx, columns=tickers)
    for ticker in tickers:
        td = df[df["ticker"] == ticker].sort_values("date")
        cum = 0.0
        for _, r in td.iterrows():
            cum += float(r["shares_signed"] or 0)
            mask = positions.index >= r["date"]
            positions.loc[mask, ticker] = cum

    # Fetch prices for all tickers at once
    prices_df = _fetch_prices(tickers, start, end)
    if prices_df.empty:
        return pd.DataFrame()
    prices_df = prices_df.reindex(idx).ffill().bfill()

    # Holdings value
    holdings_value = (positions * prices_df.reindex(columns=tickers)).sum(axis=1).fillna(0)

    # Cumulative dividends as cash drag
    div_cum = pd.Series(0.0, index=idx)
    if dividends:
        ddf = pd.DataFrame(dividends).dropna(subset=["paid_at"])
        if not ddf.empty:
            ddf["date"] = pd.to_datetime(ddf["paid_at"]).dt.normalize()
            ddf["amount"] = ddf["amount"].astype(float)
            daily_div = ddf.groupby("date")["amount"].sum()
            div_cum = daily_div.reindex(idx, fill_value=0).cumsum()

    # Cash flow on a given day = signed amount of trades that day
    daily_cf = df.groupby("date")["amount"].sum().reindex(idx, fill_value=0)

    nav = holdings_value + div_cum

    return pd.DataFrame({
        "nav": nav,
        "cash_flow": daily_cf,
        "holdings_value": holdings_value,
        "dividends_cum": div_cum,
    })


def _fetch_prices(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch daily closes for a list of tickers between start and end.

    Returns a DataFrame indexed by date with one column per ticker.
    Uses Polygon via lib.polygon_client.get_prices_dataframe.
    """
    try:
        from lib.polygon_client import get_prices_dataframe
        days = max((end - start).days + 60, 30)
        return get_prices_dataframe(tickers, days=days)
    except Exception as e:
        print(f"[perf] price fetch failed: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Returns
# ─────────────────────────────────────────────────────────────────────────────

def compute_twr(daily_nav: pd.DataFrame, periods_per_year: int = 252) -> float:
    """Annualized time-weighted return.

    Chained daily returns of holdings_value (excludes cash flows from the
    return series). Returns float (e.g., 0.124 for 12.4% annualized).
    """
    if daily_nav is None or daily_nav.empty or "holdings_value" not in daily_nav:
        return 0.0
    hv = daily_nav["holdings_value"].astype(float)
    cf = daily_nav.get("cash_flow", pd.Series(0.0, index=hv.index)).astype(float)
    # Daily return: (V_t - cf_t) / V_{t-1} - 1, ignoring days where V_{t-1} = 0
    prev = hv.shift(1).replace(0, np.nan)
    daily_ret = ((hv - cf) / prev) - 1
    daily_ret = daily_ret.dropna()
    if daily_ret.empty:
        return 0.0
    cum = (1 + daily_ret).prod() - 1
    n = len(daily_ret)
    if n <= 0 or cum <= -1:
        return float(cum)
    annualized = (1 + cum) ** (periods_per_year / n) - 1
    return float(annualized)


def compute_mwr(trades: list[dict], dividends: list[dict], current_value: float = 0.0) -> float:
    """Money-weighted return via XIRR. Buys are negative, sells/divs/current
    holdings positive. Returns float (annualized) or 0 on failure.
    """
    if xirr is None:
        return 0.0
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


def compute_alpha_beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series, rf_rate: float = 0.04, periods_per_year: int = 252) -> tuple[float, float]:
    """OLS regression of excess returns. Returns (annualized_alpha, beta)."""
    if portfolio_returns is None or benchmark_returns is None:
        return 0.0, 1.0
    p = portfolio_returns.dropna()
    b = benchmark_returns.dropna()
    aligned = pd.concat([p, b], axis=1, join="inner").dropna()
    if len(aligned) < 30:
        return 0.0, 1.0
    rf_daily = rf_rate / periods_per_year
    excess_p = aligned.iloc[:, 0] - rf_daily
    excess_b = aligned.iloc[:, 1] - rf_daily
    try:
        beta, alpha = np.polyfit(excess_b.values, excess_p.values, 1)
        annual_alpha = alpha * periods_per_year
        return float(annual_alpha), float(beta)
    except Exception:
        return 0.0, 1.0


def compute_sharpe(returns: pd.Series, rf_rate: float = 0.04, periods_per_year: int = 252) -> float:
    if returns is None or returns.empty:
        return 0.0
    r = returns.dropna()
    if r.empty or r.std() == 0:
        return 0.0
    rf_daily = rf_rate / periods_per_year
    excess = r - rf_daily
    return float((excess.mean() * periods_per_year) / (r.std() * math.sqrt(periods_per_year)))


def compute_sortino(returns: pd.Series, rf_rate: float = 0.04, periods_per_year: int = 252) -> float:
    if returns is None or returns.empty:
        return 0.0
    r = returns.dropna()
    rf_daily = rf_rate / periods_per_year
    excess = r - rf_daily
    downside = r[r < 0]
    if downside.empty or downside.std() == 0:
        return 0.0
    return float((excess.mean() * periods_per_year) / (downside.std() * math.sqrt(periods_per_year)))


def compute_max_drawdown(nav_series: pd.Series) -> tuple[float, str]:
    """Max peak-to-trough decline. Returns (max_dd_pct, trough_date_iso)."""
    if nav_series is None or nav_series.empty:
        return 0.0, ""
    s = nav_series.dropna()
    if s.empty:
        return 0.0, ""
    rolling_max = s.cummax()
    drawdown = (s - rolling_max) / rolling_max
    if drawdown.empty:
        return 0.0, ""
    trough_idx = drawdown.idxmin()
    return float(drawdown.min()), str(trough_idx.date()) if hasattr(trough_idx, "date") else str(trough_idx)


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
) -> dict:
    """Top-level dashboard stats. All numbers round-tripped through pandas/numpy."""
    n_trades = len([t for t in trades if t.get("action") in ("buy", "sell")])
    closed = match_fifo_lots(trades)
    n_closed = len(closed)

    # Win rate, avg win/loss, profit factor, expectancy
    pnls = [c["pnl_dollars"] for c in closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    win_rate = (len(wins) / len(pnls)) if pnls else 0.0
    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss = (sum(losses) / len(losses)) if losses else 0.0
    sum_loss_abs = abs(sum(losses)) if losses else 0.0
    profit_factor = (sum(wins) / sum_loss_abs) if sum_loss_abs > 0 else (float("inf") if wins else 0.0)
    expectancy = (sum(pnls) / len(pnls)) if pnls else 0.0

    # NAV-driven metrics
    daily_nav = build_daily_nav(trades, dividends)
    twr = compute_twr(daily_nav)
    mwr = compute_mwr(trades, dividends, current_value=float(daily_nav["holdings_value"].iloc[-1]) if not daily_nav.empty else 0.0)

    max_dd, max_dd_date = (0.0, "")
    sharpe = sortino = 0.0
    alpha = 0.0
    beta = 1.0
    benchmark_total_return = 0.0
    benchmark_annualized_return = 0.0
    if not daily_nav.empty:
        nav_series = daily_nav["nav"]
        max_dd, max_dd_date = compute_max_drawdown(nav_series)
        # Daily portfolio returns
        port_returns = nav_series.pct_change().dropna()
        sharpe = compute_sharpe(port_returns, rf_rate)
        sortino = compute_sortino(port_returns, rf_rate)
        # Benchmark returns
        try:
            from lib.polygon_client import get_prices_dataframe
            days = (nav_series.index[-1] - nav_series.index[0]).days + 60
            bm_df = get_prices_dataframe([benchmark_ticker], days=days)
            if not bm_df.empty and benchmark_ticker in bm_df.columns:
                bm_series = bm_df[benchmark_ticker].reindex(nav_series.index, method="ffill")
                bm_returns = bm_series.pct_change().dropna()
                alpha, beta = compute_alpha_beta(port_returns, bm_returns, rf_rate)
                if len(bm_series) > 0 and bm_series.iloc[0] > 0:
                    benchmark_total_return = float(bm_series.iloc[-1] / bm_series.iloc[0] - 1)
                    n = len(bm_returns)
                    if n > 0:
                        benchmark_annualized_return = float((1 + benchmark_total_return) ** (252 / n) - 1)
        except Exception as e:
            print(f"[perf] benchmark fetch failed: {e}")

    # Calmar = annualized return / |max drawdown|
    calmar = (twr / abs(max_dd)) if max_dd != 0 else 0.0

    # Date range
    all_dates = [t.get("executed_at") for t in trades if t.get("executed_at")]
    date_range = {
        "start": min(all_dates) if all_dates else None,
        "end": max(all_dates) if all_dates else None,
    }

    return {
        "twr": round(twr, 6),
        "mwr": round(mwr, 6),
        "twr_vs_mwr_gap": round(mwr - twr, 6),
        "alpha": round(alpha, 6),
        "beta": round(beta, 4),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
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
        "n_trades": n_trades,
        "n_closed_positions": n_closed,
        "date_range": date_range,
        "benchmark_ticker": benchmark_ticker,
        "benchmark_total_return": round(benchmark_total_return, 6),
        "benchmark_annualized_return": round(benchmark_annualized_return, 6),
    }
