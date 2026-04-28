"""Diagnostic-only script for Sprint 9B.1 review.

Runs three investigations against Patrick's real Supabase data and prints
findings. NO code changes, NO modifications to bias_detection or any
production module. Output only.

Issue 1: Why was 2022-03-07 not flagged as a panic sell?
Issue 2: MWR=0.00% vs Sprint 9A's reported +1436.75% — which is right?
Issue 3: Are inflows below $250 (DCA) timed differently than the >$250 lumps?

Usage:  python tests/diag_9b1_three_issues.py
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _load_env_local():
    sibling_env = REPO.parent / "monreale-os" / "apps" / "web" / ".env.local"
    if not sibling_env.exists():
        return
    for raw in sibling_env.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k in {"NEXT_PUBLIC_SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "POLYGON_API_KEY"} and k not in os.environ:
            os.environ[k] = v


def _fetch_supabase():
    url = os.environ["NEXT_PUBLIC_SUPABASE_URL"].rstrip("/")
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}

    trades = requests.get(
        f"{url}/rest/v1/trades",
        headers=headers,
        params={
            "select": "id,ticker,action,shares,price,amount,executed_at,cancellation_status,is_active,cancel_matched_at",
            "is_active": "eq.true",
            "order": "executed_at.asc",
            "limit": "5000",
        },
        timeout=30,
    ).json()
    dividends = requests.get(
        f"{url}/rest/v1/dividend_income",
        headers=headers,
        params={"select": "ticker,amount,paid_at", "order": "paid_at.asc", "limit": "5000"},
        timeout=30,
    ).json()
    for t in trades:
        if t.get("cancellation_status") is None:
            t["cancellation_status"] = "normal"
    return trades, dividends


def hr(s=""):
    print()
    print("=" * 80)
    if s:
        print(f"  {s}")
        print("=" * 80)


def run_issue_1(trades, daily_nav):
    """Why was 2022-03-07 not flagged as panic sell?"""
    hr("ISSUE 1 — 2022-03-07 panic-sell trigger trace")

    import pandas as pd
    from lib.bias_detection import (
        PANIC_SELL_NPOS_TRIGGER,
        PANIC_SELL_LOSS_TRIGGER_PCT,
        PANIC_SELL_DAYS_TRIGGER,
        PANIC_SELL_BUYBACK_THRESHOLD_PCT,
    )
    from lib.performance_math import match_fifo_lots, _drop_cancelled

    target_date = "2022-03-07"
    target_ts = pd.Timestamp(target_date)

    # ── 1. Show all trades on that day ──
    print(f"\n[1] All trades on {target_date} (raw, including cancelled):")
    day_rows = [t for t in trades if str(t.get("executed_at") or "")[:10] == target_date]
    if not day_rows:
        print(f"    (no trades found on {target_date})")
        # Maybe the actual liquidation was on a different nearby date — search ±5 days
        nearby = sorted(
            [t for t in trades if abs((pd.Timestamp(str(t.get("executed_at"))[:10]) - target_ts).days) <= 5],
            key=lambda t: t.get("executed_at"),
        )
        if nearby:
            print(f"\n    nearby trades within ±5 days:")
            for t in nearby:
                print(f"      {str(t.get('executed_at'))[:10]}  {t.get('ticker'):<6}  {t.get('action'):<5}  "
                      f"shares={t.get('shares')!s:<8}  amount=${float(t.get('amount') or 0):>10,.2f}  "
                      f"status={t.get('cancellation_status')}")
        return

    for t in sorted(day_rows, key=lambda t: t.get("ticker")):
        print(f"      {t.get('ticker'):<6}  {t.get('action'):<5}  shares={t.get('shares')!s:<8}  "
              f"price=${float(t.get('price') or 0):.2f}  amount=${float(t.get('amount') or 0):>10,.2f}  "
              f"status={t.get('cancellation_status')}")

    # ── 2. NAV at start vs end ──
    nav_idx = daily_nav.index
    prior = nav_idx[nav_idx < target_ts]
    same = nav_idx[nav_idx == target_ts]
    after = nav_idx[nav_idx > target_ts]
    if len(prior) == 0:
        print(f"\n[2] NAV at start: NO prior business day — {target_date} is at/before NAV start")
        return
    nav_start = float(daily_nav["nav"].loc[prior[-1]])
    nav_end = float(daily_nav["nav"].loc[same[0]]) if len(same) > 0 else None
    print(f"\n[2] NAV reference:")
    print(f"      prior business day:  {prior[-1].date()}   NAV=${nav_start:,.2f}")
    if nav_end is not None:
        print(f"      same day:            {same[0].date()}   NAV=${nav_end:,.2f}")

    # ── 3. Compute panic conditions for that day ──
    clean = _drop_cancelled(trades)
    day_clean_sells = [t for t in clean if str(t.get("executed_at") or "")[:10] == target_date and t.get("action") == "sell"]
    day_clean_buys  = [t for t in clean if str(t.get("executed_at") or "")[:10] == target_date and t.get("action") == "buy"]
    n_sells = len(day_clean_sells)
    n_buys = len(day_clean_buys)

    print(f"\n[3] After cancellation filter:")
    print(f"      n_sells_today: {n_sells}  (trigger ≥ {PANIC_SELL_NPOS_TRIGGER})")
    print(f"      n_buys_today:  {n_buys}")
    for t in sorted(day_clean_sells, key=lambda x: x.get("ticker")):
        print(f"        SELL  {t.get('ticker'):<6}  shares={t.get('shares')!s:<8}  "
              f"amount=${float(t.get('amount') or 0):>10,.2f}")
    for t in sorted(day_clean_buys, key=lambda x: x.get("ticker")):
        print(f"        BUY   {t.get('ticker'):<6}  shares={t.get('shares')!s:<8}  "
              f"amount=${float(t.get('amount') or 0):>10,.2f}")

    # Realized loss via FIFO over full history
    closed_all = match_fifo_lots(trades)
    day_closed = [c for c in closed_all if c.get("exit_date") == target_date]
    realized = sum(float(c.get("pnl_dollars") or 0) for c in day_closed)
    print(f"\n[4] FIFO-matched closed positions with exit_date={target_date}:  {len(day_closed)}")
    if day_closed:
        for c in sorted(day_closed, key=lambda x: x.get("pnl_dollars") or 0):
            print(f"        {c.get('ticker'):<6}  shares={c.get('shares'):<8}  "
                  f"cost=${float(c.get('cost_basis') or 0):>9,.2f}  "
                  f"proceeds=${float(c.get('proceeds') or 0):>9,.2f}  "
                  f"pnl=${float(c.get('pnl_dollars') or 0):>+10,.2f}  ")
    print(f"      total_realized_pnl: ${realized:+,.2f}")

    if nav_start > 0:
        loss_pct_of_nav = realized / nav_start
        print(f"      loss_pct_of_nav:   {loss_pct_of_nav:+.4%}  (trigger: |loss| > {PANIC_SELL_LOSS_TRIGGER_PCT:.0%})")

    # Liquidated value
    liquidated = sum(abs(float(t.get("amount") or 0)) for t in day_clean_sells)
    print(f"      liquidated_value:  ${liquidated:,.2f}")

    # Days idle window
    if len(after) >= PANIC_SELL_DAYS_TRIGGER:
        idle_end = after[PANIC_SELL_DAYS_TRIGGER - 1]
        # buyback within 30 trading days
        buys_in_window = [
            t for t in clean
            if t.get("action") == "buy"
            and target_ts < pd.Timestamp(str(t.get("executed_at"))[:10]) <= idle_end
        ]
        subsequent_buy_value = sum(abs(float(t.get("amount") or 0)) for t in buys_in_window)
        threshold_value = PANIC_SELL_BUYBACK_THRESHOLD_PCT * liquidated
        print(f"      idle window end (panic +{PANIC_SELL_DAYS_TRIGGER} trading days): {idle_end.date()}")
        print(f"      n_buys in window:    {len(buys_in_window)}")
        print(f"      subsequent_buy_$:    ${subsequent_buy_value:,.2f}")
        print(f"      buyback_threshold:   ${threshold_value:,.2f}  ({PANIC_SELL_BUYBACK_THRESHOLD_PCT:.0%} of liquidated)")
        if buys_in_window:
            print(f"      buys in window:")
            for t in sorted(buys_in_window, key=lambda x: x.get("executed_at")):
                print(f"        {str(t.get('executed_at'))[:10]}  {t.get('ticker'):<6}  "
                      f"amount=${float(t.get('amount') or 0):>10,.2f}")
    else:
        print(f"      forward NAV window: only {len(after)} trading days available, need {PANIC_SELL_DAYS_TRIGGER}")

    # ── 5. Verdict ──
    print(f"\n[5] Trigger-by-trigger verdict for {target_date}:")
    cond_npos = n_sells >= PANIC_SELL_NPOS_TRIGGER
    cond_loss = nav_start > 0 and abs(realized / nav_start) > PANIC_SELL_LOSS_TRIGGER_PCT and realized < 0
    cond_idle = None  # computed inline above; recompute
    if len(after) >= PANIC_SELL_DAYS_TRIGGER and liquidated > 0:
        idle_end = after[PANIC_SELL_DAYS_TRIGGER - 1]
        buys_in_window = [
            t for t in clean
            if t.get("action") == "buy"
            and target_ts < pd.Timestamp(str(t.get("executed_at"))[:10]) <= idle_end
        ]
        sbv = sum(abs(float(t.get("amount") or 0)) for t in buys_in_window)
        cond_idle = sbv < PANIC_SELL_BUYBACK_THRESHOLD_PCT * liquidated
    print(f"      [{'PASS' if cond_npos else 'FAIL'}]  n_positions ≥ {PANIC_SELL_NPOS_TRIGGER}:        {n_sells}")
    print(f"      [{'PASS' if cond_loss else 'FAIL'}]  |loss/NAV| > {PANIC_SELL_LOSS_TRIGGER_PCT:.0%} and loss<0:  realized=${realized:+,.2f}, NAV0=${nav_start:,.2f}")
    if cond_idle is not None:
        print(f"      [{'PASS' if cond_idle else 'FAIL'}]  buys_in_30TD < {PANIC_SELL_BUYBACK_THRESHOLD_PCT:.0%}·liquidated")
    else:
        print(f"      [SKIP]  idle-window check skipped (insufficient forward data)")


def run_issue_2(trades, dividends, daily_nav):
    """Compare current MWR result to whatever Sprint 9A reported."""
    hr("ISSUE 2 — MWR diagnostic (current 9B.1 path vs raw cashflows)")

    from lib.performance_math import compute_mwr, _drop_cancelled

    # Run compute_mwr the same way the smoke test / router does
    holdings_value = float(daily_nav["holdings_value"].iloc[-1]) if not daily_nav.empty else 0.0
    print(f"[1] current_value passed to compute_mwr: ${holdings_value:,.2f}")
    mwr_smoke = compute_mwr(trades, dividends, current_value=holdings_value)
    print(f"[2] compute_mwr(trades, dividends, current_value=hv) = {mwr_smoke!r}  ({mwr_smoke*100:.2f}%)")

    # Also try with current_value=0 to see if XIRR converges differently
    mwr_zero = compute_mwr(trades, dividends, current_value=0.0)
    print(f"[3] compute_mwr(trades, dividends, current_value=0)   = {mwr_zero!r}  ({mwr_zero*100:.2f}%)")

    # Build the cashflow list compute_mwr would have used and inspect
    clean = _drop_cancelled(trades)
    cashflows = []
    for t in clean:
        ex = t.get("executed_at")
        amt = t.get("amount")
        action = t.get("action")
        if not ex or amt is None:
            continue
        d = ex[:10]
        if action == "buy":
            cashflows.append((d, -abs(float(amt))))
        elif action == "sell":
            cashflows.append((d, abs(float(amt))))
    for div in dividends:
        pa = div.get("paid_at")
        amt = div.get("amount")
        if not pa or amt is None:
            continue
        cashflows.append((pa[:10], float(amt)))

    print(f"\n[4] Cashflow list compute_mwr saw:")
    print(f"      total cashflows: {len(cashflows)}")
    print(f"      buys (negative): {sum(1 for _, v in cashflows if v < 0)}")
    print(f"      sells/divs (positive): {sum(1 for _, v in cashflows if v > 0)}")
    print(f"      sum of all cashflows (excl. terminal): ${sum(v for _, v in cashflows):+,.2f}")
    print(f"      most-negative deposit: ${min(v for _, v in cashflows):,.2f}")
    print(f"      most-positive inflow:  ${max(v for _, v in cashflows):,.2f}")

    # Try pyxirr directly to see what it reports
    try:
        from pyxirr import xirr
        from datetime import datetime as _dt
        dates = [_dt.strptime(d, "%Y-%m-%d").date() for d, _ in cashflows]
        amounts = [v for _, v in cashflows]
        if holdings_value > 0:
            dates.append(_dt.now().date())
            amounts.append(holdings_value)
        result = xirr(dates, amounts)
        print(f"\n[5] Direct pyxirr.xirr(dates, amounts) with terminal value:")
        print(f"      result: {result!r}")
        if result is not None:
            print(f"      = {float(result)*100:.2f}% annualized")
    except Exception as e:
        print(f"\n[5] Direct pyxirr error: {type(e).__name__}: {e}")

    # Hypothesis: Sprint 9A's reported +1436% may have used a DIFFERENT current_value
    # source. Try a much larger terminal value (e.g., what Patrick's portfolio is
    # actually worth if all open positions were valued at current Polygon close).
    print(f"\n[6] Hypothesis test: try larger terminal values")
    for tv in [1000, 10000, 50000, 100000]:
        m = compute_mwr(trades, dividends, current_value=float(tv))
        print(f"      current_value=${tv:>9,}  -> mwr={m!r} ({m*100:.2f}%)")


def run_issue_3(trades, dividends, daily_nav, spy_daily):
    """Bimodal cash flow timing: split inflows above/below $250 threshold."""
    hr("ISSUE 3 — Bimodal cash-flow timing (DCA <$250 vs lumps ≥$250)")

    import pandas as pd
    from lib.bias_detection import (
        CFT_SPY_LOOKBACK_DAYS,
        CFT_SPY_MIN_HISTORY_DAYS,
        CFT_INFLOW_DOLLAR_THRESHOLD,
    )

    if daily_nav is None or daily_nav.empty or "cash_flow" not in daily_nav.columns:
        print("    no daily_nav.cash_flow available")
        return

    cf = daily_nav["cash_flow"].astype(float)
    inflow_all = cf[cf < 0]
    print(f"\n[1] All inflow days (cash_flow < 0): {len(inflow_all)}")
    print(f"      total absolute inflow $:      ${inflow_all.abs().sum():,.2f}")

    inflow_below = inflow_all[inflow_all.abs() < CFT_INFLOW_DOLLAR_THRESHOLD]
    inflow_above = inflow_all[inflow_all.abs() >= CFT_INFLOW_DOLLAR_THRESHOLD]
    print(f"\n[2] Split at ${CFT_INFLOW_DOLLAR_THRESHOLD:.0f} threshold:")
    print(f"      < $250:    {len(inflow_below):>4}  total ${inflow_below.abs().sum():>10,.2f}")
    print(f"      ≥ $250:    {len(inflow_above):>4}  total ${inflow_above.abs().sum():>10,.2f}")

    def _spy_pos(date_ts):
        if spy_daily is None or spy_daily.empty:
            return None
        window_start = date_ts - pd.Timedelta(days=CFT_SPY_LOOKBACK_DAYS * 2)
        window = spy_daily.loc[(spy_daily.index >= window_start) & (spy_daily.index <= date_ts)]
        if len(window) > CFT_SPY_LOOKBACK_DAYS:
            window = window.iloc[-CFT_SPY_LOOKBACK_DAYS:]
        if len(window) < CFT_SPY_MIN_HISTORY_DAYS:
            return None
        lo, hi = float(window.min()), float(window.max())
        if hi <= lo:
            return None
        return (float(window.iloc[-1]) - lo) / (hi - lo)

    def _bucket_stats(name, series):
        wn = wd = 0.0
        n_trough = n_peak = n_mid = n_missing = 0
        per_year = defaultdict(lambda: [0, 0.0])  # year -> [count, sum_abs]
        for d, amt in series.items():
            per_year[d.year][0] += 1
            per_year[d.year][1] += abs(float(amt))
            pos = _spy_pos(d)
            if pos is None:
                n_missing += 1
                continue
            w = abs(float(amt))
            wn += w * pos
            wd += w
            if pos < 0.33:
                n_trough += 1
            elif pos > 0.67:
                n_peak += 1
            else:
                n_mid += 1
        avg = (wn / wd) if wd > 0 else None
        print(f"\n[{name}] weighted_avg_pos = {avg if avg is None else f'{avg:.3f}'}")
        print(f"      n_total: {len(series)}")
        print(f"      categorized: trough={n_trough}  middle={n_mid}  peak={n_peak}  missing_history={n_missing}")
        print(f"      per-year breakdown:")
        for y in sorted(per_year.keys()):
            c, s = per_year[y]
            print(f"        {y}:  n={c:>3}   $sum={s:>11,.2f}")
        return avg

    avg_below = _bucket_stats("3", inflow_below)
    avg_above = _bucket_stats("4", inflow_above)

    print()
    print(f"[5] Hypothesis test:")
    print(f"      sub-$250 (DCA) avg SPY position:   {avg_below if avg_below is None else f'{avg_below:.3f}'}")
    print(f"      ≥$250 (lumps) avg SPY position:    {avg_above if avg_above is None else f'{avg_above:.3f}'}")
    if avg_below is not None and avg_above is not None:
        if avg_below < 0.50 and avg_above > 0.50:
            print(f"      VERDICT: hypothesis SUPPORTED — DCA at lower SPY range, lumps at higher.")
        elif avg_below > 0.50 and avg_above < 0.50:
            print(f"      VERDICT: REVERSE pattern — DCA at higher SPY range, lumps at lower (unusual).")
        else:
            print(f"      VERDICT: same direction — not bimodal at the $250 boundary.")


def main():
    _load_env_local()
    print("[diag] fetching trades + dividends from Supabase...")
    trades, dividends = _fetch_supabase()
    print(f"[diag] fetched {len(trades)} trades, {len(dividends)} dividends")

    from lib.performance_math import build_daily_nav

    print("[diag] building daily_nav (this hits Polygon)...")
    daily_nav = build_daily_nav(trades, dividends)
    print(f"[diag] daily_nav rows: {len(daily_nav)}")

    print("[diag] fetching SPY 8y direct from Polygon...")
    import pandas as pd
    api_key = os.environ.get("POLYGON_API_KEY", "")
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=2920)).strftime("%Y-%m-%d")
    spy_resp = requests.get(
        f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/{from_date}/{to_date}",
        params={"adjusted": "true", "sort": "asc", "limit": 5000, "apiKey": api_key},
        timeout=30,
    )
    spy_results = spy_resp.json().get("results") or []
    spy_daily = pd.Series(
        {datetime.fromtimestamp(r["t"] / 1000).strftime("%Y-%m-%d"): r["c"] for r in spy_results}
    )
    spy_daily.index = pd.to_datetime(spy_daily.index)
    spy_daily = spy_daily.sort_index()
    print(f"[diag] SPY rows: {len(spy_daily)}")

    run_issue_1(trades, daily_nav)
    run_issue_2(trades, dividends, daily_nav)
    run_issue_3(trades, dividends, daily_nav, spy_daily)


if __name__ == "__main__":
    main()
