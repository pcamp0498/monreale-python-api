"""Diagnostic — verify the 2024-03-05 panic event is real.

Pulls all trades on 2024-03-05 from Supabase, computes per-position FIFO P&L,
NAV start/end, and lists subsequent buybacks within the 30-trading-day window.
"""
from __future__ import annotations

import os
import sys
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


def main():
    _load_env_local()

    url = os.environ["NEXT_PUBLIC_SUPABASE_URL"].rstrip("/")
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}

    print("[diag] fetching trades + dividends from Supabase...")
    trades = requests.get(
        f"{url}/rest/v1/trades",
        headers=headers,
        params={
            "select": "id,ticker,action,shares,price,amount,executed_at,cancellation_status,is_active",
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
    print(f"[diag] fetched {len(trades)} trades")

    target_date = "2024-03-05"
    import pandas as pd
    target_ts = pd.Timestamp(target_date)

    from lib.performance_math import build_daily_nav, match_fifo_lots, _drop_cancelled

    print("[diag] building daily_nav (hits Polygon)...")
    daily_nav = build_daily_nav(trades, dividends)

    # ── Trades on the target day ──
    print()
    print("=" * 80)
    print(f"  RAW TRADES ON {target_date}")
    print("=" * 80)

    day_trades = [t for t in trades if str(t.get("executed_at") or "")[:10] == target_date]
    if not day_trades:
        # search ±5 days for context
        nearby = sorted(
            [t for t in trades if abs((pd.Timestamp(str(t.get("executed_at"))[:10]) - target_ts).days) <= 5],
            key=lambda t: t.get("executed_at"),
        )
        print(f"\n  (no trades on exact date; ±5-day context:)")
        for t in nearby:
            print(f"   {str(t.get('executed_at'))[:10]}  {t.get('ticker'):<6}  {t.get('action'):<5}  "
                  f"shares={t.get('shares')!s:<10}  price=${float(t.get('price') or 0):.2f}  "
                  f"amount=${float(t.get('amount') or 0):>10,.2f}  status={t.get('cancellation_status')}")
        return

    print(f"\n  count: {len(day_trades)}")
    for t in sorted(day_trades, key=lambda x: x.get("ticker")):
        print(f"   {t.get('ticker'):<6}  {t.get('action'):<5}  shares={t.get('shares')!s:<10}  "
              f"price=${float(t.get('price') or 0):.2f}  amount=${float(t.get('amount') or 0):>10,.2f}  "
              f"status={t.get('cancellation_status')}")

    # ── FIFO P&L for each closed position with exit_date == target_date ──
    print()
    print("=" * 80)
    print(f"  FIFO-MATCHED CLOSED POSITIONS WITH exit_date={target_date}")
    print("=" * 80)
    closed = match_fifo_lots(trades)
    day_closed = [c for c in closed if c.get("exit_date") == target_date]
    print()
    print(f"  count: {len(day_closed)}")
    print(f"   {'ticker':<7}{'shares':>10}{'entry_date':>14}{'cost':>12}{'proceeds':>12}{'pnl':>12}{'pnl_%':>10}{'hold_d':>9}")
    realized_total = 0.0
    cost_total = 0.0
    for c in sorted(day_closed, key=lambda x: x.get("pnl_dollars") or 0):
        cost = float(c.get("cost_basis") or 0)
        proceeds = float(c.get("proceeds") or 0)
        pnl = float(c.get("pnl_dollars") or 0)
        pnl_pct = float(c.get("pnl_pct") or 0)
        hold = c.get("holding_period_days") or 0
        cost_total += cost
        realized_total += pnl
        print(f"   {c.get('ticker'):<7}{c.get('shares'):>10}{c.get('entry_date'):>14}"
              f"   ${cost:>9,.2f}   ${proceeds:>9,.2f}   ${pnl:>+9,.2f}{pnl_pct:>+9.2f}%{hold:>9}")
    print(f"   {'TOTAL':<7}{'':>10}{'':>14}   ${cost_total:>9,.2f}{'':>12}   ${realized_total:>+9,.2f}")

    # ── NAV at start vs end ──
    print()
    print("=" * 80)
    print(f"  NAV REFERENCE")
    print("=" * 80)
    nav_idx = daily_nav.index
    prior = nav_idx[nav_idx < target_ts]
    same = nav_idx[nav_idx == target_ts]
    if len(prior) == 0:
        print("  no prior business day in NAV index")
        return
    nav_start = float(daily_nav["nav"].loc[prior[-1]])
    nav_end = float(daily_nav["nav"].loc[same[0]]) if len(same) > 0 else None
    print(f"\n  prior business day:  {prior[-1].date()}   NAV=${nav_start:,.2f}")
    if nav_end is not None:
        drop_pct = (nav_start - nav_end) / nav_start if nav_start > 0 else 0
        print(f"  same-day:            {same[0].date()}   NAV=${nav_end:,.2f}")
        print(f"  NAV drop:            {drop_pct*100:.2f}%")

    if nav_start > 0:
        loss_pct = realized_total / nav_start
        print(f"  realized_loss_pct_of_nav:  {loss_pct*100:.2f}%")

    # ── Subsequent buys in next 30 trading days ──
    print()
    print("=" * 80)
    print(f"  SUBSEQUENT BUYS IN NEXT 30 TRADING DAYS")
    print("=" * 80)
    nav_after = nav_idx[nav_idx > target_ts]
    if len(nav_after) >= 30:
        idle_end = nav_after[29]
        clean = _drop_cancelled(trades)
        buys_in = [
            t for t in clean
            if t.get("action") == "buy"
            and target_ts < pd.Timestamp(str(t.get("executed_at"))[:10]) <= idle_end
        ]
        print(f"\n  window: ({target_date}, {idle_end.date()}]   ({len(buys_in)} buys)")
        sub_value = sum(abs(float(t.get("amount") or 0)) for t in buys_in)
        liquidated = sum(abs(float(t.get("amount") or 0)) for t in day_trades if t.get("action") == "sell" and t.get("cancellation_status") == "normal")
        print(f"  liquidated value (SELLS on {target_date}): ${liquidated:,.2f}")
        print(f"  subsequent_buy_value:                       ${sub_value:,.2f}")
        print(f"  buyback ratio:                              {sub_value/liquidated*100:.1f}%  "
              f"(standard trigger requires < 10% to fire)")
        if buys_in:
            print(f"\n  subsequent buys:")
            for t in sorted(buys_in, key=lambda x: x.get("executed_at")):
                print(f"   {str(t.get('executed_at'))[:10]}  {t.get('ticker'):<6}  shares={t.get('shares')!s:<10}  "
                      f"amount=${float(t.get('amount') or 0):>10,.2f}")
    else:
        print(f"\n  forward NAV window only has {len(nav_after)} trading days; need 30")

    # ── Run detector to confirm output matches the trace ──
    print()
    print("=" * 80)
    print(f"  DETECTOR OUTPUT FOR {target_date}")
    print("=" * 80)
    from datetime import datetime, timedelta
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

    from lib.bias_detection import detect_panic_sells
    events = detect_panic_sells(trades, daily_nav, spy_daily)
    target_event = next((e for e in events if e["date"] == target_date), None)
    if target_event is None:
        print(f"\n  Detector did NOT flag {target_date} after fixes  ✓ (expected if false-positive logic now suppresses it)")
    else:
        print(f"\n  trigger_type:        {target_event['trigger_type']}")
        print(f"  trigger_paths_fired: {target_event['trigger_paths_fired']}")
        print(f"  severity:            {target_event['severity']}  (score {target_event['severity_score']})")
        print(f"  loss_pct_of_nav:     {target_event['total_loss_pct_of_nav']*100:.2f}%")
        print(f"  nav_drop_pct:        {target_event['nav_drop_pct']*100:.2f}%")
        print(f"  spy_6m_forward:      {target_event['market_context']['spy_return_6m_forward']*100:.2f}%" if target_event['market_context']['spy_return_6m_forward'] is not None else "  spy_6m_forward: n/a")


if __name__ == "__main__":
    main()
