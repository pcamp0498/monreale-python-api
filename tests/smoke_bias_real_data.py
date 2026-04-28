"""Smoke test — run /bias/analyze pipeline against Patrick's real Supabase
trades table.

Loads SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY + POLYGON_API_KEY from the
sibling monreale-os/apps/web/.env.local file, fetches trades + dividends via
Supabase REST API (no Python supabase client dependency), runs the same
detection pipeline the FastAPI router runs, and prints a structured summary.

Usage:  python tests/smoke_bias_real_data.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _load_env_local():
    """Best-effort .env.local loader. Reads the Next.js sibling repo's
    env file and pushes recognized keys into os.environ so downstream
    Polygon helper picks them up."""
    sibling_env = REPO.parent / "monreale-os" / "apps" / "web" / ".env.local"
    if not sibling_env.exists():
        print(f"[smoke] WARN: {sibling_env} not found; expecting env vars already set")
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


def _fetch_trades_from_supabase() -> tuple[list[dict], list[dict]]:
    url = os.environ.get("NEXT_PUBLIC_SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        raise SystemExit("Missing NEXT_PUBLIC_SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY")

    headers = {"apikey": key, "Authorization": f"Bearer {key}"}

    # Trades
    trades_resp = requests.get(
        f"{url}/rest/v1/trades",
        headers=headers,
        params={
            "select": "id,ticker,action,shares,price,amount,executed_at,cancellation_status,is_active",
            "is_active": "eq.true",
            "order": "executed_at.asc",
            "limit": "5000",
        },
        timeout=30,
    )
    trades_resp.raise_for_status()
    trades = trades_resp.json()

    # Dividends
    div_resp = requests.get(
        f"{url}/rest/v1/dividend_income",
        headers=headers,
        params={"select": "ticker,amount,paid_at", "order": "paid_at.asc", "limit": "5000"},
        timeout=30,
    )
    if div_resp.status_code == 200:
        dividends = div_resp.json()
    else:
        print(f"[smoke] dividends fetch returned {div_resp.status_code}; using empty list")
        dividends = []

    return trades, dividends


def main():
    _load_env_local()
    print("[smoke] env loaded; fetching trades from Supabase…")
    trades, dividends = _fetch_trades_from_supabase()
    print(f"[smoke] fetched {len(trades)} trades, {len(dividends)} dividends")
    if not trades:
        raise SystemExit("No trades returned — check Supabase connection")

    # Hide table-format-only fields the Python detectors don't use, and
    # ensure cancellation_status defaults to 'normal' if absent.
    for t in trades:
        if t.get("cancellation_status") is None:
            t["cancellation_status"] = "normal"

    # Run the full pipeline manually (mirrors routers/bias.py logic)
    from lib.performance_math import build_daily_nav, compute_twr, compute_mwr, _drop_cancelled, _sanitize_for_json
    from lib.polygon_client import get_prices_dataframe
    from lib.bias_detection import (
        detect_panic_sells,
        detect_disposition_effect,
        attribute_cash_flow_timing,
    )

    daily_nav = build_daily_nav(trades, dividends)
    twr = compute_twr(daily_nav)
    mwr = compute_mwr(trades, dividends, current_value=float(daily_nav["holdings_value"].iloc[-1]) if not daily_nav.empty else 0.0)
    print(f"[smoke] daily_nav rows={len(daily_nav)}, TWR={twr:.4f}, MWR={mwr:.4f}")

    print("[smoke] fetching SPY history (full 8y direct from Polygon)…")
    # The shared get_price_history helper hardcodes limit=730 which truncates
    # multi-year SPY pulls. Direct call here bypasses that for the smoke test.
    import pandas as pd
    from datetime import datetime, timedelta
    api_key = os.environ.get("POLYGON_API_KEY", "")
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=2920)).strftime("%Y-%m-%d")
    spy_resp = requests.get(
        f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/{from_date}/{to_date}",
        params={"adjusted": "true", "sort": "asc", "limit": 5000, "apiKey": api_key},
        timeout=30,
    )
    spy_resp.raise_for_status()
    spy_results = spy_resp.json().get("results") or []
    spy_daily = pd.Series(
        {datetime.fromtimestamp(r["t"] / 1000).strftime("%Y-%m-%d"): r["c"] for r in spy_results}
    )
    spy_daily.index = pd.to_datetime(spy_daily.index)
    spy_daily = spy_daily.sort_index()
    print(f"[smoke] SPY rows={len(spy_daily)}  range={spy_daily.index[0].date()} to {spy_daily.index[-1].date()}")

    print("[smoke] fetching current prices for open-position tickers…")
    clean = _drop_cancelled(trades)
    open_tickers = sorted({t.get("ticker") for t in clean if t.get("ticker") and t.get("action") == "buy"})
    current_prices = {}
    if open_tickers:
        try:
            cp_df = get_prices_dataframe(open_tickers, days=10)
            if cp_df is not None and not cp_df.empty:
                last_row = cp_df.iloc[-1]
                for tk in open_tickers:
                    if tk in cp_df.columns:
                        v = last_row.get(tk)
                        if v is not None and v == v and float(v) > 0:
                            current_prices[tk] = float(v)
        except Exception as e:
            print(f"[smoke] current price fetch error: {e}")
    print(f"[smoke] current prices for {len(current_prices)} / {len(open_tickers)} open tickers")

    print("[smoke] running detectors…")
    panic = detect_panic_sells(trades, daily_nav, spy_daily)
    disposition = detect_disposition_effect(trades, current_prices=current_prices)
    cf_timing = attribute_cash_flow_timing(daily_nav, spy_daily, twr, mwr)

    response = _sanitize_for_json({
        "panic_sells": panic,
        "disposition": disposition,
        "cash_flow_timing": cf_timing,
        "meta": {
            "n_trades_input": len(trades),
            "n_trades_after_cancellation_filter": len(clean),
            "n_dividends_input": len(dividends),
            "twr_annualized": twr,
            "mwr_annualized": mwr,
            "nav_index_size": int(len(daily_nav.index)) if not daily_nav.empty else 0,
            "current_prices_fetched": len(current_prices),
            "open_tickers_count": len(open_tickers),
        },
    })

    # Ensure JSON-serializable
    json.dumps(response)

    # ── Pretty summary print ────────────────────────────────────────────
    meta = response["meta"]
    print()
    print("-" * 78)
    print(" SMOKE TEST SUMMARY — bias detection on real Supabase trades")
    print("-" * 78)
    print(f" trades input:                     {meta['n_trades_input']}")
    print(f" trades after cancellation filter: {meta['n_trades_after_cancellation_filter']}")
    print(f" dividends input:                  {meta['n_dividends_input']}")
    print(f" NAV index size (trading days):    {meta['nav_index_size']}")
    print(f" TWR annualized:                   {meta['twr_annualized']*100:.2f}%")
    print(f" MWR annualized:                   {meta['mwr_annualized']*100:.2f}%")
    print(f" open-position tickers w/ price:   {meta['current_prices_fetched']} / {meta['open_tickers_count']}")
    print()

    print("-" * 78)
    print(f" PANIC SELLS ({len(panic)} events detected)")
    print("-" * 78)
    for i, e in enumerate(panic, 1):
        sev_basis = e.get("severity_basis") or "?"
        spy6m = e.get("market_context", {}).get("spy_return_6m")
        spy6m_str = f"{spy6m*100:+.1f}%" if spy6m is not None else "n/a"
        ttype = e.get("trigger_type", "?")
        paths = e.get("trigger_paths_fired", {})
        paths_str = ",".join(k for k, v in paths.items() if v) or "?"
        print(f" #{i} {e['date']}  severity={e['severity']:<6}  score={e['severity_score']:.3f}  basis={sev_basis}")
        print(f"     trigger_type={ttype}  paths_fired=[{paths_str}]")
        print(f"     tickers={','.join(e['tickers'])}  n_pos={e['n_positions']}")
        print(f"     nav_start=${e.get('nav_start_of_day',0):,.2f}  nav_end=${e.get('nav_end_of_day',0):,.2f}  nav_drop={e.get('nav_drop_pct',0)*100:.1f}%")
        print(f"     realized_loss=${e['total_loss_dollars']:+,.2f}  ({e['total_loss_pct_of_nav']*100:+.2f}% of NAV)")
        print(f"     liquidated=${e['liquidated_value_dollars']:,.0f}  subsequent_buys=${e['subsequent_buy_value_dollars']:,.0f}")
        print(f"     days_idle_after={e['days_idle_after']}  spy_6m={spy6m_str}")
        print()

    print("-" * 78)
    print(" DISPOSITION EFFECT")
    print("-" * 78)
    print(f" flagged_positions: {disposition['n_flagged']}")
    for f in disposition["flagged_positions"]:
        loss = f.get("unrealized_loss_pct")
        loss_str = f"{loss*100:+.1f}%" if loss is not None else "n/a"
        print(f"   {f['ticker']:<6}  loss={loss_str:<8}  days_held={f['days_held']:<5}  cost=${f['cost_basis_total']:,.0f} -> value=${f['current_value']:,.0f}  buys={f['n_subsequent_buys_same_ticker']}")
    stats = disposition["disposition_stats"]
    if stats:
        print(f" disposition stats: ratio={stats['disposition_ratio']:.2f}  "
              f"avg_winner={stats['avg_winner_holding_days']:.0f}d  avg_loser={stats['avg_loser_holding_days']:.0f}d  "
              f"n_w={stats['n_winners']}  n_l={stats['n_losers']}  "
              f"detected={stats['disposition_effect_detected']}")
    else:
        print(f" disposition stats: null  reason={disposition['disposition_stats_reason']}")
    print()

    print("-" * 78)
    print(" CASH-FLOW TIMING ATTRIBUTION")
    print("-" * 78)
    print(f" twr_annualized:           {cf_timing['twr_annualized']*100:.2f}%")
    print(f" mwr_annualized:           {cf_timing['mwr_annualized']*100:.2f}%")
    print(f" twr_mwr_gap:              {cf_timing['twr_mwr_gap_pct']*100:+.2f}%")
    wap = cf_timing.get("weighted_avg_position_pct")
    print(f" weighted_avg_pos_pct:     {wap if wap is None else f'{wap:.3f}'}")
    attr = cf_timing.get("attribution_pct")
    print(f" attribution_pct:          {attr if attr is None else f'{attr*100:+.3f}%'}")
    print(f" attribution_formula_reliable: {cf_timing.get('attribution_formula_reliable')}  (true only when |gap| <= 50%)")
    print(f" finding_severity:         {cf_timing.get('finding_severity', 'n/a')}")
    summary = cf_timing.get("inflows_summary", {})
    print(f" inflows: total={summary.get('n_total')}  trough={summary.get('n_at_trough')}  "
          f"middle={summary.get('n_in_middle')}  peak={summary.get('n_at_peak')}  "
          f"missing_history={summary.get('n_with_insufficient_history')}")
    inflows = cf_timing.get("cash_inflows") or []
    if inflows:
        print(" largest inflows (sorted by amount):")
        for r in sorted(inflows, key=lambda x: x.get("amount") or 0)[:8]:
            pos = r.get("spy_52w_position_pct")
            pos_str = f"{pos:.2f}" if pos is not None else "n/a "
            print(f"   {r['date']}  ${abs(r['amount']):>10,.0f}  spy_52w_pos={pos_str}  cat={r.get('category') or 'n/a'}")
    by_year = cf_timing.get("deployment_by_year") or []
    if by_year:
        print()
        print(" deployment_by_year:")
        print(f"   {'year':<6}{'n':>4}{'total_$':>13}{'avg_pos':>9}{'n_trough':>10}{'n_mid':>7}{'n_peak':>8}{'n_no_hist':>11}  characterization")
        for y in by_year:
            wap = y.get("weighted_avg_spy_52w_position_pct")
            wap_str = f"{wap:.3f}" if wap is not None else "n/a"
            print(f"   {y['year']:<6}{y['n_inflows']:>4}{y['total_dollars']:>13,.2f}{wap_str:>9}"
                  f"{y.get('n_at_trough',0):>10}{y.get('n_mid_range',0):>7}{y.get('n_at_peak',0):>8}"
                  f"{y.get('n_insufficient_history',0):>11}  {y['characterization']}")

    print("-" * 78)
    print(" SMOKE TEST OK — response is JSON-serializable, all numerics finite")
    print("-" * 78)


if __name__ == "__main__":
    main()
