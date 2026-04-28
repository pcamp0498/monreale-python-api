"""Smoke verification — runs compute_headline_stats against Patrick's real
Supabase trades and prints the benchmark fields.

Confirms the SPY-coverage-truncation fix produces non-zero
benchmark_total_return / benchmark_annualized_return + correct
benchmark_coverage_start / benchmark_coverage_end values.
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

    print("[smoke] fetching trades + dividends from Supabase...")
    trades = requests.get(
        f"{url}/rest/v1/trades",
        headers=headers,
        params={
            "select": "id,ticker,action,shares,price,amount,executed_at,asset_type,cancellation_status,is_active",
            "is_active": "eq.true",
            "order": "executed_at.asc",
            "limit": "5000",
        },
        timeout=30,
    ).json()
    dividends = requests.get(
        f"{url}/rest/v1/dividend_income",
        headers=headers,
        params={"select": "ticker,amount,paid_at,dividend_type", "order": "paid_at.asc", "limit": "5000"},
        timeout=30,
    ).json()
    for t in trades:
        if t.get("cancellation_status") is None:
            t["cancellation_status"] = "normal"
    print(f"[smoke] fetched {len(trades)} trades, {len(dividends)} dividends")

    from lib.performance_math import compute_headline_stats

    print("[smoke] running compute_headline_stats (hits Polygon for SPY)...")
    stats = compute_headline_stats(trades, dividends, benchmark_ticker="SPY", rf_rate=0.04)

    print()
    print("=" * 78)
    print(" BENCHMARK COVERAGE FIX VERIFICATION")
    print("=" * 78)
    print(f" benchmark_ticker:             {stats.get('benchmark_ticker')}")
    print(f" benchmark_total_return:       {stats.get('benchmark_total_return')}  ({(stats.get('benchmark_total_return') or 0)*100:.2f}%)")
    print(f" benchmark_annualized_return:  {stats.get('benchmark_annualized_return')}  ({(stats.get('benchmark_annualized_return') or 0)*100:.2f}%)")
    print(f" benchmark_coverage_start:     {stats.get('benchmark_coverage_start')}")
    print(f" benchmark_coverage_end:       {stats.get('benchmark_coverage_end')}")
    print()
    print(f" portfolio date_range.start:   {stats.get('date_range', {}).get('start')}")
    print(f" portfolio date_range.end:     {stats.get('date_range', {}).get('end')}")
    print("=" * 78)

    btr = stats.get("benchmark_total_return") or 0
    bar = stats.get("benchmark_annualized_return") or 0
    if btr == 0.0 and bar == 0.0:
        print(" ✗ BUG STILL PRESENT — both benchmark fields are 0.00%")
        sys.exit(1)
    print(" ✓ benchmark fields are non-zero — SPY-truncation fix is working")
    if stats.get("benchmark_coverage_start") and stats.get("benchmark_coverage_end"):
        print(f" ✓ coverage range emitted: {stats['benchmark_coverage_start']} -> {stats['benchmark_coverage_end']}")
    print("=" * 78)


if __name__ == "__main__":
    main()
