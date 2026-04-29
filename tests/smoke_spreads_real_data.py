"""Smoke verification — detect spreads against Patrick's live Supabase data.

Pulls all options_closed_positions for Patrick's user_id, runs the
detection algorithm, prints the spread_type breakdown + per-type
realized_pnl totals, and asserts the expected baseline from the SQL
diagnostic Patrick ran during 9C.2 planning.

Required env:
    SUPABASE_URL   (or NEXT_PUBLIC_SUPABASE_URL)
    SUPABASE_SERVICE_ROLE_KEY

Run: python tests/smoke_spreads_real_data.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lib.options_spreads import detect_spreads

USER_ID = "33e2c662-d13f-4aa9-bfc1-1187cbab9d24"


def hr(s=""):
    print()
    print("=" * 78)
    if s:
        print(f"  {s}")
        print("=" * 78)


def main():
    try:
        from supabase import create_client
    except ImportError:
        raise SystemExit("supabase-py not installed; pip install supabase")

    url = os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise SystemExit(
            "SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY required in environment"
        )

    sb = create_client(url, key)

    hr("FETCH")
    rows = (
        sb.table("options_closed_positions")
        .select("*")
        .eq("user_id", USER_ID)
        .execute()
    )
    closed_positions = rows.data or []
    print(f" pulled {len(closed_positions)} closed_positions for user {USER_ID}")

    hr("DETECT")
    spreads = detect_spreads(closed_positions)
    n_legs = sum(len(s.get("leg_position_ids") or []) for s in spreads)
    print(f" detected {len(spreads)} spreads, clustered {n_legs} legs")

    hr("BREAKDOWN BY SPREAD TYPE")
    breakdown = Counter(s["spread_type"] for s in spreads)
    for t, n in sorted(breakdown.items(), key=lambda x: -x[1]):
        pnl = sum(float(s.get("realized_pnl") or 0) for s in spreads if s["spread_type"] == t)
        print(f" {t:<22} n={n:<3} total_pnl=${pnl:+,.2f}")

    hr("CONFIDENCE BREAKDOWN")
    conf = Counter(s.get("detection_confidence") for s in spreads)
    for c, n in sorted(conf.items()):
        print(f" {c:<8} n={n}")

    hr("SAMPLE — top 10 by realized_pnl desc")
    for s in sorted(spreads, key=lambda x: -(x.get("realized_pnl") or 0))[:10]:
        print(
            f" {s['underlying_ticker']:<6} {s['spread_type']:<22} "
            f"open={str(s['open_date'])[:10]} exp={str(s['expiration_date'])[:10]}  "
            f"pnl=${s['realized_pnl']:+9.2f}  pct="
            f"{(s['realized_pnl_pct'] if s['realized_pnl_pct'] is not None else 0):+.1f}%  "
            f"legs={len(s['leg_position_ids'])}"
        )

    # ─── Assertions against the planning-session diagnostic baseline ─────
    hr("ASSERTIONS")
    assert len(spreads) >= 9, f"expected >=9 spreads, got {len(spreads)}"
    print(f" OK: spreads_detected ({len(spreads)}) >= 9")

    # GME — 2021 vertical. The exact direction depends on Patrick's leg
    # signs, so accept either bull_call_spread or bear_call_spread.
    gme_verticals = [
        s for s in spreads
        if s["underlying_ticker"] == "GME"
        and s["spread_type"] in ("bull_call_spread", "bear_call_spread")
    ]
    assert gme_verticals, "expected at least one GME vertical call spread"
    print(f" OK: GME vertical detected: {[s['spread_type'] for s in gme_verticals]}")

    # TGT 2019-11 long_straddle (call+put same strike, both long, same date)
    tgt_straddles = [
        s for s in spreads
        if s["underlying_ticker"] == "TGT" and s["spread_type"] == "long_straddle"
    ]
    if not tgt_straddles:
        # If TGT was a strangle (different strikes) instead of straddle, accept that
        tgt_strangles = [
            s for s in spreads
            if s["underlying_ticker"] == "TGT" and "strangle" in s["spread_type"]
        ]
        assert tgt_strangles, "expected TGT long_straddle or long_strangle"
        print(f" OK: TGT detected as {tgt_strangles[0]['spread_type']} (strangle in lieu of straddle)")
    else:
        print(f" OK: TGT long_straddle detected (n={len(tgt_straddles)})")

    # At least one strangle in the dataset (per planning-session diagnostic)
    strangles = [s for s in spreads if "strangle" in s["spread_type"]]
    assert strangles, "expected at least one strangle"
    print(f" OK: strangles detected: {[(s['underlying_ticker'], s['spread_type']) for s in strangles[:3]]}")

    # At least one stacked_long_calls (NIO/LULU pattern)
    stacked = [s for s in spreads if s["spread_type"] == "stacked_long_calls"]
    assert stacked, "expected at least one stacked_long_calls"
    print(f" OK: stacked_long_calls detected: {[s['underlying_ticker'] for s in stacked[:3]]}")

    hr("SMOKE OK")


if __name__ == "__main__":
    main()
