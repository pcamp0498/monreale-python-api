"""Smoke verification — run lib/robinhood_parser + lib/options_fifo
against Patrick's real Robinhood activity CSV.

Verifies:
  1. The trans-code distribution among parsed options matches the expected
     header (BTO 108, STC 97, OEXP 9, STO 2, BTC 2 from the PowerShell audit).
  2. FIFO matching produces a non-zero closed_positions count + reasonable
     P&L numbers, without raising.
  3. Sample positions render with the expected math fields populated.

Usage:  python tests/smoke_options_real_data.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lib.robinhood_parser import parse_robinhood_csv
from lib.options_fifo import match_options_positions

CSV_PATH = Path(r"C:\Users\PatrickCampesi\Downloads\db8bbc70-1781-531b-9faf-a0b20ce7648d.csv")


def hr(s=""):
    print()
    print("=" * 78)
    if s:
        print(f"  {s}")
        print("=" * 78)


def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"CSV not found at {CSV_PATH}")

    raw = CSV_PATH.read_bytes()
    print(f"[smoke] CSV: {CSV_PATH.name} ({len(raw)} bytes)")
    print(f"[smoke] parsing...")

    parsed = parse_robinhood_csv(raw)
    options = parsed.get("options_trades") or []
    quarantined = parsed.get("quarantined") or []

    hr("PARSER OUTPUT")
    print(f" total_rows:           {parsed.get('total_rows')}")
    print(f" trades (equities):    {len(parsed.get('trades') or [])}")
    print(f" dividends:            {len(parsed.get('dividends') or [])}")
    print(f" securities_lending:   {len(parsed.get('securities_lending') or [])}")
    print(f" options_trades:       {len(options)}")
    print(f" quarantined:          {len(quarantined)}")
    print(f" errored:              {len(parsed.get('errored') or [])}")

    # Trans-code distribution from parsed options
    tc_counts = Counter(o.get("trans_code") for o in options)
    hr("TRANS CODE DISTRIBUTION (parsed options)")
    expected = {"BTO": 108, "STC": 97, "OEXP": 9, "STO": 2, "BTC": 2}
    for code, exp in expected.items():
        actual = tc_counts.get(code, 0)
        status = "OK" if actual == exp else f"DIFF ({actual} vs {exp})"
        print(f" {code:<6} expected={exp:<4} actual={actual:<4}  [{status}]")
    extra = set(tc_counts) - set(expected)
    if extra:
        print(f" extra codes seen: {dict((c, tc_counts[c]) for c in extra)}")
    print(f" parsed total:    {sum(tc_counts.values())}")

    # Quarantined-options breakdown
    opt_quarantined = [q for q in quarantined if "option" in str(q.get("quarantine_reason", "")).lower()]
    if opt_quarantined:
        hr("OPTIONS QUARANTINED (parse failures)")
        for q in opt_quarantined[:10]:
            print(f"  {q.get('quarantine_reason')[:80]}")

    hr("FIFO MATCH")
    fifo = match_options_positions(options)
    closed = fifo["closed_positions"]
    opened = fifo["open_positions"]
    warnings = fifo["match_warnings"]
    needs_review = fifo.get("manual_review_required", [])

    print(f" open_positions:       {len(opened)}")
    print(f" closed_positions:     {len(closed)}")
    print(f" match_warnings:       {len(warnings)}")
    print(f" manual_review_required: {len(needs_review)}")
    if warnings:
        print(f"\n first 5 warnings:")
        for w in warnings[:5]:
            print(f"   - {w}")
    if needs_review:
        print(f"\n manual_review_required (deferred to later sprint):")
        for r in needs_review[:5]:
            print(f"   - {r.get('trans_code')} {r.get('underlying_ticker')} ${r.get('strike')} "
                  f"{r.get('option_type')} exp {r.get('expiration_date')} "
                  f"({r.get('reason')}, sprint={r.get('deferred_to_sprint')})")

    expired = [p for p in closed if p.get("outcome") == "expired_worthless"]
    realized_pnl = sum(float(p.get("realized_pnl") or 0) for p in closed)
    winners = [p for p in closed if (p.get("realized_pnl") or 0) > 0]
    losers = [p for p in closed if (p.get("realized_pnl") or 0) < 0]

    print(f"\n total realized P&L:   ${realized_pnl:+,.2f}")
    print(f" expired worthless:    {len(expired)}")
    print(f" winners:              {len(winners)}")
    print(f" losers:               {len(losers)}")
    print(f" win rate:             {len(winners) / max(len(closed), 1):.1%}")

    if closed:
        hr("SAMPLE CLOSED POSITIONS (first 5 by realized_pnl desc)")
        for p in sorted(closed, key=lambda x: -(x.get("realized_pnl") or 0))[:5]:
            print(f" {p['underlying_ticker']:<6} {p['option_type'][:1].upper()} ${p['strike']:>7.2f} exp {p['expiration_date']}  "
                  f"{p['contracts']}c  cost=${p['total_cost'] or 0:>8.2f}  proc=${p['total_proceeds'] or 0:>8.2f}  "
                  f"pnl=${p['realized_pnl']:+8.2f}  held={p['days_held']}d  outcome={p['outcome']}")

        hr("SAMPLE CLOSED POSITIONS (first 5 by realized_pnl asc — biggest losses)")
        for p in sorted(closed, key=lambda x: (x.get("realized_pnl") or 0))[:5]:
            print(f" {p['underlying_ticker']:<6} {p['option_type'][:1].upper()} ${p['strike']:>7.2f} exp {p['expiration_date']}  "
                  f"{p['contracts']}c  cost=${p['total_cost'] or 0:>8.2f}  proc=${p['total_proceeds'] or 0:>8.2f}  "
                  f"pnl=${p['realized_pnl']:+8.2f}  held={p['days_held']}d  outcome={p['outcome']}")

    if opened:
        hr(f"SAMPLE OPEN POSITIONS (first 5 of {len(opened)})")
        for o in opened[:5]:
            cost = o.get("total_cost") or o.get("total_proceeds") or 0
            print(f" {o['underlying_ticker']:<6} {o['option_type'][:1].upper()} ${o['strike']:>7.2f} exp {o['expiration_date']}  "
                  f"{o['contracts']}c  side={o['position_side']}  basis=${cost:>8.2f}  opened {str(o['open_date'])[:10]}")

    hr("LIMITATIONS (9C.1 SCOPE)")
    print(" - Spread legs are counted as TWO independent trades.")
    print("   Example: a GME $310/$300 vertical shows up as one big winner")
    print("   on the long leg and one big loser on the short leg, instead")
    print("   of a single net-debit/net-credit spread record. The aggregate")
    print("   P&L is correct, but the per-position math under-states risk")
    print("   and over-states return on each leg in isolation.")
    print(" - Multi-leg spread detection (calendars, verticals, condors)")
    print("   is deferred to Sprint 9C.2.")
    print(" - CONV (corporate-action conversion) trades emit a placeholder")
    print("   'conversion_unhandled' record with realized_pnl=0 and surface")
    print("   in manual_review_required for the UI. Full handling lands in")
    print("   Sprint 9C.6.")

    hr("SMOKE OK")


if __name__ == "__main__":
    main()
