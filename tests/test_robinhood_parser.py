"""Synthetic-CSV unit tests for the Robinhood parser.

Run with:  python -m pytest tests/test_robinhood_parser.py -v
"""
import sys
from pathlib import Path

# Ensure repo root is importable when run via pytest from the package dir
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.robinhood_parser import parse_robinhood_csv, parse_amount, parse_date


# Synthetic CSV exercising every required path:
#   row 1: AAPL Buy           → trade buy
#   row 2: MSFT Sell          → trade sell
#   row 3: VTI CDIV           → dividend (cash)
#   row 4: AAPL option STC    → quarantined (options)
#   row 5: BTC Buy            → quarantined (crypto)
#   row 6: ACH (no instrument)→ ignored silently
#   row 7: GOOGL XYZ          → errored (unknown_trans_code)
SAMPLE_CSV = """Activity Date,Process Date,Settle Date,Instrument,Description,Trans Code,Quantity,Price,Amount
01/15/2025,01/15/2025,01/17/2025,AAPL,APPLE INC,Buy,10,$150.00,($1500.00)
02/20/2025,02/20/2025,02/22/2025,MSFT,MICROSOFT CORP,Sell,5,$300.00,$1500.00
03/15/2025,03/15/2025,03/15/2025,VTI,VANGUARD ETF,CDIV,,,$45.50
04/01/2025,04/01/2025,04/01/2025,AAPL 240119C00150000,APPLE CALL OPTION,STC,1,$5.00,$500.00
05/01/2025,05/01/2025,05/01/2025,BTC,BITCOIN,Buy,0.001,$50000.00,($50.00)
06/01/2025,06/01/2025,06/01/2025,,ACH DEPOSIT,ACH,,,$1000.00
07/01/2025,07/01/2025,07/01/2025,GOOGL,ALPHABET,XYZ,1,$100.00,($100.00)
"""


def test_synthetic_csv_partition():
    result = parse_robinhood_csv(SAMPLE_CSV.encode("utf-8"))
    assert len(result["trades"]) == 2, f"expected 2 trades, got {len(result['trades'])}"
    assert len(result["dividends"]) == 1
    assert len(result["quarantined"]) == 2
    assert len(result["errored"]) == 1
    assert result["total_rows"] == 7

    actions = sorted(t["action"] for t in result["trades"])
    assert actions == ["buy", "sell"]
    tickers = sorted(t["ticker"] for t in result["trades"])
    assert tickers == ["AAPL", "MSFT"]


def test_dividend_fields():
    result = parse_robinhood_csv(SAMPLE_CSV.encode("utf-8"))
    div = result["dividends"][0]
    assert div["ticker"] == "VTI"
    assert div["amount"] == 45.50
    assert div["paid_at"] == "2025-03-15"
    assert div["dividend_type"] == "cash"


def test_quarantine_reasons():
    result = parse_robinhood_csv(SAMPLE_CSV.encode("utf-8"))
    reasons = sorted(q["quarantine_reason"] for q in result["quarantined"])
    assert reasons == ["crypto", "options"]
    by_reason = {q["quarantine_reason"]: q for q in result["quarantined"]}
    assert by_reason["options"]["ticker_or_symbol"] == "AAPL 240119C00150000"
    assert by_reason["crypto"]["ticker_or_symbol"] == "BTC"


def test_errored_carries_code():
    result = parse_robinhood_csv(SAMPLE_CSV.encode("utf-8"))
    assert "unknown_trans_code:XYZ" in result["errored"][0]["reason"]


def test_buy_amount_parens_negative():
    result = parse_robinhood_csv(SAMPLE_CSV.encode("utf-8"))
    aapl = next(t for t in result["trades"] if t["ticker"] == "AAPL")
    assert aapl["amount"] == -1500.00
    assert aapl["price"] == 150.00
    assert aapl["shares"] == 10


def test_date_range():
    result = parse_robinhood_csv(SAMPLE_CSV.encode("utf-8"))
    start, end = result["date_range"]
    # dates are collected only for rows that hit equity/dividend/split branches
    assert start == "2025-01-15"
    assert end == "2025-07-01"


def test_bom_handling():
    csv_with_bom = "﻿" + SAMPLE_CSV
    result = parse_robinhood_csv(csv_with_bom.encode("utf-8"))
    assert len(result["trades"]) == 2  # BOM doesn't break header parsing


def test_buy_buy_sell_three_trades():
    """FIFO matching is performance-engine territory; here we just confirm the
    parser preserves all three rows in order so the engine can match them."""
    csv = (
        "Activity Date,Process Date,Settle Date,Instrument,Description,Trans Code,Quantity,Price,Amount\n"
        "01/01/2025,01/01/2025,01/03/2025,AAPL,APPLE,Buy,10,$100.00,($1000.00)\n"
        "01/15/2025,01/15/2025,01/17/2025,AAPL,APPLE,Buy,10,$110.00,($1100.00)\n"
        "02/01/2025,02/01/2025,02/03/2025,AAPL,APPLE,Sell,5,$120.00,$600.00\n"
    )
    result = parse_robinhood_csv(csv.encode("utf-8"))
    assert len(result["trades"]) == 3
    assert [t["action"] for t in result["trades"]] == ["buy", "buy", "sell"]


def test_parse_amount_helpers():
    assert parse_amount("$1,500.00") == 1500.00
    assert parse_amount("($1,500.00)") == -1500.00
    assert parse_amount("") is None
    assert parse_amount(None) is None
    assert parse_amount("garbage") is None


def test_parse_date_helpers():
    assert parse_date("01/15/2025") == "2025-01-15"
    assert parse_date("") is None
    assert parse_date("not-a-date") is None
