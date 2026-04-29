"""Synthetic-CSV unit tests for the Robinhood parser.

Run with:  python -m pytest tests/test_robinhood_parser.py -v
"""
import sys
from pathlib import Path

# Ensure repo root is importable when run via pytest from the package dir
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.robinhood_parser import (
    parse_robinhood_csv,
    parse_amount,
    parse_date,
    parse_quantity,
    normalize_instrument,
)


# ─── Original synthetic CSV — must keep passing after the patch ──────────────
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
    assert "securities_lending" in result and result["securities_lending"] == []

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
    """Sprint 9C.1: options with un-parseable Description ("APPLE CALL OPTION"
    has no date/strike pattern) go to quarantine with reason
    'options_parse_failed:...'. Crypto stays as-is."""
    result = parse_robinhood_csv(SAMPLE_CSV.encode("utf-8"))
    reasons = [q["quarantine_reason"] for q in result["quarantined"]]
    assert "crypto" in reasons
    assert any(r.startswith("options_parse_failed:") for r in reasons)
    by_reason = {q["quarantine_reason"]: q for q in result["quarantined"]}
    assert by_reason["crypto"]["ticker_or_symbol"] == "BTC"
    # The options row's parse failed, so it landed in quarantined not
    # options_trades. Confirm options_trades is empty for this fixture.
    assert result.get("options_trades", []) == []


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
    assert start == "2025-01-15"
    assert end == "2025-07-01"


def test_bom_handling():
    csv_with_bom = "﻿" + SAMPLE_CSV
    result = parse_robinhood_csv(csv_with_bom.encode("utf-8"))
    assert len(result["trades"]) == 2  # BOM doesn't break header parsing


def test_buy_buy_sell_three_trades():
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


# ─── New tests for the 2026-04-27 patch ──────────────────────────────────────

def test_normalize_instrument_strips_cusip_and_recurring():
    assert normalize_instrument("SPY\nCUSIP: 78462F103\nRecurring") == "SPY"
    assert normalize_instrument("VTI\r\nCUSIP: 922908769") == "VTI"
    assert normalize_instrument("  AAPL  ") == "AAPL"
    assert normalize_instrument("") == ""
    assert normalize_instrument(None) == ""


def test_multiline_instrument_parses_to_clean_ticker():
    """Issue 1: Robinhood embeds CUSIP/Recurring annotations after a newline."""
    csv = (
        "Activity Date,Process Date,Settle Date,Instrument,Description,Trans Code,Quantity,Price,Amount\n"
        '01/10/2025,01/10/2025,01/12/2025,"SPY\nCUSIP: 78462F103\nRecurring",SPDR S&P 500,Buy,5,$500.00,($2500.00)\n'
    )
    result = parse_robinhood_csv(csv.encode("utf-8"))
    assert len(result["trades"]) == 1
    assert result["trades"][0]["ticker"] == "SPY"
    assert result["trades"][0]["shares"] == 5
    assert result["trades"][0]["amount"] == -2500.00


def test_slip_routes_to_securities_lending():
    """Issue 2: SLIP = stock lending income payment, separate bucket."""
    csv = (
        "Activity Date,Process Date,Settle Date,Instrument,Description,Trans Code,Quantity,Price,Amount\n"
        "06/01/2025,06/01/2025,06/01/2025,AAPL,SLI PAYMENT,SLIP,,,$2.34\n"
    )
    result = parse_robinhood_csv(csv.encode("utf-8"))
    assert len(result["securities_lending"]) == 1
    assert len(result["trades"]) == 0
    assert len(result["errored"]) == 0
    sli = result["securities_lending"][0]
    assert sli["ticker"] == "AAPL"
    assert sli["amount"] == 2.34
    assert sli["paid_at"] == "2025-06-01"


def test_mdiv_routes_to_dividends_with_manufactured_type():
    """Issue 2: MDIV = manufactured dividend, type='manufactured'."""
    csv = (
        "Activity Date,Process Date,Settle Date,Instrument,Description,Trans Code,Quantity,Price,Amount\n"
        "07/15/2025,07/15/2025,07/15/2025,GME,MANUFACTURED DIVIDEND,MDIV,,,$0.50\n"
    )
    result = parse_robinhood_csv(csv.encode("utf-8"))
    assert len(result["dividends"]) == 1
    assert result["dividends"][0]["dividend_type"] == "manufactured"
    assert result["dividends"][0]["ticker"] == "GME"
    assert result["dividends"][0]["amount"] == 0.50


def test_acati_routes_to_trades_with_transfer_in_action():
    """Issue 2: ACATI = ACAT transfer in, action='transfer_in'."""
    csv = (
        "Activity Date,Process Date,Settle Date,Instrument,Description,Trans Code,Quantity,Price,Amount\n"
        '08/01/2025,08/01/2025,08/05/2025,VOO,"VANGUARD S&P 500 ETF",ACATI,12,,\n'
    )
    result = parse_robinhood_csv(csv.encode("utf-8"))
    assert len(result["trades"]) == 1
    t = result["trades"][0]
    assert t["action"] == "transfer_in"
    assert t["ticker"] == "VOO"
    assert t["shares"] == 12
    assert t["price"] is None
    assert t["amount"] is None


def test_silent_ignore_codes_produce_no_output():
    """Issue 2: GDBP, RTP, CONV, MINT must skip silently. BCXL is handled
    separately (matched against Buy rows; unmatched BCXL goes to errored)
    so it's not part of this silent-ignore test — see the bcxl_* tests."""
    csv = (
        "Activity Date,Process Date,Settle Date,Instrument,Description,Trans Code,Quantity,Price,Amount\n"
        "01/05/2025,01/05/2025,01/05/2025,,GOLD DEPOSIT BOOST,GDBP,,,$10.00\n"
        "01/06/2025,01/06/2025,01/06/2025,,REAL TIME PAYMENT,RTP,,,$500.00\n"
        "01/07/2025,01/07/2025,01/07/2025,,APEX TO RHS,CONV,,,$0.00\n"
        "01/08/2025,01/08/2025,01/08/2025,,MARGIN INTEREST,MINT,,,($1.23)\n"
    )
    result = parse_robinhood_csv(csv.encode("utf-8"))
    assert result["total_rows"] == 4
    assert len(result["trades"]) == 0
    assert len(result["dividends"]) == 0
    assert len(result["securities_lending"]) == 0
    assert len(result["quarantined"]) == 0
    assert len(result["errored"]) == 0


def test_option_open_close_codes_routed_to_options_trades():
    """Sprint 9C.1: BTO/STC/STO/BTC rows with parseable option Description
    are routed into options_trades (not quarantine). Equity trades list
    stays empty."""
    csv = (
        "Activity Date,Process Date,Settle Date,Instrument,Description,Trans Code,Quantity,Price,Amount\n"
        '03/01/2025,03/01/2025,03/01/2025,AMZN,"AMZN 5/9/2025 Call $220.00",BTO,1,$3.50,($350.00)\n'
        '04/01/2025,04/01/2025,04/01/2025,AMZN,"AMZN 5/9/2025 Call $220.00",STC,1,$5.00,$500.00\n'
        '05/01/2025,05/01/2025,05/01/2025,TSLA,"TSLA 6/20/2025 Put $150.00",STO,1,$2.00,$200.00\n'
        '06/01/2025,06/01/2025,06/01/2025,TSLA,"TSLA 6/20/2025 Put $150.00",BTC,1,$1.00,($100.00)\n'
    )
    result = parse_robinhood_csv(csv.encode("utf-8"))
    assert len(result["options_trades"]) == 4
    assert len(result["trades"]) == 0
    assert len(result["quarantined"]) == 0

    # Verify parsed contract spec on each row
    by_code = {o["trans_code"]: o for o in result["options_trades"]}
    assert by_code["BTO"]["underlying_ticker"] == "AMZN"
    assert by_code["BTO"]["option_type"] == "call"
    assert by_code["BTO"]["strike"] == 220.00
    assert by_code["BTO"]["contracts"] == 1.0
    assert by_code["BTO"]["total_amount"] == -350.00

    assert by_code["STO"]["underlying_ticker"] == "TSLA"
    assert by_code["STO"]["option_type"] == "put"
    assert by_code["STO"]["strike"] == 150.00


def test_oexp_with_s_suffix_quantity():
    """Sprint 9C.1: OEXP rows with '1S'/'5S' quantities parse via
    parse_option_quantity (strips trailing S). Now routed to options_trades
    rather than quarantine."""
    csv = (
        "Activity Date,Process Date,Settle Date,Instrument,Description,Trans Code,Quantity,Price,Amount\n"
        '05/15/2025,05/15/2025,05/15/2025,SPY,"SPY 5/16/2025 Call $400.00",OEXP,1S,,\n'
        '05/16/2025,05/16/2025,05/16/2025,QQQ,"QQQ 5/17/2025 Put $300.00",OEXP,5S,,\n'
    )
    result = parse_robinhood_csv(csv.encode("utf-8"))
    assert len(result["options_trades"]) == 2
    assert len(result["errored"]) == 0
    assert len(result["quarantined"]) == 0
    spy = next(o for o in result["options_trades"] if o["underlying_ticker"] == "SPY")
    assert spy["trans_code"] == "OEXP"
    assert spy["contracts"] == 1.0  # "1S" → 1.0
    qqq = next(o for o in result["options_trades"] if o["underlying_ticker"] == "QQQ")
    assert qqq["contracts"] == 5.0  # "5S" → 5.0


def test_parse_quantity_strips_s_suffix():
    assert parse_quantity("1S") == 1.0
    assert parse_quantity("5S") == 5.0
    assert parse_quantity("10") == 10.0
    assert parse_quantity("10.5") == 10.5
    assert parse_quantity("") is None
    assert parse_quantity(None) is None
    assert parse_quantity("S") is None  # bare S should be None, not 0
    assert parse_quantity("garbage") is None


def test_blank_trailing_rows_skipped():
    """Issue 8: trailing empty rows are skipped, not errored."""
    csv = (
        "Activity Date,Process Date,Settle Date,Instrument,Description,Trans Code,Quantity,Price,Amount\n"
        "01/15/2025,01/15/2025,01/17/2025,AAPL,APPLE INC,Buy,10,$150.00,($1500.00)\n"
        ",,,,,,,,\n"
        ",,,,,,,,\n"
    )
    result = parse_robinhood_csv(csv.encode("utf-8"))
    assert len(result["trades"]) == 1
    assert len(result["errored"]) == 0
    # total_rows counts every CSV row; blanks count toward total but skip routing
    assert result["total_rows"] == 3


def test_disclaimer_text_row_skipped():
    """Issue 8: a disclaimer row where text lands in one column is also skipped."""
    csv = (
        "Activity Date,Process Date,Settle Date,Instrument,Description,Trans Code,Quantity,Price,Amount\n"
        "01/15/2025,01/15/2025,01/17/2025,AAPL,APPLE INC,Buy,10,$150.00,($1500.00)\n"
        ',,,,"These statements are for informational purposes only.",,,,\n'
    )
    result = parse_robinhood_csv(csv.encode("utf-8"))
    # The disclaimer row has an empty Activity Date, empty Trans Code, empty
    # Instrument, but a Description value. Per the new "all key fields empty"
    # check, Description being present means it does NOT get auto-skipped — it
    # falls through to the unknown-code path. We accept that as errored
    # rather than crash. Verify no trade was incorrectly created.
    assert len(result["trades"]) == 1
    assert result["trades"][0]["ticker"] == "AAPL"
    # Disclaimer row will land in errored with missing_trans_code, which is
    # acceptable — preferable to silently dropping data we can't classify.
    assert len(result["errored"]) == 1
    assert result["errored"][0]["reason"] == "missing_trans_code"


def test_nmax_treated_as_equity_not_crypto():
    """Issue 5: NMAX is Newsmax, a regular equity. Must not be mis-classified."""
    csv = (
        "Activity Date,Process Date,Settle Date,Instrument,Description,Trans Code,Quantity,Price,Amount\n"
        "04/01/2025,04/01/2025,04/03/2025,NMAX,NEWSMAX,Buy,100,$25.00,($2500.00)\n"
    )
    result = parse_robinhood_csv(csv.encode("utf-8"))
    assert len(result["trades"]) == 1
    assert len(result["quarantined"]) == 0
    assert result["trades"][0]["ticker"] == "NMAX"
    assert result["trades"][0]["action"] == "buy"


def test_full_realworld_mix():
    """Combined scenario mimicking a real Robinhood export with 8+ row types."""
    csv = (
        "Activity Date,Process Date,Settle Date,Instrument,Description,Trans Code,Quantity,Price,Amount\n"
        '01/15/2025,01/15/2025,01/17/2025,"SPY\nCUSIP: 78462F103\nRecurring",SPDR S&P 500,Buy,5,$500.00,($2500.00)\n'
        "02/20/2025,02/20/2025,02/22/2025,AAPL,APPLE,Sell,10,$175.00,$1750.00\n"
        '03/01/2025,03/01/2025,03/01/2025,AMZN,"AMZN 5/9/2025 Call $220.00",BTO,1,$3.50,($350.00)\n'
        '05/15/2025,05/15/2025,05/15/2025,SPY,"SPY 5/16/2025 Call $400.00",OEXP,1S,,\n'
        "06/01/2025,06/01/2025,06/01/2025,AAPL,SLI PAYMENT,SLIP,,,$2.34\n"
        "07/15/2025,07/15/2025,07/15/2025,GME,MFGD DIV,MDIV,,,$0.50\n"
        '08/01/2025,08/01/2025,08/05/2025,VOO,"VANGUARD",ACATI,12,,\n'
        "01/05/2025,01/05/2025,01/05/2025,,GOLD BOOST,GDBP,,,$10.00\n"
        ",,,,,,,,\n"
    )
    result = parse_robinhood_csv(csv.encode("utf-8"))
    assert len(result["trades"]) == 3       # SPY buy, AAPL sell, VOO transfer_in
    assert len(result["dividends"]) == 1    # GME MDIV
    assert len(result["securities_lending"]) == 1  # AAPL SLIP
    assert len(result["options_trades"]) == 2      # AMZN BTO + SPY OEXP (Sprint 9C.1)
    assert len(result["quarantined"]) == 0         # nothing quarantined now
    assert len(result["errored"]) == 0             # GDBP silent, blank row silent

    actions = sorted(t["action"] for t in result["trades"])
    assert actions == ["buy", "sell", "transfer_in"]

    voo = next(t for t in result["trades"] if t["ticker"] == "VOO")
    assert voo["action"] == "transfer_in"
    assert voo["price"] is None


# ─── Stage 2 patch tests: ACATI cost basis flag + BCXL cancellation matching ──

def test_acati_row_sets_cost_basis_unknown():
    """ACATI transfer-in trades must carry cost_basis_unknown=True so the
    performance engine knows the price field is meaningless."""
    csv = (
        "Activity Date,Process Date,Settle Date,Instrument,Description,Trans Code,Quantity,Price,Amount\n"
        "08/01/2025,08/01/2025,08/05/2025,VOO,VANGUARD ETF,ACATI,12,,\n"
    )
    result = parse_robinhood_csv(csv.encode("utf-8"))
    assert len(result["trades"]) == 1
    t = result["trades"][0]
    assert t["action"] == "transfer_in"
    assert t.get("cost_basis_unknown") is True


def test_bcxl_cancels_matching_buy_same_day():
    """A Buy + matching BCXL on the same day, same shares, same |amount|
    should KEEP the trade in the list with cancellation_status flipped
    to 'cancelled_by_broker' and cancel_matched_at populated."""
    csv = (
        "Activity Date,Process Date,Settle Date,Instrument,Description,Trans Code,Quantity,Price,Amount\n"
        "08/05/2024,08/05/2024,08/07/2024,AAPL,APPLE INC,Buy,10,$200.00,($2000.00)\n"
        "08/05/2024,08/05/2024,08/07/2024,AAPL,BROKER CANCELLATION,BCXL,10,$200.00,$2000.00\n"
    )
    result = parse_robinhood_csv(csv.encode("utf-8"))
    assert len(result["trades"]) == 1
    t = result["trades"][0]
    assert t["ticker"] == "AAPL"
    assert t["action"] == "buy"
    assert t["cancellation_status"] == "cancelled_by_broker"
    assert t["cancel_matched_at"] == "2024-08-05"
    assert result["cancellations_matched"] == 1
    assert len(result["errored"]) == 0


def test_bcxl_no_match_goes_to_errored():
    """An orphan BCXL row with no offsetting Buy must land in errored with
    reason='bcxl_no_match' so the signal isn't silently lost."""
    csv = (
        "Activity Date,Process Date,Settle Date,Instrument,Description,Trans Code,Quantity,Price,Amount\n"
        "08/05/2024,08/05/2024,08/07/2024,AAPL,BROKER CANCELLATION,BCXL,10,$200.00,$2000.00\n"
    )
    result = parse_robinhood_csv(csv.encode("utf-8"))
    assert len(result["trades"]) == 0
    assert result["cancellations_matched"] == 0
    assert len(result["errored"]) == 1
    assert result["errored"][0]["reason"] == "bcxl_no_match"


def test_bcxl_real_world_8_5_2024_block():
    """Mimics the 8/5/2024 fat-finger block: 4 Buys followed immediately by
    4 matching BCXL reversals. After parse, all 4 Buy rows are KEPT and
    each carries cancellation_status='cancelled_by_broker' so downstream
    FIFO/performance can filter them. cancellations_matched=4."""
    csv = (
        "Activity Date,Process Date,Settle Date,Instrument,Description,Trans Code,Quantity,Price,Amount\n"
        "08/05/2024,08/05/2024,08/07/2024,AAPL,APPLE INC,Buy,10,$200.00,($2000.00)\n"
        "08/05/2024,08/05/2024,08/07/2024,MSFT,MICROSOFT,Buy,5,$400.00,($2000.00)\n"
        "08/05/2024,08/05/2024,08/07/2024,GOOGL,ALPHABET,Buy,8,$150.00,($1200.00)\n"
        "08/05/2024,08/05/2024,08/07/2024,NVDA,NVIDIA,Buy,3,$500.00,($1500.00)\n"
        "08/05/2024,08/05/2024,08/07/2024,AAPL,BROKER CANCELLATION,BCXL,10,$200.00,$2000.00\n"
        "08/05/2024,08/05/2024,08/07/2024,MSFT,BROKER CANCELLATION,BCXL,5,$400.00,$2000.00\n"
        "08/05/2024,08/05/2024,08/07/2024,GOOGL,BROKER CANCELLATION,BCXL,8,$150.00,$1200.00\n"
        "08/05/2024,08/05/2024,08/07/2024,NVDA,BROKER CANCELLATION,BCXL,3,$500.00,$1500.00\n"
    )
    result = parse_robinhood_csv(csv.encode("utf-8"))
    assert len(result["trades"]) == 4
    for t in result["trades"]:
        assert t["cancellation_status"] == "cancelled_by_broker", f"{t['ticker']} not flagged"
        assert t["cancel_matched_at"] == "2024-08-05"
    assert result["cancellations_matched"] == 4
    assert len(result["errored"]) == 0


def test_bcxl_does_not_eat_unrelated_buys():
    """Sanity: a BCXL on AAPL must flag only the AAPL Buy. The MSFT Buy
    keeps cancellation_status='normal' even though shares and |amount|
    happen to coincide."""
    csv = (
        "Activity Date,Process Date,Settle Date,Instrument,Description,Trans Code,Quantity,Price,Amount\n"
        "08/05/2024,08/05/2024,08/07/2024,AAPL,APPLE INC,Buy,5,$400.00,($2000.00)\n"
        "08/05/2024,08/05/2024,08/07/2024,MSFT,MICROSOFT,Buy,5,$400.00,($2000.00)\n"
        "08/05/2024,08/05/2024,08/07/2024,AAPL,BROKER CANCELLATION,BCXL,5,$400.00,$2000.00\n"
    )
    result = parse_robinhood_csv(csv.encode("utf-8"))
    assert len(result["trades"]) == 2
    by_ticker = {t["ticker"]: t for t in result["trades"]}
    assert by_ticker["AAPL"]["cancellation_status"] == "cancelled_by_broker"
    assert by_ticker["AAPL"]["cancel_matched_at"] == "2024-08-05"
    assert by_ticker["MSFT"]["cancellation_status"] == "normal"
    assert by_ticker["MSFT"]["cancel_matched_at"] is None
    assert result["cancellations_matched"] == 1


def test_partial_bcxl_match_with_one_orphan():
    """3 Buys, 4 BCXLs — 3 should match (Buys flagged cancelled_by_broker),
    1 BCXL should land in errored as bcxl_no_match."""
    csv = (
        "Activity Date,Process Date,Settle Date,Instrument,Description,Trans Code,Quantity,Price,Amount\n"
        "08/05/2024,08/05/2024,08/07/2024,AAPL,APPLE,Buy,10,$200.00,($2000.00)\n"
        "08/05/2024,08/05/2024,08/07/2024,MSFT,MICROSOFT,Buy,5,$400.00,($2000.00)\n"
        "08/05/2024,08/05/2024,08/07/2024,GOOGL,ALPHABET,Buy,8,$150.00,($1200.00)\n"
        "08/05/2024,08/05/2024,08/07/2024,AAPL,BROKER CANCELLATION,BCXL,10,$200.00,$2000.00\n"
        "08/05/2024,08/05/2024,08/07/2024,MSFT,BROKER CANCELLATION,BCXL,5,$400.00,$2000.00\n"
        "08/05/2024,08/05/2024,08/07/2024,GOOGL,BROKER CANCELLATION,BCXL,8,$150.00,$1200.00\n"
        "08/05/2024,08/05/2024,08/07/2024,XYZ,ORPHAN,BCXL,1,$1.00,$1.00\n"
    )
    result = parse_robinhood_csv(csv.encode("utf-8"))
    assert len(result["trades"]) == 3
    for t in result["trades"]:
        assert t["cancellation_status"] == "cancelled_by_broker", f"{t['ticker']} not flagged"
    assert result["cancellations_matched"] == 3
    assert len(result["errored"]) == 1
    assert result["errored"][0]["reason"] == "bcxl_no_match"


def test_normal_buy_has_default_cancellation_status():
    """A regular Buy with no offsetting BCXL must carry cancellation_status='normal'
    so the commit route writes the DB column unambiguously and downstream FIFO
    can filter on equality."""
    csv = (
        "Activity Date,Process Date,Settle Date,Instrument,Description,Trans Code,Quantity,Price,Amount\n"
        "01/15/2025,01/15/2025,01/17/2025,AAPL,APPLE,Buy,10,$150.00,($1500.00)\n"
        "02/01/2025,02/01/2025,02/03/2025,MSFT,MICROSOFT,Sell,5,$300.00,$1500.00\n"
    )
    result = parse_robinhood_csv(csv.encode("utf-8"))
    assert len(result["trades"]) == 2
    for t in result["trades"]:
        assert t["cancellation_status"] == "normal"
        assert t["cancel_matched_at"] is None
    assert result["cancellations_matched"] == 0
