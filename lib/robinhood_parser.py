"""Robinhood Activity CSV parser.

Quarantines options + crypto BEFORE attempting equity parse so a bad row
never aborts a batch. Trans-code mapping is hardcoded; unknown codes
land in `errored` with the original code preserved.

Patched 2026-04-27 to handle real-world Robinhood CSV variance:
- Multi-line Instrument cells (e.g. "SPY\\nCUSIP: 78462F103\\nRecurring")
- Option open/close codes BTO/STC/STO/BTC + OEXP
- Option detection via Description regex when Instrument lacks the
  standard option contract format
- Securities lending income (SLIP) routed to its own bucket
- Manufactured dividends (MDIV) routed to dividends with type='manufactured'
- ACAT transfers in (ACATI) routed to trades with action='transfer_in'
- New silent-ignore codes: GDBP, RTP, CONV, MINT, BCXL
- Quantity field with trailing 'S' (option contracts)
- Trailing blank rows and disclaimer text rows
"""
from __future__ import annotations

import csv
import io
import re
from datetime import datetime
from typing import Any

# ─── Detection regexes ───────────────────────────────────────────────────────
# Standard OCC-style option contract embedded in Instrument: "AAPL 240119C00150000"
OPTION_RE = re.compile(r"^[A-Z]+\s+\d{6}[CP]\d+$")
# Robinhood's Description for an option leg: "AMZN 5/9/2025 Call $220.00"
OPTIONS_DESCRIPTION_REGEX = re.compile(r"(Call|Put)\s+\$\d")

CRYPTO_TICKERS = {
    "BTC", "ETH", "DOGE", "LTC", "BCH", "ETC", "BSV", "LINK",
    "AAVE", "MATIC", "COMP", "SHIB", "UNI", "AVAX", "XLM", "XTZ",
}

# ─── Trans code routing ──────────────────────────────────────────────────────
# Equity buys/sells ONLY — BTO/STC/STO/BTC moved to OPTION_TRANS_CODES because
# they are option opens/closes, not equity.
EQUITY_BUY_CODES = {"Buy"}
EQUITY_SELL_CODES = {"Sell"}
OPTION_TRANS_CODES = {"BTO", "STO", "BTC", "STC", "OEXP"}

# Dividends: cash (CDIV/DIV), tax-withheld (DIVTAX/DTAX), manufactured (MDIV)
DIVIDEND_CODES = {"CDIV", "DIV", "DIVTAX", "DTAX"}
TAX_WITHHELD_CODES = {"DIVTAX", "DTAX"}
MANUFACTURED_DIV_CODES = {"MDIV"}

# Stock-action codes
SPLIT_CODES = {"SPL", "SPR"}

# Securities lending income (SLIP) — separate bucket
SECURITIES_LENDING_CODES = {"SLIP"}

# ACAT transfers in
TRANSFER_IN_CODES = {"ACATI"}

# Broker cancellations — captured during main loop, post-processed to match
# and remove the offsetting Buy from the trades list.
BROKER_CANCELLATION_CODES = {"BCXL"}

# Silent ignore: cash transfers, interest, broker bonuses, account conversions.
# (BCXL is NOT in this set — it's handled separately in post-processing.)
IGNORE_CODES = {
    "ACH", "WIRE", "JNL", "INT", "CINT",      # original: bank/cash/interest
    "GOLD", "MARGIN",                          # original: Gold subscription, margin
    "GDBP",                                     # Gold Deposit Boost Payment
    "RTP",                                      # Real-time payment
    "CONV",                                     # Apex→RHS legacy account conversion
    "MINT",                                     # Margin interest
}


# ─── Field normalizers ───────────────────────────────────────────────────────
def normalize_instrument(raw: str | None) -> str:
    """Return the ticker only — strip CUSIP/Recurring/etc. annotations.

    Robinhood embeds extra lines in the Instrument cell:
        "SPY\\nCUSIP: 78462F103\\nRecurring"  →  "SPY"
    Take only the first non-empty line, strip whitespace, uppercase nothing
    here (caller decides) — but for matching we work in upper.
    """
    if not raw:
        return ""
    # Take first line; tolerate \r\n, \n, \r
    first = re.split(r"[\r\n]+", str(raw), maxsplit=1)[0]
    return first.strip()


def parse_amount(s: str | None) -> float | None:
    """Parse Robinhood amount strings: '$1,234.56' or '($1,234.56)' for negatives.

    Empty string / None / unparseable → None.
    """
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1]
    s = s.replace("$", "").replace(",", "").strip()
    if not s:
        return None
    try:
        v = float(s)
        return -v if negative else v
    except ValueError:
        return None


def parse_quantity(s: str | None) -> float | None:
    """Quantity field. Robinhood appends 'S' on OEXP option rows ('1S', '5S').

    Strip the trailing S before numeric parse. Empty → None.
    """
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    if s.endswith("S") or s.endswith("s"):
        s = s[:-1].strip()
    if not s:
        return None
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def parse_date(s: str | None) -> str | None:
    """Parse Robinhood Activity Date 'MM/DD/YYYY' → ISO 'YYYY-MM-DD'."""
    if not s:
        return None
    s = str(s).strip()
    if not s:
        return None
    try:
        return datetime.strptime(s, "%m/%d/%Y").strftime("%Y-%m-%d")
    except ValueError:
        return None


def _is_blank_row(row: dict) -> bool:
    """True if every cell is empty/whitespace. Catches trailing CSV junk and
    multi-cell disclaimer text where the disclaimer occupies a single cell
    (most cells empty) — but only blanks are skipped here; disclaimer text
    in a single cell that has no Activity Date / Trans Code / Instrument
    will fall through to the all-empty-key-fields check below.
    """
    for v in row.values():
        if v is None:
            continue
        if str(v).strip():
            return False
    return True


def _quarantine_record(row: dict, reason: str, asset_type: str, symbol: str) -> dict:
    return {
        "raw_row": row,
        "quarantine_reason": reason,
        "detected_asset_type": asset_type,
        "ticker_or_symbol": symbol,
        "executed_at": parse_date(row.get("Activity Date")),
    }


def parse_robinhood_csv(file_bytes: bytes) -> dict[str, Any]:
    """Parse a Robinhood Activity CSV byte stream.

    Returns a dict with keys:
        trades              — equity/ETF trades (action: buy/sell/split/transfer_in)
        dividends           — dividend records (type: cash, tax_withheld, manufactured)
        securities_lending  — SLIP rows (Stock Lending Income Payment)
        quarantined         — options + crypto rows held for future sprints
        errored             — unknown trans codes
        date_range          — (min, max) ISO strings or (None, None)
        total_rows          — total CSV rows the reader saw
    """
    text = file_bytes.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))

    trades: list[dict] = []
    dividends: list[dict] = []
    securities_lending: list[dict] = []
    quarantined: list[dict] = []
    errored: list[dict] = []
    bcxl_rows: list[dict] = []  # captured for post-processing match against Buy rows
    dates: list[str] = []
    total_rows = 0

    for row in reader:
        total_rows += 1

        # 0. Skip completely blank rows (trailing CSV junk)
        if _is_blank_row(row):
            continue

        # Normalize Instrument (strip CUSIP / Recurring / etc.)
        instrument_raw = row.get("Instrument") or ""
        instrument = normalize_instrument(instrument_raw)
        description = (row.get("Description") or "").strip()
        trans_code = (row.get("Trans Code") or "").strip()
        activity_date = (row.get("Activity Date") or "").strip()

        # Defensive: if every key field is blank, skip silently (disclaimer rows
        # whose first column is "These statements are for…" still fall through
        # because that text lands in Activity Date — handled by the fact that
        # parse_date returns None and trans_code/instrument are empty; below).
        if not activity_date and not trans_code and not instrument and not description:
            continue

        # 1. Quarantine options BEFORE any other routing.
        # Two trigger paths:
        #   a) Instrument matches OCC-style option contract regex
        #   b) Description matches "Call|Put $<digit>" (covers BTO/STC/STO/BTC
        #      where Instrument is the underlying ticker, not the contract)
        #   c) Trans code is an explicit option code (BTO/STO/BTC/STC/OEXP)
        is_option = False
        if instrument and OPTION_RE.match(instrument):
            is_option = True
        elif description and OPTIONS_DESCRIPTION_REGEX.search(description):
            is_option = True
        elif trans_code in OPTION_TRANS_CODES:
            is_option = True

        if is_option:
            # OEXP rows have Quantity like '1S' — parse_quantity handles it.
            # We don't need shares for quarantine, but make sure the row doesn't
            # raise. _quarantine_record only reads Activity Date.
            symbol = instrument or description[:80]
            quarantined.append(_quarantine_record(row, "options", "option", symbol))
            continue

        # 2. Crypto quarantine
        if instrument and instrument.upper() in CRYPTO_TICKERS:
            quarantined.append(_quarantine_record(row, "crypto", "crypto", instrument.upper()))
            continue

        # 3a. Capture broker cancellations for post-processing match.
        if trans_code in BROKER_CANCELLATION_CODES:
            bcxl_rows.append(row)
            continue

        # 3b. Silent ignore for non-trade cash events (broker bonuses, transfers,
        #     interest, account conversions).
        if trans_code in IGNORE_CODES:
            continue

        executed_date = parse_date(activity_date)
        if executed_date:
            dates.append(executed_date)

        # 4. Route by trans code
        if trans_code in EQUITY_BUY_CODES or trans_code in EQUITY_SELL_CODES:
            action = "buy" if trans_code in EQUITY_BUY_CODES else "sell"
            trades.append({
                "action": action,
                "ticker": instrument.upper(),
                "shares": parse_quantity(row.get("Quantity")),
                "price": parse_amount(row.get("Price")),
                "amount": parse_amount(row.get("Amount")),
                "fees": 0.0,
                "executed_at": executed_date,
                "settled_at": parse_date(row.get("Settle Date")),
                "asset_type": "equity",
                "cancellation_status": "normal",
                "cancel_matched_at": None,
                "raw_row": row,
            })

        elif trans_code in TRANSFER_IN_CODES:
            trades.append({
                "action": "transfer_in",
                "ticker": instrument.upper() if instrument else "",
                "shares": parse_quantity(row.get("Quantity")),
                "price": parse_amount(row.get("Price")),  # may be None
                "amount": parse_amount(row.get("Amount")),  # may be None
                "fees": 0.0,
                "executed_at": executed_date,
                "settled_at": parse_date(row.get("Settle Date")),
                "asset_type": "equity",
                "cost_basis_unknown": True,
                "cancellation_status": "normal",
                "cancel_matched_at": None,
                "raw_row": row,
            })

        elif trans_code in DIVIDEND_CODES:
            div_type = "tax_withheld" if trans_code in TAX_WITHHELD_CODES else "cash"
            dividends.append({
                "ticker": instrument.upper(),
                "amount": parse_amount(row.get("Amount")),
                "paid_at": executed_date,
                "dividend_type": div_type,
                "raw_row": row,
            })

        elif trans_code in MANUFACTURED_DIV_CODES:
            dividends.append({
                "ticker": instrument.upper(),
                "amount": parse_amount(row.get("Amount")),
                "paid_at": executed_date,
                "dividend_type": "manufactured",
                "raw_row": row,
            })

        elif trans_code in SECURITIES_LENDING_CODES:
            securities_lending.append({
                "ticker": instrument.upper() if instrument else "",
                "amount": parse_amount(row.get("Amount")),
                "paid_at": executed_date,
                "raw_row": row,
            })

        elif trans_code in SPLIT_CODES:
            trades.append({
                "action": "split",
                "ticker": instrument.upper(),
                "shares": parse_quantity(row.get("Quantity")),
                "price": None,
                "amount": None,
                "fees": 0.0,
                "executed_at": executed_date,
                "settled_at": None,
                "asset_type": "equity",
                "cancellation_status": "normal",
                "cancel_matched_at": None,
                "raw_row": row,
            })

        else:
            errored.append({
                "raw_row": row,
                "reason": f"unknown_trans_code:{trans_code}" if trans_code else "missing_trans_code",
            })

    # ─── BCXL post-processing ────────────────────────────────────────────────
    # Match each broker-cancellation row against an offsetting Buy in the
    # trades list using same-day + ticker + shares + |amount|. Matched Buy
    # rows are KEPT in the trades list with cancellation_status flipped to
    # 'cancelled_by_broker' and cancel_matched_at set to the BCXL row's
    # executed_at — preserves the audit trail. Downstream FIFO/performance
    # logic is responsible for filtering on cancellation_status='normal'.
    # Unmatched BCXL rows go to errored so the signal isn't lost.
    cancellations_matched = 0
    matched_trade_indices: set[int] = set()
    for bcxl in bcxl_rows:
        bcxl_date = parse_date(bcxl.get("Activity Date"))
        bcxl_instrument = normalize_instrument(bcxl.get("Instrument") or "").upper()
        bcxl_qty = parse_quantity(bcxl.get("Quantity"))
        bcxl_amount = parse_amount(bcxl.get("Amount"))
        bcxl_amount_abs = abs(bcxl_amount) if bcxl_amount is not None else None

        matched_idx: int | None = None
        for i, t in enumerate(trades):
            if i in matched_trade_indices:
                continue
            if t.get("action") != "buy":
                continue
            if (t.get("ticker") or "").upper() != bcxl_instrument:
                continue
            if t.get("executed_at") != bcxl_date:
                continue
            t_shares = t.get("shares")
            if t_shares is None or bcxl_qty is None:
                continue
            if abs(float(t_shares) - float(bcxl_qty)) > 0.0001:
                continue
            t_amount = t.get("amount")
            if t_amount is not None and bcxl_amount_abs is not None:
                if abs(abs(float(t_amount)) - float(bcxl_amount_abs)) > 0.01:
                    continue
            matched_idx = i
            break

        if matched_idx is not None:
            matched_trade_indices.add(matched_idx)
            trades[matched_idx]["cancellation_status"] = "cancelled_by_broker"
            trades[matched_idx]["cancel_matched_at"] = bcxl_date
            cancellations_matched += 1
        else:
            errored.append({
                "raw_row": bcxl,
                "reason": "bcxl_no_match",
            })

    date_range = (min(dates), max(dates)) if dates else (None, None)

    return {
        "trades": trades,
        "dividends": dividends,
        "securities_lending": securities_lending,
        "quarantined": quarantined,
        "errored": errored,
        "cancellations_matched": cancellations_matched,
        "date_range": date_range,
        "total_rows": total_rows,
    }
