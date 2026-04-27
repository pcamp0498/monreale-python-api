"""Robinhood Activity CSV parser.

Quarantines options + crypto BEFORE attempting equity parse so a bad row
never aborts a batch. Trans-code mapping is hardcoded; unknown codes
land in `errored` with the original code preserved.
"""
from __future__ import annotations

import csv
import io
import re
from datetime import datetime
from typing import Any

# Quarantine detection
OPTION_RE = re.compile(r"^[A-Z]+\s+\d{6}[CP]\d+$")
CRYPTO_TICKERS = {
    "BTC", "ETH", "DOGE", "LTC", "BCH", "ETC", "BSV", "LINK",
    "AAVE", "MATIC", "COMP", "SHIB", "UNI", "AVAX", "XLM", "XTZ",
}

# Trans code mapping — anything not in these sets goes to `errored`
BUY_CODES = {"Buy", "BTO"}
SELL_CODES = {"Sell", "STC"}
DIVIDEND_CODES = {"CDIV", "DIV", "DIVTAX", "DTAX"}
SPLIT_CODES = {"SPL", "SPR"}
IGNORE_CODES = {"ACH", "WIRE", "JNL", "INT", "CINT", "GOLD", "MARGIN"}


def parse_amount(s: str | None) -> float | None:
    """Parse Robinhood amount strings: '$1,234.56' or '($1,234.56)' for negatives."""
    if not s:
        return None
    s = str(s).strip()
    if not s:
        return None
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1]
    s = s.replace("$", "").replace(",", "").strip()
    try:
        v = float(s)
        return -v if negative else v
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
        trades        — list of equity/ETF trade dicts (action in buy/sell/split)
        dividends     — list of dividend records (cash and tax-withheld)
        quarantined   — list of options + crypto rows with reason
        errored       — list of unrecognized trans codes with reason
        date_range    — tuple (min_date, max_date) ISO strings or (None, None)
        total_rows    — total rows seen by the CSV reader
    """
    # utf-8-sig strips a UTF-8 BOM if present without raising on its absence
    text = file_bytes.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))

    trades: list[dict] = []
    dividends: list[dict] = []
    quarantined: list[dict] = []
    errored: list[dict] = []
    dates: list[str] = []
    total_rows = 0

    for row in reader:
        total_rows += 1
        instrument = (row.get("Instrument") or "").strip()
        trans_code = (row.get("Trans Code") or "").strip()
        activity_date = (row.get("Activity Date") or "").strip()

        # 1. Quarantine BEFORE equity parse — bad rows never leak through
        if instrument and OPTION_RE.match(instrument):
            quarantined.append(_quarantine_record(row, "options", "option", instrument))
            continue
        if instrument and instrument.upper() in CRYPTO_TICKERS:
            quarantined.append(_quarantine_record(row, "crypto", "crypto", instrument.upper()))
            continue

        # 2. Silent ignore for non-trade cash events
        if trans_code in IGNORE_CODES:
            continue

        executed_date = parse_date(activity_date)
        if executed_date:
            dates.append(executed_date)

        # 3. Map trans code to typed record
        if trans_code in BUY_CODES or trans_code in SELL_CODES:
            action = "buy" if trans_code in BUY_CODES else "sell"
            trades.append({
                "action": action,
                "ticker": instrument.upper(),
                "shares": parse_amount(row.get("Quantity")),
                "price": parse_amount(row.get("Price")),
                "amount": parse_amount(row.get("Amount")),
                "fees": 0.0,
                "executed_at": executed_date,
                "settled_at": parse_date(row.get("Settle Date")),
                "asset_type": "equity",
                "raw_row": row,
            })
        elif trans_code in DIVIDEND_CODES:
            div_type = "tax_withheld" if trans_code in {"DIVTAX", "DTAX"} else "cash"
            dividends.append({
                "ticker": instrument.upper(),
                "amount": parse_amount(row.get("Amount")),
                "paid_at": executed_date,
                "dividend_type": div_type,
                "raw_row": row,
            })
        elif trans_code in SPLIT_CODES:
            trades.append({
                "action": "split",
                "ticker": instrument.upper(),
                "shares": parse_amount(row.get("Quantity")),
                "price": None,
                "amount": None,
                "fees": 0.0,
                "executed_at": executed_date,
                "settled_at": None,
                "asset_type": "equity",
                "raw_row": row,
            })
        else:
            errored.append({
                "raw_row": row,
                "reason": f"unknown_trans_code:{trans_code}" if trans_code else "missing_trans_code",
            })

    date_range = (min(dates), max(dates)) if dates else (None, None)

    return {
        "trades": trades,
        "dividends": dividends,
        "quarantined": quarantined,
        "errored": errored,
        "date_range": date_range,
        "total_rows": total_rows,
    }
