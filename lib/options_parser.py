"""Robinhood options-row parsing.

The activity CSV's `Description` field encodes the contract spec:
    "NKE 1/9/2026 Call $68.00"
    "NMAX 1/16/2026 Call $40.00"
    "BRK.B 12/19/2025 Put $400.00"

`Quantity` is integer contract count for normal trades but carries an "S"
suffix on OEXP rows ("1S" — "S" likely meaning "settled" in Robinhood's
internal code; we strip and parse the leading number).

`Price` and `Amount` reuse parse_amount() from robinhood_parser — same
"$X.XX" / "($X.XX)" format as equities. OEXP rows have empty Price + Amount.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Optional


# Compiled once. Tolerates:
#   - leading prefix text ("Option Expiration for NMAX ...") via re.search
#   - multi-line whitespace via pre-normalization (collapse \s+ → " ")
#   - punctuation in ticker (BRK.B), commas in strike ($1,250.00)
# The boundary `(?<!\S)` before the ticker prevents matching mid-word.
_OPTION_DESC_RE = re.compile(
    r"(?<!\S)([A-Z][A-Z0-9.]*)\s+(\d+/\d+/\d{4})\s+(Call|Put)\s+\$([0-9,]+\.?\d*)(?!\S)",
    re.IGNORECASE,
)


def parse_option_description(desc: str | None) -> Optional[dict]:
    """Parse a Robinhood option description into structured fields.

    Returns dict with `ticker`, `expiration_date` (date), `strike` (float),
    `option_type` ("call"/"put"). Returns None on parse failure — caller
    should treat that as a quarantine signal.

    Handles multiple Description shapes:
        "NKE 1/9/2026 Call $68.00"                              (BTO/STC/STO/BTC)
        "Option Expiration for NMAX 1/16/2026 Call $40.00"      (OEXP)
        "NMAX 1/16/2026\nCall $40.00"                           (multi-line variant)

    Whitespace is collapsed pre-match; the `(?<!\S)...(?!\S)` boundaries
    on the regex prevent picking up mid-word matches.
    """
    if not desc:
        return None
    # Collapse any internal whitespace runs (incl. newlines) to single spaces.
    normalized = re.sub(r"\s+", " ", desc.strip())

    m = _OPTION_DESC_RE.search(normalized)
    if not m:
        return None

    ticker_raw, exp_raw, opt_type, strike_raw = m.group(1), m.group(2), m.group(3), m.group(4)

    # Expiration: "M/D/YYYY" → date(YYYY, M, D)
    try:
        month, day, year = exp_raw.split("/")
        expiration = date(int(year), int(month), int(day))
    except (ValueError, IndexError):
        return None

    # Strike: strip commas before float conversion ("$1,250.00" → 1250.00)
    try:
        strike = float(strike_raw.replace(",", ""))
    except ValueError:
        return None

    return {
        "ticker": ticker_raw.upper(),
        "expiration_date": expiration,
        "strike": strike,
        "option_type": opt_type.lower(),
    }


def parse_option_quantity(qty: str | None) -> Optional[float]:
    """Parse Robinhood option Quantity, handling the OEXP "S" suffix.

    Examples:
        "1"   -> 1.0
        "2"   -> 2.0
        "1S"  -> 1.0   (OEXP rows; "S" = settled)
        "0.5" -> 0.5
        ""    -> None
    """
    if qty is None:
        return None
    s = str(qty).strip()
    if not s:
        return None
    # Strip trailing "S" suffix (case-insensitive). Robinhood's OEXP rows show
    # "1S", "2S" etc — meaning the contract was settled at expiration.
    if s.upper().endswith("S"):
        s = s[:-1].strip()
    try:
        return float(s)
    except ValueError:
        return None
