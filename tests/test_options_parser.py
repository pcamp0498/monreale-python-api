"""Unit tests for lib/options_parser.

Run: python -m pytest tests/test_options_parser.py -v
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.options_parser import parse_option_description, parse_option_quantity


def test_parse_description_call():
    out = parse_option_description("NKE 1/9/2026 Call $68.00")
    assert out == {
        "ticker": "NKE",
        "expiration_date": date(2026, 1, 9),
        "strike": 68.00,
        "option_type": "call",
    }


def test_parse_description_put():
    out = parse_option_description("AAPL 12/19/2025 Put $200.00")
    assert out == {
        "ticker": "AAPL",
        "expiration_date": date(2025, 12, 19),
        "strike": 200.00,
        "option_type": "put",
    }


def test_parse_description_with_comma_in_strike():
    """Some strikes formatted with thousand separator: '$1,250.00'."""
    out = parse_option_description("NVDA 6/20/2025 Call $1,250.00")
    assert out is not None
    assert out["strike"] == 1250.00
    assert out["ticker"] == "NVDA"


def test_parse_description_multiline():
    """Robinhood occasionally writes Description with embedded newlines.
    Parser must collapse whitespace before regex match."""
    desc = "NMAX 1/16/2026\nCall $40.00"
    out = parse_option_description(desc)
    assert out is not None
    assert out["ticker"] == "NMAX"
    assert out["expiration_date"] == date(2026, 1, 16)
    assert out["strike"] == 40.00
    assert out["option_type"] == "call"


def test_parse_description_invalid_format():
    """Random text returns None — caller routes to quarantine."""
    assert parse_option_description("AAPL stock dividend") is None
    assert parse_option_description("") is None
    assert parse_option_description(None) is None
    assert parse_option_description("NVDA $100") is None  # missing date and type


def test_parse_description_strike_no_dollar_sign():
    """If the $ is missing, parsing should fail (signals not-an-option)."""
    out = parse_option_description("NKE 1/9/2026 Call 68.00")
    assert out is None


def test_parse_quantity_integer():
    assert parse_option_quantity("1") == 1.0
    assert parse_option_quantity("5") == 5.0
    assert parse_option_quantity("100") == 100.0


def test_parse_quantity_with_S_suffix():
    """OEXP rows have '1S' meaning 'settled'. Parser strips before parse."""
    assert parse_option_quantity("1S") == 1.0
    assert parse_option_quantity("5S") == 5.0
    # Lowercase too — defensive
    assert parse_option_quantity("3s") == 3.0


def test_parse_quantity_decimal():
    assert parse_option_quantity("0.5") == 0.5
    assert parse_option_quantity("2.5") == 2.5


def test_parse_quantity_invalid():
    """Empty or unparseable returns None."""
    assert parse_option_quantity("") is None
    assert parse_option_quantity(None) is None
    assert parse_option_quantity("abc") is None


def test_parse_description_brk_b():
    """BRK.B (period in ticker) — Robinhood uses dots, not slashes."""
    out = parse_option_description("BRK.B 12/19/2025 Put $400.00")
    assert out is not None
    assert out["ticker"] == "BRK.B"
    assert out["strike"] == 400.00
