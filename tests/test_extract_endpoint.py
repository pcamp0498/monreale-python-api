"""Endpoint-level test for /extract/robinhood-csv.

Mounts ONLY the extract_trades router on a minimal FastAPI app so the test
doesn't pull in the rest of the codebase's heavyweight imports. Auth is
short-circuited via dependency_overrides.

Run with:  python -m pytest tests/test_extract_endpoint.py -v
"""
import sys
import pytest
from pathlib import Path

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Skip the whole module if FastAPI + httpx aren't available (Railway has them;
# bare local Pythons may not). Existing parser tests still cover the behavior.
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.extract_trades import router, build_parse_response
from lib.auth import verify_api_key
from lib.robinhood_parser import parse_robinhood_csv


@pytest.fixture(scope="module")
def client():
    app = FastAPI()
    # Override auth so we don't need MICROSERVICE_API_KEY for tests
    app.dependency_overrides[verify_api_key] = lambda: "test-key"
    app.include_router(router, prefix="/extract")
    return TestClient(app)


# CSV fixture: 1 NVDA Buy + matching BCXL + 1 QCOM ACATI + 1 AAPL SLIP
SAMPLE_CSV = (
    "Activity Date,Process Date,Settle Date,Instrument,Description,Trans Code,Quantity,Price,Amount\n"
    "08/05/2024,08/05/2024,08/07/2024,NVDA,NVIDIA,Buy,10,$200.00,($2000.00)\n"
    "08/05/2024,08/05/2024,08/07/2024,NVDA,BROKER CANCELLATION,BCXL,10,$200.00,$2000.00\n"
    "07/30/2024,07/30/2024,08/01/2024,QCOM,QUALCOMM,ACATI,15,,\n"
    "06/01/2025,06/01/2025,06/01/2025,AAPL,SLI PAYMENT,SLIP,,,$2.34\n"
)


def test_extract_endpoint_returns_securities_lending_and_cancellations_matched(client):
    """End-to-end: POST a CSV with one SLIP row and one BCXL pair (plus an
    ACATI row for breadth). Confirm the HTTP response includes both new
    fields at the top level — not just in the parser's internal dict."""
    resp = client.post(
        "/extract/robinhood-csv",
        content=SAMPLE_CSV.encode("utf-8"),
        headers={"Content-Type": "text/csv"},
    )
    assert resp.status_code == 200, f"got {resp.status_code}: {resp.text}"
    body = resp.json()

    # Top-level fields the user reported as missing
    assert "securities_lending" in body, "securities_lending missing from response"
    assert "cancellations_matched" in body, "cancellations_matched missing from response"

    # Counts sub-object should expose both too
    assert "counts" in body
    assert "securities_lending" in body["counts"]
    assert "cancellations_matched" in body["counts"]

    # Functional content
    assert isinstance(body["securities_lending"], list)
    assert len(body["securities_lending"]) == 1
    slip = body["securities_lending"][0]
    assert slip["ticker"] == "AAPL"
    assert abs(float(slip["amount"]) - 2.34) < 0.001
    assert slip["paid_at"] == "2025-06-01"

    assert body["cancellations_matched"] == 1
    assert body["counts"]["securities_lending"] == 1
    assert body["counts"]["cancellations_matched"] == 1

    # NVDA Buy is preserved with the cancellation flag (KEEP semantics)
    nvda = next((t for t in body["trades"] if t["ticker"] == "NVDA"), None)
    assert nvda is not None
    assert nvda["cancellation_status"] == "cancelled_by_broker"
    assert nvda["cancel_matched_at"] == "2024-08-05"

    # QCOM ACATI is preserved with cost_basis_unknown
    qcom = next((t for t in body["trades"] if t["ticker"] == "QCOM"), None)
    assert qcom is not None
    assert qcom["action"] == "transfer_in"
    assert qcom["cost_basis_unknown"] is True

    # No phantom errored or quarantined rows
    assert body["counts"]["errored"] == 0
    assert body["counts"]["quarantined"] == 0


def test_build_parse_response_helper_includes_new_fields():
    """Pure-function test of the response builder — runs even on Python
    environments without httpx (which TestClient requires)."""
    parsed = parse_robinhood_csv(SAMPLE_CSV.encode("utf-8"))
    response = build_parse_response(parsed)

    assert "securities_lending" in response
    assert "cancellations_matched" in response
    assert response["cancellations_matched"] == 1
    assert len(response["securities_lending"]) == 1
    assert response["counts"]["securities_lending"] == 1
    assert response["counts"]["cancellations_matched"] == 1
    # Existing fields preserved
    for key in ("trades", "dividends", "quarantined", "errored", "date_range", "total_rows"):
        assert key in response, f"{key} missing from response payload"


def test_empty_body_returns_400(client):
    resp = client.post(
        "/extract/robinhood-csv",
        content=b"",
        headers={"Content-Type": "text/csv"},
    )
    assert resp.status_code == 400
