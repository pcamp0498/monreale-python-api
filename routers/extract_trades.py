"""Trade-history extraction endpoints (Sprint 9A).

POST /extract/robinhood-csv accepts raw CSV bytes and returns parsed
trades/dividends/quarantined/errored/securities_lending and the
cancellations_matched counter. NO trade data is logged anywhere on
the server side beyond the response — the caller is responsible for
persistence.
"""
from fastapi import APIRouter, Depends, HTTPException, Request

from lib.auth import verify_api_key
from lib.robinhood_parser import parse_robinhood_csv
from lib.split_adjuster import fetch_splits, adjust_trade_for_splits

router = APIRouter()


def build_parse_response(result: dict) -> dict:
    """Map a parse_robinhood_csv() return dict to the HTTP response shape.

    Pulled out as a pure function so unit tests can assert the response
    contract without spinning up a TestClient (and pulling in every other
    router's heavyweight import).
    """
    start, end = result.get("date_range", (None, None))
    securities_lending = result.get("securities_lending") or []
    cancellations_matched = int(result.get("cancellations_matched") or 0)

    return {
        "trades": result.get("trades") or [],
        "dividends": result.get("dividends") or [],
        "securities_lending": securities_lending,
        "quarantined": result.get("quarantined") or [],
        "errored": result.get("errored") or [],
        "cancellations_matched": cancellations_matched,
        "date_range": {"start": start, "end": end},
        "total_rows": int(result.get("total_rows") or 0),
        "counts": {
            "trades": len(result.get("trades") or []),
            "dividends": len(result.get("dividends") or []),
            "securities_lending": len(securities_lending),
            "quarantined": len(result.get("quarantined") or []),
            "errored": len(result.get("errored") or []),
            "cancellations_matched": cancellations_matched,
        },
    }


@router.post("/robinhood-csv", dependencies=[Depends(verify_api_key)])
async def extract_robinhood_csv(request: Request):
    """Parse a Robinhood Activity CSV. Body: raw bytes."""
    try:
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="Empty body — send the CSV bytes as the request body")

        result = parse_robinhood_csv(body)
        return build_parse_response(result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/split-adjust", dependencies=[Depends(verify_api_key)])
async def split_adjust_trades(body: dict):
    """Apply Polygon split adjustments to a list of trades.

    Body: {trades: [...]} where each trade has ticker, executed_at, shares, price.
    Returns: {trades: [...adjusted], adjusted_count: N, splits_by_ticker: {...}}.
    """
    try:
        trades = body.get("trades") or []
        if not isinstance(trades, list):
            raise HTTPException(status_code=400, detail="trades must be a list")

        # Cache splits per unique ticker so we hit Polygon once per symbol
        unique_tickers = {t.get("ticker") for t in trades if t.get("ticker")}
        splits_by_ticker: dict[str, list[dict]] = {}
        for tk in unique_tickers:
            splits_by_ticker[tk] = fetch_splits(tk)

        adjusted: list[dict] = []
        adjusted_count = 0
        for t in trades:
            tk = t.get("ticker")
            new_t = adjust_trade_for_splits(t, splits_by_ticker.get(tk, []))
            if new_t.get("is_split_adjusted"):
                adjusted_count += 1
            adjusted.append(new_t)

        return {
            "trades": adjusted,
            "adjusted_count": adjusted_count,
            "splits_by_ticker": {k: len(v) for k, v in splits_by_ticker.items()},
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
