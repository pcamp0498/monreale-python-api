"""Trade-history extraction endpoints (Sprint 9A).

POST /extract/robinhood-csv accepts raw CSV bytes and returns parsed
trades/dividends/quarantined/errored. NO trade data is logged anywhere
on the server side beyond the response — the caller is responsible for
persistence.
"""
from fastapi import APIRouter, Depends, HTTPException, Request

from lib.auth import verify_api_key
from lib.robinhood_parser import parse_robinhood_csv
from lib.split_adjuster import fetch_splits, adjust_trade_for_splits

router = APIRouter()


@router.post("/robinhood-csv", dependencies=[Depends(verify_api_key)])
async def extract_robinhood_csv(request: Request):
    """Parse a Robinhood Activity CSV. Body: raw bytes."""
    try:
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="Empty body — send the CSV bytes as the request body")

        result = parse_robinhood_csv(body)
        start, end = result["date_range"]

        return {
            "trades": result["trades"],
            "dividends": result["dividends"],
            "quarantined": result["quarantined"],
            "errored": result["errored"],
            "date_range": {"start": start, "end": end},
            "total_rows": result["total_rows"],
            "counts": {
                "trades": len(result["trades"]),
                "dividends": len(result["dividends"]),
                "quarantined": len(result["quarantined"]),
                "errored": len(result["errored"]),
            },
        }
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
