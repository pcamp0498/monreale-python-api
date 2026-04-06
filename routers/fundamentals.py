from fastapi import APIRouter, Depends, HTTPException, Query
from lib.auth import verify_api_key

router = APIRouter()


@router.get("/{ticker}", dependencies=[Depends(verify_api_key)])
async def get_financial_statements(
    ticker: str,
    period: str = Query("annual", pattern="^(annual|quarterly)$"),
):
    """Get standardized financial statements via SimFin."""
    try:
        import simfin as sf
        import os

        sf.set_api_key(os.getenv("SIMFIN_API_KEY", "free"))
        sf.set_data_dir("/tmp/simfin_data")

        # Income statement
        try:
            income = sf.load_income(
                variant=period, market="us", ticker=ticker.upper()
            )
            income_data = (
                income.tail(8).to_dict("records") if income is not None else []
            )
        except Exception:
            income_data = []

        # Balance sheet
        try:
            balance = sf.load_balance(
                variant=period, market="us", ticker=ticker.upper()
            )
            balance_data = (
                balance.tail(8).to_dict("records") if balance is not None else []
            )
        except Exception:
            balance_data = []

        # Cash flow
        try:
            cashflow = sf.load_cashflow(
                variant=period, market="us", ticker=ticker.upper()
            )
            cashflow_data = (
                cashflow.tail(8).to_dict("records")
                if cashflow is not None
                else []
            )
        except Exception:
            cashflow_data = []

        return {
            "ticker": ticker.upper(),
            "period": period,
            "income_statement": income_data,
            "balance_sheet": balance_data,
            "cash_flow": cashflow_data,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
