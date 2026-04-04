from fastapi import APIRouter, Depends, HTTPException
from lib.auth import verify_api_key
from pydantic import BaseModel
from typing import List, Optional
from datetime import date, datetime, timedelta

router = APIRouter()


class TaxLotInput(BaseModel):
    ticker: str
    shares: float
    cost_basis_per_share: float
    acquisition_date: str
    current_price: float
    account_type: Optional[str] = "taxable"


class TaxCalculationRequest(BaseModel):
    tax_lots: List[TaxLotInput]
    federal_rate: float = 0.37
    state_rate: float = 0.0
    ltcg_rate: float = 0.20


@router.post("/calculate", dependencies=[Depends(verify_api_key)])
async def calculate_tax_impact(request: TaxCalculationRequest):
    try:
        today = date.today()
        results = []
        total_st_gain = 0.0
        total_lt_gain = 0.0
        total_harvestable = 0.0

        for lot in request.tax_lots:
            acq = datetime.strptime(lot.acquisition_date, "%Y-%m-%d").date()
            hold_days = (today - acq).days
            is_lt = hold_days >= 365
            days_to_lt = max(0, 365 - hold_days)
            wash_safe = (today + timedelta(days=30)).isoformat()

            unrealized = (lot.current_price - lot.cost_basis_per_share) * lot.shares
            tax_rate = (request.ltcg_rate if is_lt else request.federal_rate) + request.state_rate
            est_tax = max(0, unrealized * tax_rate)

            if unrealized > 0:
                if is_lt:
                    total_lt_gain += unrealized
                else:
                    total_st_gain += unrealized
            else:
                total_harvestable += abs(unrealized)

            results.append({
                "ticker": lot.ticker,
                "shares": lot.shares,
                "cost_basis": lot.cost_basis_per_share,
                "current_price": lot.current_price,
                "unrealized_gain": unrealized,
                "unrealized_gain_pct": (unrealized / (lot.cost_basis_per_share * lot.shares)) * 100 if lot.cost_basis_per_share * lot.shares > 0 else 0,
                "holding_days": hold_days,
                "is_long_term": is_lt,
                "days_to_long_term": days_to_lt,
                "tax_type": "LONG-TERM" if is_lt else "SHORT-TERM",
                "tax_rate": tax_rate,
                "estimated_tax_if_sold": est_tax,
                "wash_sale_safe_date": wash_safe,
                "is_harvestable_loss": unrealized < 0,
                "potential_tax_saving": abs(unrealized) * tax_rate if unrealized < 0 else 0,
            })

        net_gain = total_st_gain + total_lt_gain - total_harvestable
        est_federal = (total_st_gain * request.federal_rate) + (total_lt_gain * request.ltcg_rate)
        est_state = net_gain * request.state_rate

        return {
            "positions": results,
            "summary": {
                "total_short_term_gain": total_st_gain,
                "total_long_term_gain": total_lt_gain,
                "total_harvestable_loss": total_harvestable,
                "net_taxable_gain": net_gain,
                "estimated_federal_tax": est_federal,
                "estimated_state_tax": est_state,
                "estimated_total_tax": est_federal + est_state,
            },
            "disclaimer": "Analytical estimate only. Not tax advice. Consult your tax advisor.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
