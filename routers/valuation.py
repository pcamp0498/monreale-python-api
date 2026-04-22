from fastapi import APIRouter, Depends, HTTPException
from lib.auth import verify_api_key
from pydantic import BaseModel
from typing import Optional
import math

router = APIRouter()


def clean_val(v):
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    except (TypeError, ValueError):
        return None


class GordonGrowthRequest(BaseModel):
    ticker: str
    dividend_per_share: Optional[float] = None
    growth_rate: Optional[float] = None
    required_return: Optional[float] = None
    use_sustainable_growth: bool = True


class TwoStageDDMRequest(BaseModel):
    ticker: str
    high_growth_rate: float
    high_growth_years: int = 5
    terminal_growth_rate: float = 0.03
    required_return: float = 0.10
    current_dividend: Optional[float] = None


@router.post("/gordon-growth", dependencies=[Depends(verify_api_key)])
async def gordon_growth_model(request: GordonGrowthRequest):
    """Gordon (Constant) Growth DDM: V = D1 / (r - g)"""
    try:
        from lib.polygon_client import get_snapshot, get_dividends, get_parsed_financials

        # Get current price from Polygon
        snap = get_snapshot(request.ticker)
        current_price = snap.get("price")

        # Get dividend data from Polygon
        D0 = request.dividend_per_share
        if D0 is None:
            divs = get_dividends(request.ticker, limit=4)
            # Sum last 4 quarterly dividends for annual rate
            D0 = sum(float(d.get("cash_amount", 0)) for d in divs[:4]) if divs else 0.0

        r = request.required_return
        if r is None:
            rf = 0.053
            erp = 0.055
            r = rf + 1.0 * erp  # assume beta=1 (Polygon doesn't provide beta directly)

        g = request.growth_rate
        if g is None and request.use_sustainable_growth:
            # Estimate sustainable growth from Polygon financials
            fins = get_parsed_financials(request.ticker, limit=2)
            if fins:
                latest = fins[0]
                ni = latest.get("net_income") or 0
                eq = latest.get("equity") or 1
                roe = ni / eq if eq != 0 else 0.10
                divs_paid = abs(latest.get("dividends_paid") or 0)
                payout = divs_paid / ni if ni > 0 else 0.40
                retention = max(0, 1 - payout)
                g = retention * roe
                g = min(max(g, 0.01), 0.15)
            else:
                g = 0.03

        g = g or 0.03
        info = {}  # compatibility for later references

        if D0 <= 0:
            return {
                "ticker": request.ticker.upper(),
                "model": "Gordon Growth DDM",
                "applicable": False,
                "reason": "Company does not pay dividends. Consider FCFE model instead.",
                "inputs": {"D0": D0, "r": clean_val(r), "g": clean_val(g)},
            }

        if r <= g:
            return {
                "ticker": request.ticker.upper(),
                "model": "Gordon Growth DDM",
                "applicable": False,
                "reason": f"Required return ({r:.1%}) must exceed growth rate ({g:.1%})",
                "inputs": {"D0": D0, "r": clean_val(r), "g": clean_val(g)},
            }

        D1 = D0 * (1 + g)
        intrinsic_value = D1 / (r - g)

        premium_discount = None
        upside_downside = None
        if current_price and current_price > 0:
            premium_discount = (intrinsic_value - current_price) / current_price
            upside_downside = "undervalued" if intrinsic_value > current_price else "overvalued"

        # Sensitivity analysis
        sensitivity = []
        for g_delta in [-0.02, -0.01, 0, 0.01, 0.02]:
            row = []
            for r_delta in [-0.02, -0.01, 0, 0.01, 0.02]:
                g_s = g + g_delta
                r_s = r + r_delta
                if r_s > g_s and r_s > 0:
                    row.append(round(D1 / (r_s - g_s), 2))
                else:
                    row.append(None)
            sensitivity.append({"g": round(g + g_delta, 4), "values": row})

        r_labels = [round(r + d, 4) for d in [-0.02, -0.01, 0, 0.01, 0.02]]

        # Compute payout and PE from Polygon financials
        fins_for_pe = get_parsed_financials(request.ticker, limit=1) if not locals().get("fins") else fins
        _latest_fin = fins_for_pe[0] if fins_for_pe else {}
        _ni = _latest_fin.get("net_income") or 0
        _divs_paid_abs = abs(_latest_fin.get("dividends_paid") or 0)
        payout = _divs_paid_abs / _ni if _ni > 0 else 0.40
        justified_pe = payout / (r - g) if r > g else None
        _eps = _latest_fin.get("eps_diluted")
        actual_pe = (current_price / _eps) if (current_price and _eps and _eps > 0) else None

        return {
            "ticker": request.ticker.upper(),
            "model": "Gordon Growth DDM",
            "applicable": True,
            "inputs": {
                "D0": clean_val(D0),
                "D1": clean_val(D1),
                "growth_rate_g": clean_val(g),
                "required_return_r": clean_val(r),
                "sustainable_growth_used": request.use_sustainable_growth,
                "roe": clean_val((_latest_fin.get("net_income") or 0) / (_latest_fin.get("equity") or 1)),
                "retention_rate_b": clean_val(1 - payout),
            },
            "output": {
                "intrinsic_value": clean_val(intrinsic_value),
                "current_price": clean_val(current_price),
                "premium_discount_pct": clean_val(premium_discount * 100 if premium_discount else None),
                "assessment": upside_downside,
                "justified_pe": clean_val(justified_pe),
                "actual_pe": clean_val(actual_pe),
            },
            "sensitivity": {
                "r_labels": r_labels,
                "rows": sensitivity,
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/two-stage-ddm", dependencies=[Depends(verify_api_key)])
async def two_stage_ddm(request: TwoStageDDMRequest):
    """Two-Stage DDM: high growth for n years, then terminal value."""
    try:
        from lib.polygon_client import get_snapshot, get_dividends

        snap = get_snapshot(request.ticker)
        current_price = snap.get("price")

        D0 = request.current_dividend
        if D0 is None:
            divs = get_dividends(request.ticker, limit=4)
            D0 = sum(float(d.get("cash_amount", 0)) for d in divs[:4]) if divs else 0.0
        r = request.required_return
        gs = request.high_growth_rate
        gl = request.terminal_growth_rate
        n = request.high_growth_years

        if D0 <= 0:
            return {
                "ticker": request.ticker.upper(),
                "model": "Two-Stage DDM",
                "applicable": False,
                "reason": "No dividend. Cannot apply DDM.",
            }

        if r <= gl:
            raise HTTPException(status_code=400, detail="Required return must exceed terminal growth rate")

        pv_stage1 = 0
        dividends = []
        for t in range(1, n + 1):
            Dt = D0 * (1 + gs) ** t
            pv = Dt / (1 + r) ** t
            pv_stage1 += pv
            dividends.append({"year": t, "dividend": round(Dt, 4), "pv": round(pv, 4)})

        Dn_plus1 = D0 * (1 + gs) ** n * (1 + gl)
        terminal_value = Dn_plus1 / (r - gl)
        pv_terminal = terminal_value / (1 + r) ** n
        intrinsic_value = pv_stage1 + pv_terminal

        premium_discount = None
        if current_price and current_price > 0:
            premium_discount = (intrinsic_value - current_price) / current_price

        return {
            "ticker": request.ticker.upper(),
            "model": "Two-Stage DDM",
            "applicable": True,
            "inputs": {
                "D0": clean_val(D0),
                "high_growth_rate": gs,
                "high_growth_years": n,
                "terminal_growth_rate": gl,
                "required_return": r,
            },
            "output": {
                "pv_stage1_dividends": clean_val(pv_stage1),
                "terminal_value": clean_val(terminal_value),
                "pv_terminal_value": clean_val(pv_terminal),
                "intrinsic_value": clean_val(intrinsic_value),
                "current_price": clean_val(current_price),
                "premium_discount_pct": clean_val(premium_discount * 100 if premium_discount else None),
                "assessment": ("undervalued" if premium_discount and premium_discount > 0 else "overvalued") if premium_discount else None,
                "stage1_pct_of_value": clean_val(pv_stage1 / intrinsic_value * 100 if intrinsic_value else None),
                "stage2_pct_of_value": clean_val(pv_terminal / intrinsic_value * 100 if intrinsic_value else None),
            },
            "dividend_schedule": dividends,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dol-dfl/{ticker}", dependencies=[Depends(verify_api_key)])
async def get_operating_financial_leverage(ticker: str):
    """DOL = %dOI / %dRev, DFL = %dNI / %dOI, DTL = DOL x DFL"""
    try:
        results = []

        # Try SimFin first (more reliable on Railway)
        try:
            import simfin as sf
            import os

            sf.set_api_key(os.environ.get("SIMFIN_API_KEY", "free"))
            sf.set_data_dir("/tmp/simfin_data")

            income = sf.load_income(variant="annual", market="us")
            if income is not None and not income.empty and ticker.upper() in income.index.get_level_values(0):
                df = income.loc[ticker.upper()].copy().sort_index(ascending=False).head(4)
                for i in range(len(df) - 1):
                    curr, prev = df.iloc[i], df.iloc[i + 1]
                    try:
                        rev_c = float(curr.get("Revenue", 0) or 0)
                        rev_p = float(prev.get("Revenue", 0) or 0)
                        op_c = float(curr.get("Operating Income (Loss)", 0) or curr.get("Operating Income", 0) or 0)
                        op_p = float(prev.get("Operating Income (Loss)", 0) or prev.get("Operating Income", 0) or 0)
                        net_c = float(curr.get("Net Income", 0) or 0)
                        net_p = float(prev.get("Net Income", 0) or 0)
                        if rev_p == 0 or op_p == 0:
                            continue
                        pct_rev = (rev_c - rev_p) / abs(rev_p)
                        pct_op = (op_c - op_p) / abs(op_p)
                        pct_net = (net_c - net_p) / abs(net_p) if net_p != 0 else None
                        dol = pct_op / pct_rev if pct_rev != 0 else None
                        dfl = pct_net / pct_op if (pct_op != 0 and pct_net is not None) else None
                        dtl = dol * dfl if (dol and dfl) else None
                        results.append({
                            "period": str(df.index[i]),
                            "revenue_growth": clean_val(pct_rev * 100),
                            "operating_income_growth": clean_val(pct_op * 100),
                            "net_income_growth": clean_val(pct_net * 100 if pct_net else None),
                            "dol": clean_val(dol), "dfl": clean_val(dfl), "dtl": clean_val(dtl),
                        })
                    except (KeyError, TypeError, ZeroDivisionError):
                        continue
        except Exception:
            pass

        # Fallback: try Polygon financials
        if not results:
            try:
                from lib.polygon_client import get_parsed_financials
                fins = get_parsed_financials(ticker, timeframe="annual", limit=4)
                for i in range(len(fins) - 1):
                    curr, prev = fins[i], fins[i + 1]
                    try:
                        rev_c = float(curr.get("revenue") or 0)
                        rev_p = float(prev.get("revenue") or 0)
                        op_c = float(curr.get("operating_income") or 0)
                        op_p = float(prev.get("operating_income") or 0)
                        net_c = float(curr.get("net_income") or 0)
                        net_p = float(prev.get("net_income") or 0)
                        if rev_p == 0 or op_p == 0:
                            continue
                        pct_rev = (rev_c - rev_p) / abs(rev_p)
                        pct_op = (op_c - op_p) / abs(op_p)
                        pct_net = (net_c - net_p) / abs(net_p) if net_p != 0 else None
                        dol = pct_op / pct_rev if pct_rev != 0 else None
                        dfl = pct_net / pct_op if (pct_op != 0 and pct_net is not None) else None
                        dtl = dol * dfl if (dol and dfl) else None
                        results.append({
                            "period": curr.get("period", ""),
                            "revenue_growth": clean_val(pct_rev * 100),
                            "operating_income_growth": clean_val(pct_op * 100),
                            "net_income_growth": clean_val(pct_net * 100 if pct_net else None),
                            "dol": clean_val(dol), "dfl": clean_val(dfl), "dtl": clean_val(dtl),
                        })
                    except (KeyError, TypeError, ZeroDivisionError):
                        continue
            except Exception:
                pass

        latest = results[0] if results else {}

        def interpret_dol(dol):
            if dol is None:
                return "Insufficient data"
            if dol > 5:
                return "Very high operating leverage — profits very sensitive to revenue"
            if dol > 3:
                return "High operating leverage — significant fixed cost base"
            if dol > 1:
                return "Moderate operating leverage"
            return "Low operating leverage — mostly variable costs"

        return {
            "ticker": ticker.upper(),
            "historical": results,
            "latest": {
                "dol": latest.get("dol"),
                "dfl": latest.get("dfl"),
                "dtl": latest.get("dtl"),
                "dol_interpretation": interpret_dol(latest.get("dol")),
            },
            "data_available": len(results) > 0,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
