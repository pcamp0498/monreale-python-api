from fastapi import APIRouter, Depends, HTTPException, Query
from lib.auth import verify_api_key
import pandas as pd
import numpy as np
import urllib.request
import zipfile
import io
from typing import Optional

router = APIRouter()


def _fetch_ff_csv_lines(url: str) -> list:
    """Download and return raw CSV lines from a Ken French zip file."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        zip_data = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
        # Get first file (handles both .CSV and .csv extensions)
        names = z.namelist()
        if not names:
            raise ValueError("Empty zip file")
        csv_name = names[0]
        with z.open(csv_name) as f:
            content = f.read().decode("utf-8", errors="ignore")
    return content.strip().split("\n")


def get_ff5_factors(start_date: str = "2015-01-01") -> pd.DataFrame:
    """Robust parser for FF5 daily factors."""
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"

    try:
        lines = _fetch_ff_csv_lines(url)

        # Find the first line where the first field is an 8-digit date
        data_start = None
        data_end = None
        for i, line in enumerate(lines):
            parts = line.strip().split(",")
            if len(parts) >= 6:
                date_val = parts[0].strip()
                if len(date_val) == 8 and date_val.isdigit():
                    if data_start is None:
                        data_start = i
            elif data_start is not None and not line.strip():
                data_end = i
                break

        if data_start is None:
            raise ValueError("Could not find FF5 data section")
        if data_end is None:
            data_end = len(lines)

        records = []
        for line in lines[data_start:data_end]:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            try:
                date_str = parts[0].strip()
                if len(date_str) != 8 or not date_str.isdigit():
                    continue
                records.append({
                    "date": pd.to_datetime(date_str, format="%Y%m%d"),
                    "Mkt-RF": float(parts[1].strip()) / 100,
                    "SMB": float(parts[2].strip()) / 100,
                    "HML": float(parts[3].strip()) / 100,
                    "RMW": float(parts[4].strip()) / 100,
                    "CMA": float(parts[5].strip()) / 100,
                    "RF": float(parts[6].strip()) / 100 if len(parts) > 6 else 0.0,
                })
            except (ValueError, IndexError):
                continue

        if not records:
            raise ValueError("No valid FF5 records parsed")

        df = pd.DataFrame(records).set_index("date").sort_index()
        df = df[df.index >= pd.Timestamp(start_date)]
        return df

    except Exception as e:
        raise ValueError(f"Failed to fetch FF5 data: {str(e)}")


def get_momentum_factor(start_date: str = "2015-01-01") -> pd.DataFrame:
    """Robust parser for momentum factor."""
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"

    try:
        lines = _fetch_ff_csv_lines(url)

        data_start = None
        data_end = None
        for i, line in enumerate(lines):
            parts = line.strip().split(",")
            if len(parts) >= 2:
                date_val = parts[0].strip()
                if len(date_val) == 8 and date_val.isdigit():
                    if data_start is None:
                        data_start = i
            elif data_start is not None and not line.strip():
                data_end = i
                break

        if data_start is None:
            raise ValueError("Could not find momentum data section")
        if data_end is None:
            data_end = len(lines)

        records = []
        for line in lines[data_start:data_end]:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            try:
                date_str = parts[0].strip()
                if len(date_str) != 8 or not date_str.isdigit():
                    continue
                records.append({
                    "date": pd.to_datetime(date_str, format="%Y%m%d"),
                    "UMD": float(parts[1].strip()) / 100,
                })
            except (ValueError, IndexError):
                continue

        if not records:
            raise ValueError("No valid momentum records parsed")

        df = pd.DataFrame(records).set_index("date").sort_index()
        df = df[df.index >= pd.Timestamp(start_date)]
        return df

    except Exception as e:
        raise ValueError(f"Failed to fetch momentum data: {str(e)}")


def _sig_stars(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


@router.get("/ff5/{ticker}", dependencies=[Depends(verify_api_key)])
async def get_factor_loadings(
    ticker: str,
    period: str = Query("3y", pattern="^(1y|2y|3y|5y)$"),
):
    """Calculate Fama-French 5 Factor + Momentum loadings for a ticker."""
    try:
        from lib.polygon_client import get_price_history
        import statsmodels.api as sm
        from datetime import datetime, timedelta

        period_days = {"1y": 365, "2y": 730, "3y": 1095, "5y": 1825}
        days = period_days.get(period, 1095)

        # Always fetch at least 3 years of price data to overlap with FF5
        # (FF5 has 2-4 week publication lag from Ken French)
        fetch_days = max(days, 1095)
        results = get_price_history(ticker.upper(), days=fetch_days)
        if not results:
            raise HTTPException(status_code=400, detail=f"No price data for {ticker}")

        prices = pd.Series(
            {pd.Timestamp.fromtimestamp(r["t"] / 1000): r["c"] for r in results}
        )
        prices = prices.sort_index()
        # Normalize to date (drop time component) for clean alignment
        prices.index = prices.index.normalize()
        returns = prices.pct_change().dropna()
        returns.name = "stock_return"

        start_date = (datetime.now() - timedelta(days=fetch_days)).strftime("%Y-%m-%d")
        factors = get_ff5_factors(start_date)

        # Try to get momentum
        try:
            mom = get_momentum_factor(start_date)
            factors = factors.join(mom, how="left")
            has_momentum = True
        except Exception:
            has_momentum = False

        print(f"[ff5/{ticker}] returns: {len(returns)} days, {returns.index[0].date()} to {returns.index[-1].date()}")
        print(f"[ff5/{ticker}] factors: {len(factors)} days, {factors.index[0].date()} to {factors.index[-1].date()}")

        # Outer join + forward fill FF5 factors to bridge publication lag
        # Factor values change slowly — last known value is a valid proxy
        combined = pd.concat([returns, factors], axis=1, join="outer")
        factor_fill_cols = [c for c in factors.columns]
        combined[factor_fill_cols] = combined[factor_fill_cols].ffill()

        # Drop rows where stock return is missing (weekends, before listing)
        aligned = combined.dropna(subset=["stock_return"])
        # Drop rows where factors are still NaN (before FF5 history begins)
        aligned = aligned.dropna(subset=factor_fill_cols)

        print(f"[ff5/{ticker}] aligned rows: {len(aligned)}")

        if len(aligned) < 30:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient overlapping data: {len(aligned)} rows (need 30+). Returns: {len(returns)}, Factors: {len(factors)}",
            )

        excess_returns = aligned["stock_return"] - aligned["RF"]
        factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
        if has_momentum and "UMD" in aligned.columns:
            factor_cols.append("UMD")

        X = sm.add_constant(aligned[factor_cols])
        model = sm.OLS(excess_returns, X).fit()

        params = model.params
        tvalues = model.tvalues
        pvalues = model.pvalues

        def _interp(name: str, val: float) -> str:
            thresholds = {
                "Mkt-RF": ("High market sensitivity", "Low market sensitivity", "Moderate market sensitivity"),
                "SMB": ("Small-cap tilt", "Large-cap tilt", "Neutral size"),
                "HML": ("Value tilt", "Growth tilt", "Neutral value/growth"),
                "RMW": ("High profitability", "Low profitability", "Neutral profitability"),
                "CMA": ("Conservative investment", "Aggressive investment", "Neutral investment"),
                "UMD": ("Strong momentum", "Contrarian/reversal", "Neutral momentum"),
            }
            hi, lo, neut = thresholds.get(name, ("Positive", "Negative", "Neutral"))
            return hi if val > 0.2 else lo if val < -0.2 else neut

        factors_result = {}
        for name, display in [("const", "alpha"), ("Mkt-RF", "market_beta"), ("SMB", "smb"), ("HML", "hml"), ("RMW", "rmw"), ("CMA", "cma"), ("UMD", "momentum")]:
            if name not in params:
                continue
            val = float(params[name])
            t = float(tvalues[name])
            p = float(pvalues[name])
            entry = {
                "value": round(val, 6 if name == "const" else 4),
                "t_stat": round(t, 3),
                "significant": p < 0.05,
                "stars": _sig_stars(p),
            }
            if name == "const":
                entry["annualized"] = round(val * 252, 4)
            else:
                entry["interpretation"] = _interp(name, val)
            factors_result[display] = entry

        return {
            "ticker": ticker.upper(),
            "period": period,
            "observations": len(aligned),
            "r_squared": round(float(model.rsquared), 4),
            "r_squared_adj": round(float(model.rsquared_adj), 4),
            "factors": factors_result,
            "model": "FF5+Momentum" if has_momentum and "UMD" in factor_cols else "FF5",
            "data_start": str(aligned.index[0].date()),
            "data_end": str(aligned.index[-1].date()),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/portfolio-decomposition", dependencies=[Depends(verify_api_key)])
async def decompose_portfolio_factors(body: dict):
    """Decompose portfolio factor exposures (weighted sum of individual loadings)."""
    try:
        holdings = body.get("holdings", [])
        if not holdings:
            raise HTTPException(status_code=400, detail="Holdings required")

        factor_names = ["market_beta", "smb", "hml", "rmw", "cma", "momentum"]
        portfolio_factors = {f: 0.0 for f in factor_names}
        portfolio_alpha = 0.0
        results = []
        total_weight = sum(float(h.get("weight", 0)) for h in holdings if h.get("ticker"))

        for holding in holdings:
            ticker = holding.get("ticker", "")
            weight = float(holding.get("weight", 0)) / max(total_weight, 1)
            if not ticker or weight <= 0:
                continue

            try:
                factor_data = await get_factor_loadings(ticker, "3y")
                factors = factor_data["factors"]
                result = {"ticker": ticker, "weight": round(weight, 4)}

                for fname in factor_names:
                    if fname in factors:
                        loading = factors[fname]["value"]
                        portfolio_factors[fname] += weight * loading
                        result[fname] = loading

                alpha = factors.get("alpha", {}).get("annualized", 0)
                portfolio_alpha += weight * alpha
                result["alpha"] = alpha
                results.append(result)
            except Exception as e:
                results.append({"ticker": ticker, "weight": round(weight, 4), "error": str(e)})

        return {
            "portfolio_exposures": {f: round(v, 4) for f, v in portfolio_factors.items()},
            "portfolio_alpha_annualized": round(portfolio_alpha, 4),
            "holdings_analyzed": len([r for r in results if "error" not in r]),
            "holdings_failed": len([r for r in results if "error" in r]),
            "individual_loadings": results,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
