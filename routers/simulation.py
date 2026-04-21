from fastapi import APIRouter, Depends, HTTPException
from lib.auth import verify_api_key
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pandas as pd

router = APIRouter()


class MonteCarloRequest(BaseModel):
    weights: List[float]
    tickers: List[str]
    initial_value: float = 100000
    periods: int = 252
    simulations: int = 1000
    confidence_interval: float = 0.95
    risk_free_rate: float = 0.053


class StressTestRequest(BaseModel):
    weights: List[float]
    tickers: List[str]
    initial_value: float = 100000
    scenarios: Optional[dict] = None


class UtilityOptRequest(BaseModel):
    tickers: List[str]
    risk_aversion: float = 5.0
    risk_free_rate: float = 0.053
    target_return: Optional[float] = None
    max_position: float = 0.40


@router.post("/monte-carlo", dependencies=[Depends(verify_api_key)])
async def run_monte_carlo(request: MonteCarloRequest):
    """Run Monte Carlo simulation for portfolio."""
    try:
        from lib.polygon_client import get_prices_dataframe

        if len(request.weights) != len(request.tickers):
            raise HTTPException(
                status_code=400,
                detail="weights and tickers must have same length",
            )

        weights = np.array(request.weights)
        weights = weights / weights.sum()

        prices = get_prices_dataframe(request.tickers, days=756)

        if prices.empty:
            raise HTTPException(status_code=400, detail="Could not fetch price data")

        returns = prices.pct_change().dropna()

        available_tickers = [t for t in request.tickers if t in returns.columns]
        if not available_tickers:
            raise HTTPException(
                status_code=400, detail="No tickers with price data found"
            )

        returns = returns[available_tickers]
        weights = weights[: len(available_tickers)]
        weights = weights / weights.sum()

        mean_returns = returns.mean()
        cov_matrix = returns.cov()

        port_mean = np.dot(weights, mean_returns)
        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        np.random.seed(42)
        simulations = request.simulations
        periods = request.periods

        # Cholesky decomposition for correlated returns
        L = np.linalg.cholesky(
            cov_matrix + np.eye(len(available_tickers)) * 1e-10
        )

        portfolio_paths = np.zeros((simulations, periods + 1))
        portfolio_paths[:, 0] = request.initial_value

        for t in range(1, periods + 1):
            Z = np.random.standard_normal((len(available_tickers), simulations))
            corr_returns = mean_returns.values.reshape(-1, 1) + L @ Z
            port_returns = np.dot(weights, corr_returns)
            portfolio_paths[:, t] = portfolio_paths[:, t - 1] * (1 + port_returns)

        final_values = portfolio_paths[:, -1]

        # Percentile paths for fan chart
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        step = max(1, periods // 50)
        time_axis = list(range(0, periods + 1, step))
        percentile_paths = {}
        for p in percentiles:
            percentile_paths[str(p)] = [
                float(np.percentile(portfolio_paths[:, t], p)) for t in time_axis
            ]

        return {
            "initial_value": request.initial_value,
            "periods": periods,
            "simulations": simulations,
            "tickers": available_tickers,
            "weights": weights.tolist(),
            "statistics": {
                "median_final": round(float(np.median(final_values)), 2),
                "mean_final": round(float(np.mean(final_values)), 2),
                "best_case": round(float(np.percentile(final_values, 95)), 2),
                "worst_case": round(float(np.percentile(final_values, 5)), 2),
                "var_95": round(
                    float(np.percentile(final_values, 5) - request.initial_value), 2
                ),
                "cvar_95": round(
                    float(
                        np.mean(
                            final_values[
                                final_values <= np.percentile(final_values, 5)
                            ]
                        )
                        - request.initial_value
                    ),
                    2,
                ),
                "prob_of_loss": round(
                    float(np.mean(final_values < request.initial_value)), 4
                ),
                "prob_double": round(
                    float(np.mean(final_values >= request.initial_value * 2)), 4
                ),
                "expected_return": round(float(port_mean * 252), 4),
                "expected_volatility": round(float(port_std * np.sqrt(252)), 4),
                "sharpe_ratio": round(
                    float(
                        (port_mean * 252 - request.risk_free_rate)
                        / (port_std * np.sqrt(252))
                    )
                    if port_std > 0
                    else 0,
                    4,
                ),
            },
            "percentile_paths": percentile_paths,
            "time_axis": time_axis,
            "histogram": {
                "values": [
                    round(float(v), 2)
                    for v in np.percentile(
                        final_values, range(0, 101, 2)
                    ).tolist()
                ],
                "bins": list(range(0, 101, 2)),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stress-test", dependencies=[Depends(verify_api_key)])
async def run_stress_test(request: StressTestRequest):
    """Run portfolio stress tests against historical and custom scenarios."""
    try:
        from lib.polygon_client import get_prices_dataframe

        weights = np.array(request.weights)
        weights = weights / weights.sum()

        prices = get_prices_dataframe(request.tickers, days=2000)
        returns = prices.pct_change().dropna()

        available = [t for t in request.tickers if t in returns.columns]
        returns = returns[available]
        weights = weights[: len(available)]
        weights = weights / weights.sum()

        def portfolio_return(rets, w):
            return float(np.dot(w, rets))

        # Historical scenarios
        historical_scenarios = {}

        scenario_defs = [
            ("covid_crash_2020", "COVID Crash (Feb-Mar 2020)", "2020-02-19", "2020-03-23"),
            ("gfc_2008", "Global Financial Crisis (2008-09)", "2008-09-15", "2009-03-09"),
            ("rate_shock_2022", "Rate Shock (2022)", "2022-01-03", "2022-12-31"),
        ]

        for key, name, start, end in scenario_defs:
            try:
                mask = (returns.index >= start) & (returns.index <= end)
                period_rets = returns.loc[mask]
                if len(period_rets) > 0:
                    cumret = (1 + period_rets).prod() - 1
                    historical_scenarios[key] = {
                        "name": name,
                        "period": f"{start} to {end}",
                        "portfolio_return": round(portfolio_return(cumret, weights), 4),
                        "worst_position": str(cumret.idxmin()),
                        "worst_position_return": round(float(cumret.min()), 4),
                    }
            except Exception:
                continue

        # Hypothetical scenarios
        hypothetical_scenarios = {
            "recession_mild": {
                "name": "Mild Recession (-15% equity)",
                "description": "GDP -1%, rates +50bps, spreads +150bps",
                "equity_shock": -0.15,
                "bond_shock": -0.03,
                "portfolio_impact": round(
                    sum(weights[i] * -0.15 for i in range(len(available))), 4
                ),
            },
            "recession_severe": {
                "name": "Severe Recession (-35% equity)",
                "description": "GDP -3%, rates +200bps, spreads +500bps",
                "equity_shock": -0.35,
                "bond_shock": -0.08,
                "portfolio_impact": round(
                    sum(weights[i] * -0.35 for i in range(len(available))), 4
                ),
            },
            "inflation_surge": {
                "name": "Inflation Surge",
                "description": "CPI +4%, Fed hikes 300bps, yields spike",
                "equity_shock": -0.20,
                "bond_shock": -0.15,
                "portfolio_impact": round(
                    sum(weights[i] * -0.20 for i in range(len(available))), 4
                ),
            },
            "tech_crash": {
                "name": "Tech Sector Crash (-40%)",
                "description": "Valuation compression, rate sensitivity",
                "equity_shock": -0.40,
                "bond_shock": 0.05,
                "portfolio_impact": round(
                    sum(weights[i] * -0.40 for i in range(len(available))), 4
                ),
            },
            "soft_landing": {
                "name": "Soft Landing (Bull Case)",
                "description": "Rates normalize, earnings grow 10%",
                "equity_shock": 0.20,
                "bond_shock": 0.05,
                "portfolio_impact": round(
                    sum(weights[i] * 0.20 for i in range(len(available))), 4
                ),
            },
        }

        for scenario in hypothetical_scenarios.values():
            scenario["dollar_impact"] = round(
                request.initial_value * scenario["portfolio_impact"], 2
            )
            scenario["final_value"] = round(
                request.initial_value * (1 + scenario["portfolio_impact"]), 2
            )

        for scenario in historical_scenarios.values():
            scenario["dollar_impact"] = round(
                request.initial_value * scenario["portfolio_return"], 2
            )
            scenario["final_value"] = round(
                request.initial_value * (1 + scenario["portfolio_return"]), 2
            )

        return {
            "initial_value": request.initial_value,
            "tickers": available,
            "weights": weights.tolist(),
            "historical_scenarios": historical_scenarios,
            "hypothetical_scenarios": hypothetical_scenarios,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/utility-optimization", dependencies=[Depends(verify_api_key)])
async def optimize_with_utility(request: UtilityOptRequest):
    """
    Mean-variance optimization using investor utility function.
    U = E(R) - 0.5 * A * sigma^2
    """
    try:
        from lib.polygon_client import get_prices_dataframe
        from pypfopt import EfficientFrontier, risk_models, expected_returns
        import scipy.optimize as sco

        prices = get_prices_dataframe(request.tickers, days=756)
        returns = prices.pct_change().dropna()

        available = [t for t in request.tickers if t in returns.columns]
        returns = returns[available]

        mu = expected_returns.mean_historical_return(returns)
        S = risk_models.sample_cov(returns)

        n = len(available)
        A = request.risk_aversion

        def neg_utility(w):
            w = np.array(w)
            port_return = np.dot(w, mu)
            port_variance = np.dot(w.T, np.dot(S, w))
            return -(port_return - 0.5 * A * port_variance)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0, request.max_position)] * n
        w0 = np.array([1 / n] * n)

        result = sco.minimize(
            neg_utility,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        if not result.success:
            optimal_weights = w0
        else:
            optimal_weights = result.x
            optimal_weights = np.maximum(optimal_weights, 0)
            optimal_weights = optimal_weights / optimal_weights.sum()

        port_return = float(np.dot(optimal_weights, mu))
        port_variance = float(np.dot(optimal_weights.T, np.dot(S, optimal_weights)))
        port_std = np.sqrt(port_variance)
        utility = port_return - 0.5 * A * port_variance
        sharpe = (
            (port_return - request.risk_free_rate) / port_std if port_std > 0 else 0
        )

        # Efficient frontier points
        frontier_points = []
        try:
            frontier_risks = np.linspace(port_std * 0.5, port_std * 2.5, 20)
            for target_std in frontier_risks:
                try:
                    ef_temp = EfficientFrontier(
                        mu, S, weight_bounds=(0, request.max_position)
                    )
                    ef_temp.efficient_risk(float(target_std))
                    perf = ef_temp.portfolio_performance(verbose=False)
                    frontier_points.append(
                        {
                            "risk": round(float(perf[1]), 4),
                            "return": round(float(perf[0]), 4),
                            "sharpe": round(float(perf[2]), 4),
                        }
                    )
                except Exception:
                    continue
        except Exception:
            pass

        # Capital Market Line
        cml_points = []
        for risk in np.linspace(0, port_std * 2, 20):
            ret = request.risk_free_rate + sharpe * risk
            cml_points.append(
                {"risk": round(float(risk), 4), "return": round(float(ret), 4)}
            )

        # Indifference curves
        indifference_curves = []
        for utility_level in [utility - 0.05, utility, utility + 0.05]:
            curve_points = []
            for sigma in np.linspace(0.05, 0.60, 30):
                req_return = utility_level + 0.5 * A * sigma**2
                if 0 <= req_return <= 1:
                    curve_points.append(
                        {
                            "risk": round(float(sigma), 4),
                            "return": round(float(req_return), 4),
                        }
                    )
            if curve_points:
                indifference_curves.append(
                    {"utility": round(float(utility_level), 4), "points": curve_points}
                )

        return {
            "tickers": available,
            "risk_aversion": A,
            "optimal_weights": {
                available[i]: round(float(optimal_weights[i]), 4)
                for i in range(len(available))
            },
            "portfolio_metrics": {
                "expected_return": round(port_return, 4),
                "volatility": round(port_std, 4),
                "utility": round(float(utility), 4),
                "sharpe_ratio": round(float(sharpe), 4),
            },
            "efficient_frontier": frontier_points,
            "capital_market_line": cml_points,
            "indifference_curves": indifference_curves,
            "optimal_point": {"risk": round(port_std, 4), "return": round(port_return, 4)},
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
