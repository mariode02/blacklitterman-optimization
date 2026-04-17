from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.pre_selection import DropCorrelated
from skfolio.prior import BlackLitterman, EmpiricalPrior
from skfolio.moments import EquilibriumMu, LedoitWolf
from sklearn.pipeline import Pipeline
import pandas as pd


_TRADING_DAYS = 252


def build_pipeline(
    view_strings: list[str],
    market_cap_weights: pd.Series | None = None,
    objective: ObjectiveFunction = ObjectiveFunction.MAXIMIZE_RATIO,
    risk_free_rate: float = 0.04,
    correlation_threshold: float = 0.90,
    min_weight: float = 0.0,
    max_weight: float = 0.40,
    risk_aversion: float = 1.0,
    tau: float = 0.05,
) -> Pipeline:
    """Build the full optimization pipeline.

    Args:
        view_strings: BL absolute view strings, e.g. ["AAPL == 0.08"].
        market_cap_weights: Market-cap weights for EquilibriumMu prior.
        objective: Optimization objective (MAXIMIZE_RATIO by default).
        risk_free_rate: Annual risk-free rate for Sharpe computation.
        correlation_threshold: Assets with pairwise correlation above this
            are dropped by DropCorrelated (keeps the one with higher return).
        min_weight: Minimum allocation per asset.
        max_weight: Maximum allocation per asset (concentration limit).
        risk_aversion: Lambda parameter for EquilibriumMu.
        tau: Scalar expressing uncertainty in the equilibrium prior.

    Returns:
        Unfitted sklearn Pipeline ready to call .fit(returns).
    """
    prior = BlackLitterman(
        views=view_strings,
        prior_estimator=EmpiricalPrior(
            mu_estimator=EquilibriumMu(
                risk_aversion=risk_aversion,
                weights=market_cap_weights,
            ),
            covariance_estimator=LedoitWolf(),
        ),
        tau=tau,
    )

    daily_rf = (1 + risk_free_rate) ** (1 / _TRADING_DAYS) - 1

    optimizer = MeanRisk(
        objective_function=objective,
        prior_estimator=prior,
        risk_free_rate=daily_rf,
        min_weights=min_weight,
        max_weights=max_weight,
    )

    pipeline = Pipeline([
        ("pre", DropCorrelated(threshold=correlation_threshold)),
        ("opt", optimizer),
    ])
    pipeline.set_output(transform="pandas")
    return pipeline
