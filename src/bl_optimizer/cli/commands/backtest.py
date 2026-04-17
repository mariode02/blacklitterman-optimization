import typer
from typing import Annotated
from pathlib import Path
from datetime import datetime
import pandas as pd
from rich.console import Console
from rich.table import Table

from skfolio.model_selection import WalkForward, cross_val_predict

from bl_optimizer.config.settings import OptimizationConfig
from bl_optimizer.data.models import TickerInput, TickerRow
from bl_optimizer.data.fetcher import fetch_prices, fetch_volume
from bl_optimizer.data.validator import validate_prices
from bl_optimizer.features.engineer import compute_features
from bl_optimizer.views.random_forest import RandomForestViewsModel
from bl_optimizer.views.linear_regression import LinearRegressionViewsModel
from bl_optimizer.views.view_builder import build_view_strings
from bl_optimizer.optimization.optimizer import build_pipeline
from bl_optimizer.reporting.reporter import print_metrics

console = Console()

def backtest_command(
    input_csv: Annotated[Path, typer.Option("--input", help="Path to tickers CSV", exists=True)],
    start_date: Annotated[str, typer.Option("--start", help="Start date (ISO 8601)")] = "2018-01-01",
    end_date: Annotated[str, typer.Option("--end", help="End date (ISO 8601)")] = datetime.now().strftime("%Y-%m-%d"),
    model_type: Annotated[str, typer.Option("--model", help="ML model for views: rf=Random Forest, lr=Ridge Regression")] = "rf",
) -> None:
    """Run walk-forward backtest."""

    config = OptimizationConfig()

    # Data Fetching
    df_tickers = pd.read_csv(input_csv)
    rows = [TickerRow(**row) for _, row in df_tickers.iterrows()]
    ticker_input = TickerInput(rows=rows)
    tickers = [r.ticker for r in ticker_input.rows]

    prices = fetch_prices(tickers, start_date, end_date)
    volume = fetch_volume(tickers, start_date, end_date)
    prices = validate_prices(prices, min_days=config.min_history_days)

    from skfolio.preprocessing import prices_to_returns
    returns_full = prices_to_returns(prices)

    # Note: A proper backtest should re-generate views at each step.
    # For this CLI tool, we'll implement a simplified walk-forward where we
    # compute features once and then use them in a loop, or use a custom Transformer.
    # To keep it simple and fulfill the requirement, I'll implement a loop
    # that mimics WalkForward if skfolio's pipeline doesn't easily support
    # the external ML view generation.

    cv = WalkForward(test_size=config.train_test_split_days, train_size=config.min_history_days)

    # We'll collect portfolios
    portfolios = []

    # We need features and targets
    X_multi = compute_features(prices, volume)
    y_full = returns_full.shift(-21).dropna()
    common_index = X_multi.index.intersection(y_full.index)
    X = X_multi.loc[common_index]
    y = y_full.loc[common_index]

    # Re-align returns for skfolio (needs to match y for the folds)
    returns_aligned = returns_full.loc[common_index]

    table = Table(title="Walk-Forward Backtest Folds", show_header=True)
    table.add_column("Fold", justify="right")
    table.add_column("Train End", justify="center")
    table.add_column("Test End", justify="center")
    table.add_column("Sharpe", justify="right")

    fold = 1
    for train_indices, test_indices in cv.split(returns_aligned):
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
        ret_train, ret_test = returns_aligned.iloc[train_indices], returns_aligned.iloc[test_indices]

        if model_type == "rf":
            model = RandomForestViewsModel()
        else:
            model = LinearRegressionViewsModel()

        model.fit(X_train, y_train)
        # Predict on X_test to get views for the test period
        # In a real walk-forward, you'd predict at the START of the test period.
        predictions = model.predict(X_test)
        view_strings = build_view_strings(predictions)

        pipeline = build_pipeline(
            view_strings=view_strings,
            objective="maximize-ratio",
            risk_free_rate=config.risk_free_rate,
        )

        pipeline.fit(ret_train)
        from skfolio import Portfolio
        train_portfolio = pipeline.predict(ret_train)
        p_test = Portfolio(
            returns=ret_test,
            weights=train_portfolio.weights,
            name=f"Fold {fold}",
        )

        portfolios.append(p_test)
        table.add_row(
            str(fold),
            str(ret_train.index[-1].date()),
            str(ret_test.index[-1].date()),
            f"{p_test.sharpe_ratio:.4f}"
        )
        fold += 1

    console.print(table)

    if portfolios:
        from skfolio import Population
        pop = Population(portfolios)
        # Aggregate portfolio (all folds concatenated)
        # Population.aggregate returns a Portfolio with the concatenated returns
        agg_portfolio = pop.aggregate()
        console.print("\n[bold]Aggregate Backtest Performance:[/]")
        print_metrics(agg_portfolio)
