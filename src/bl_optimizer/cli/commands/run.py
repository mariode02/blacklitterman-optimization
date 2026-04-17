import typer
from typing import Annotated, Optional
from pathlib import Path
from datetime import datetime
import pandas as pd
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console

from bl_optimizer.config.settings import OptimizationConfig
from bl_optimizer.data.models import TickerInput, TickerRow
from bl_optimizer.data.fetcher import fetch_prices, fetch_volume
from bl_optimizer.data.validator import validate_prices
from bl_optimizer.features.engineer import compute_features
from bl_optimizer.views.random_forest import RandomForestViewsModel
from bl_optimizer.views.linear_regression import LinearRegressionViewsModel
from bl_optimizer.views.view_builder import build_view_strings
from bl_optimizer.optimization.optimizer import build_pipeline
from bl_optimizer.reporting.reporter import print_weights, print_metrics, export_weights_csv

app = typer.Typer(help="Run portfolio optimization.")
console = Console()

@app.callback(invoke_without_command=True)
def run_command(
    input_csv: Annotated[Path, typer.Option("--input", help="Path to tickers CSV", exists=True)],
    start_date: Annotated[str, typer.Option("--start", help="Start date (ISO 8601)")] = "2018-01-01",
    end_date: Annotated[str, typer.Option("--end", help="End date (ISO 8601)")] = datetime.now().strftime("%Y-%m-%d"),
    model_type: Annotated[str, typer.Option("--model", help="ML model for views: rf=Random Forest, lr=Ridge Regression")] = "rf",
    objective: Annotated[str, typer.Option("--objective", help="Optimization objective")] = "maximize-ratio",
    output: Annotated[Optional[Path], typer.Option("--output", help="If set, writes weights CSV to this path")] = None,
) -> None:
    """Run Black-Litterman portfolio optimization with ML views."""

    config = OptimizationConfig()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        # 1. Parsing and validating input CSV
        progress.add_task(description="[1/5] Parsing and validating input CSV", total=None)
        df_tickers = pd.read_csv(input_csv)
        rows = [TickerRow(**row) for _, row in df_tickers.iterrows()]
        ticker_input = TickerInput(rows=rows)
        tickers = [r.ticker for r in ticker_input.rows]
        market_cap_weights = None
        if all(r.market_cap_weight is not None for r in ticker_input.rows):
             market_cap_weights = pd.Series(
                 [r.market_cap_weight for r in ticker_input.rows],
                 index=tickers
             )

        # 2. Fetching price history via yfinance
        progress.add_task(description="[2/5] Fetching price history via yfinance", total=None)
        prices = fetch_prices(tickers, start_date, end_date)
        volume = fetch_volume(tickers, start_date, end_date)
        prices = validate_prices(prices, min_days=config.min_history_days)

        # 3. Engineering features
        progress.add_task(description="[3/5] Engineering features", total=None)
        X_multi = compute_features(prices, volume)

        # Prepare target: 21-day forward return
        # Returns conversion
        from skfolio.preprocessing import prices_to_returns
        _ret = prices_to_returns(prices)
        returns_full: pd.DataFrame = _ret if isinstance(_ret, pd.DataFrame) else _ret[0]

        y_full: pd.DataFrame = returns_full.shift(-21).dropna()
        common_index = X_multi.index.intersection(y_full.index)
        X = X_multi.loc[common_index]
        y = y_full.loc[common_index]

        # 4. Training ML model and generating views
        progress.add_task(description="[4/5] Training ML model and generating views", total=None)

        # Split
        split_idx = -config.train_test_split_days
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        if model_type == "rf":
            model = RandomForestViewsModel()
        elif model_type == "lr":
            model = LinearRegressionViewsModel()
        else:
            raise typer.BadParameter(f"Invalid model type: {model_type}")

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        view_strings = build_view_strings(predictions)

        # 5. Running portfolio optimization
        progress.add_task(description="[5/5] Running portfolio optimization", total=None)
        from skfolio.optimization import ObjectiveFunction
        obj_map = {
            "maximize-ratio": ObjectiveFunction.MAXIMIZE_RATIO,
            "maximize-return": ObjectiveFunction.MAXIMIZE_RETURN,
            "minimize-risk": ObjectiveFunction.MINIMIZE_RISK,
        }
        pipeline = build_pipeline(
            view_strings=view_strings,
            market_cap_weights=market_cap_weights,
            objective=obj_map.get(objective, ObjectiveFunction.MAXIMIZE_RATIO),
            risk_free_rate=config.risk_free_rate,
            correlation_threshold=config.correlation_drop_threshold,
            min_weight=config.min_weight,
            max_weight=config.max_weight,
            risk_aversion=config.risk_aversion,
            tau=config.tau,
        )

        pipeline.fit(returns_full)
        portfolio = pipeline.predict(returns_full)

    # Final Reporting
    print_weights(portfolio)
    print_metrics(portfolio)

    if output:
        export_weights_csv(portfolio, str(output))
