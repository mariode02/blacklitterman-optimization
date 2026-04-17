from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from skfolio import Portfolio
import pandas as pd

console = Console()

def print_weights(portfolio: Portfolio) -> None:
    """Print optimized portfolio weights to the terminal."""
    table = Table(title="Optimized Portfolio Weights", show_header=True)
    table.add_column("Ticker", style="cyan")
    table.add_column("Weight", justify="right")
    table.add_column("Weight %", justify="right", style="green")

    # skfolio Portfolio has asset names and weights
    for ticker, weight in zip(portfolio.assets, portfolio.weights):
        table.add_row(ticker, f"{weight:.4f}", f"{weight * 100:.2f}%")
    console.print(table)


def print_metrics(portfolio: Portfolio) -> None:
    """Print portfolio performance metrics to the terminal."""
    lines = [
        f"Sharpe Ratio         : {portfolio.sharpe_ratio:.4f}",
        f"Sortino Ratio        : {portfolio.sortino_ratio:.4f}",
        f"CVaR (95%)           : {portfolio.cvar * 100:.2f}%",
        f"Max Drawdown         : {portfolio.max_drawdown * 100:.2f}%",
        f"Annualized Return    : {portfolio.annualized_mean * 100:.2f}%",
        f"Annualized Volatility: {portfolio.annualized_standard_deviation * 100:.2f}%",
    ]
    console.print(Panel("\n".join(lines), title="Portfolio Performance Metrics"))


def export_weights_csv(portfolio: Portfolio, output_path: str) -> None:
    """Export portfolio weights to a CSV file."""
    df = pd.DataFrame({"ticker": portfolio.assets, "weight": portfolio.weights})
    df.to_csv(output_path, index=False)
    console.print(f"[green]Weights exported to:[/] {output_path}")
