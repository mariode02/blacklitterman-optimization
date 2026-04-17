import typer
from typing import Annotated
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.table import Table
import yfinance as yf

from bl_optimizer.data.models import TickerInput, TickerRow

console = Console()

def validate_command(
    input_csv: Annotated[Path, typer.Option("--input", help="Path to tickers CSV", exists=True)],
) -> None:
    """Validate input CSV and ticker availability."""

    try:
        df_tickers = pd.read_csv(input_csv)
        rows = [TickerRow(**row) for _, row in df_tickers.iterrows()]
        ticker_input = TickerInput(rows=rows)
        tickers = [r.ticker for r in ticker_input.rows]
    except Exception as e:
        console.print(f"[red]Error parsing CSV:[/] {e}")
        raise typer.Exit(code=1)

    table = Table(title="Ticker Validation", show_header=True)
    table.add_column("Ticker", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Details")

    all_valid = True
    for ticker in tickers:
        try:
            # Download 5 days to check availability
            data = yf.download(ticker, period="5d", progress=False)
            if data is None or data.empty:
                table.add_row(ticker, "[red]Failed[/]", "No data found")
                all_valid = False
            else:
                table.add_row(ticker, "[green]Valid[/]", f"{len(data)} days found")
        except Exception as e:
            table.add_row(ticker, "[red]Error[/]", str(e))
            all_valid = False

    console.print(table)

    if not all_valid:
        console.print("[red]Validation failed for one or more tickers.[/]")
        raise typer.Exit(code=1)
    else:
        console.print("[green]All tickers validated successfully.[/]")
