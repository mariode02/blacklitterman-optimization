import pandas as pd
from rich.console import Console

console = Console()

class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass

def validate_prices(prices: pd.DataFrame, min_days: int = 252) -> pd.DataFrame:
    """Validate and clean a price DataFrame.

    Raises:
        DataValidationError: If critical quality thresholds are not met.
    """
    for ticker in prices.columns:
        col = prices[ticker]
        if col.isna().all():
            raise DataValidationError(f"No data returned for {ticker}.")
        missing_pct = col.isna().mean()
        if missing_pct > 0.05:
            console.print(f"[yellow]Warning:[/] {ticker} has {missing_pct:.1%} missing values — forward-filling.")
        if len(col.dropna()) < min_days:
            raise DataValidationError(
                f"{ticker} has only {len(col.dropna())} trading days. Minimum required: {min_days}."
            )
    return prices.ffill()
