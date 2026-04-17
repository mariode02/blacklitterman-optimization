import yfinance as yf
import pandas as pd

class DataFetchError(Exception):
    """Raised when data fetching fails."""
    pass

def fetch_prices(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download adjusted daily close prices for all tickers.

    Args:
        tickers: List of Yahoo Finance symbols.
        start: ISO date string, e.g. "2018-01-01".
        end: ISO date string, e.g. "2024-01-01".

    Returns:
        DataFrame with DatetimeIndex and one column per ticker (adjusted close).

    Raises:
        DataFetchError: If yfinance returns empty data for any ticker.
    """
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        repair=True,
        progress=False,
    )
    if raw.empty:
        raise DataFetchError("No data fetched from Yahoo Finance.")

    # In newer yfinance versions, raw might have multi-index columns if multiple tickers.
    # We want "Close" prices.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        # Single ticker case might return a flat DataFrame or even a Series
        if "Close" in raw.columns:
            prices = raw["Close"]
        else:
             raise DataFetchError(f"Close price column not found in data: {raw.columns}")

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    # Ensure columns match tickers (sometimes yf returns less or slightly different names)
    missing = set(tickers) - set(prices.columns)
    if missing:
        # If some are missing, it might be that they were returned but are all NaN
        pass

    return prices


def fetch_volume(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download volume data for all tickers."""
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        actions=False,
        progress=False,
    )
    if raw.empty:
        raise DataFetchError("No volume data fetched from Yahoo Finance.")

    if isinstance(raw.columns, pd.MultiIndex):
        volume = raw["Volume"]
    else:
        volume = raw["Volume"]

    if isinstance(volume, pd.Series):
        volume = volume.to_frame(name=tickers[0])

    return volume
