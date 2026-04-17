import yfinance as yf
import pandas as pd


class DataFetchError(Exception):
    pass


def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted daily close prices for all tickers."""
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        repair=True,
        progress=False,
    )
    if raw is None or raw.empty:
        raise DataFetchError("No data fetched from Yahoo Finance.")

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
        prices: pd.DataFrame = close if isinstance(close, pd.DataFrame) else close.to_frame(name=tickers[0])
    elif "Close" in raw.columns:
        prices = raw[["Close"]]
    else:
        raise DataFetchError(f"Close price column not found: {raw.columns.tolist()}")

    return prices


def fetch_volume(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download volume data for all tickers."""
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )
    if raw is None or raw.empty:
        raise DataFetchError("No volume data fetched from Yahoo Finance.")

    if isinstance(raw.columns, pd.MultiIndex):
        vol = raw["Volume"]
        volume: pd.DataFrame = vol if isinstance(vol, pd.DataFrame) else vol.to_frame(name=tickers[0])
    else:
        volume = raw[["Volume"]]

    return volume
