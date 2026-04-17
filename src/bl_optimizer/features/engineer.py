import pandas as pd
import numpy as np

def compute_features(prices: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Compute ML features for all tickers.

    Returns:
        DataFrame indexed by date with MultiIndex columns (ticker, feature).
        Rows with any NaN are dropped.
    """
    frames: list[pd.DataFrame] = []
    for ticker in prices.columns:
        close = prices[ticker]
        vol = volume[ticker] if ticker in volume.columns else None
        daily_ret = close.pct_change()

        features = pd.DataFrame(index=close.index)
        features["return_1m"]     = close.pct_change(21)
        features["return_3m"]     = close.pct_change(63)
        features["return_6m"]     = close.pct_change(126)
        features["return_12m"]    = close.pct_change(252)
        features["volatility_21d"]= daily_ret.rolling(21).std()
        features["rsi_14"]        = _compute_rsi(daily_ret, window=14)

        if vol is not None:
             features["volume_trend"]  = vol.rolling(5).mean() / vol.rolling(20).mean().replace(0, float("nan"))
        else:
             # If volume is missing for some reason, we still need the column structure.
             features["volume_trend"] = 1.0

        features["ma_crossover"]  = close.rolling(50).mean() / close.rolling(200).mean().replace(0, float("nan"))

        features.columns = pd.MultiIndex.from_product([[ticker], features.columns])
        frames.append(features)

    return pd.concat(frames, axis=1).dropna()


def _compute_rsi(returns: pd.Series, window: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    gains = returns.clip(lower=0).rolling(window).mean()
    losses = (-returns.clip(upper=0)).rolling(window).mean()
    rs = gains / losses.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))
