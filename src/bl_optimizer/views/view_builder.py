import pandas as pd

def build_view_strings(predictions: pd.Series) -> list[str]:
    """Convert per-ticker return predictions to BL absolute view strings.

    Args:
        predictions: Series indexed by ticker, values are predicted annualized returns.

    Returns:
        List of strings like ["AAPL == 0.0823", "MSFT == 0.0712", ...].
    """
    return [f"{ticker} == {ret:.6f}" for ticker, ret in predictions.items()]
