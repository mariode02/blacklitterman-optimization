from typing import Protocol
import pandas as pd

class ViewsModel(Protocol):
    """Contract for ML models that generate per-asset return forecasts."""

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        """Train the model on historical feature/target pairs."""
        ...

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict expected returns for each asset.

        Returns:
            Series indexed by ticker symbol, values are predicted
            annualized returns (float).
        """
        ...
