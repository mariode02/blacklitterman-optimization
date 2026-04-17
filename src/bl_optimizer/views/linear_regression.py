from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd

class LinearRegressionViewsModel:
    """Generates views using Ridge regression with cross-validated alpha.

    Uses RidgeCV with TimeSeriesSplit to avoid look-ahead bias during alpha selection.
    """

    def __init__(self, alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)) -> None:
        self._model = RidgeCV(alphas=alphas, cv=TimeSeriesSplit(n_splits=5))
        self._tickers: list[str] = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        """Fit the model to the training data.

        X_train should be (n_samples, n_features * n_tickers)
        y_train should be (n_samples, n_tickers)
        """
        self._tickers = list(y_train.columns)
        self._model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict expected returns for each asset.

        Takes the last row of X as the most recent data point for prediction.
        Returns a Series indexed by ticker.
        """
        predictions = self._model.predict(X)
        # RidgeCV.predict returns (n_samples, n_outputs)
        return pd.Series(predictions[-1], index=self._tickers)
