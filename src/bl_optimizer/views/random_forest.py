from sklearn.ensemble import RandomForestRegressor
import pandas as pd

class RandomForestViewsModel:
    """Generates views using a Random Forest regressor.

    Trains one multi-output regressor where each output corresponds to one ticker.
    """

    def __init__(self, n_estimators: int = 200, random_state: int = 42) -> None:
        self._model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
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
        # We assume the predictions are for a specific horizon (e.g. 21 days).
        # We take the last row as the current forecast.
        return pd.Series(predictions[-1], index=self._tickers)
