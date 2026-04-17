from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class OptimizationConfig(BaseSettings):
    """Optimization hyperparameters.

    All fields are configurable via environment variables with the BL_ prefix.
    Example: BL_RISK_FREE_RATE=0.05

    CLI flags override environment variables when both are present.
    """

    model_config = SettingsConfigDict(env_prefix="BL_", case_sensitive=False)

    risk_free_rate: float = Field(default=0.04, gt=0.0, description="Annual risk-free rate.")
    risk_aversion: float = Field(default=1.0, gt=0.0, description="EquilibriumMu lambda parameter.")
    tau: float = Field(default=0.05, gt=0.0, lt=1.0, description="Black-Litterman tau scalar.")
    min_weight: float = Field(default=0.0, ge=0.0, description="Minimum weight per asset.")
    max_weight: float = Field(default=0.40, gt=0.0, le=1.0, description="Maximum weight per asset.")
    correlation_drop_threshold: float = Field(
        default=0.90, gt=0.0, le=1.0,
        description="Pairwise correlation above this drops the lower-return asset."
    )
    min_history_days: int = Field(default=252, gt=0, description="Minimum trading days required.")
    train_test_split_days: int = Field(default=60, gt=0, description="Test set size in trading days.")
