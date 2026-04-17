from pydantic import BaseModel, Field, model_validator

class TickerRow(BaseModel):
    ticker: str = Field(min_length=1, pattern=r"^[A-Z0-9.\-^=]+$")
    market_cap_weight: float | None = Field(default=None, gt=0.0, le=1.0)

class TickerInput(BaseModel):
    rows: list[TickerRow] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_weights_consistency(self) -> "TickerInput":
        has_weights = [r.market_cap_weight is not None for r in self.rows]
        if any(has_weights) and not all(has_weights):
            raise ValueError("market_cap_weight must be provided for all tickers or none.")
        if all(has_weights):
            total = sum(r.market_cap_weight for r in self.rows)  # type: ignore[arg-type]
            if abs(total - 1.0) > 0.001:
                raise ValueError(f"market_cap_weight values must sum to 1.0, got {total:.4f}.")
        return self
