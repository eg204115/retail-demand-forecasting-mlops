"""Request and response contracts.

Validation at the edge is cheap insurance: a silently coerced null lag produces a
plausible-looking forecast, which is far worse than a 422.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, field_validator


class ForecastRequest(BaseModel):
    store_id: str = Field(..., examples=["CA_1"])
    item_id: str = Field(..., examples=["ITEM_042"])
    forecast_date: date | None = Field(
        None, description="Origin date. Defaults to today when omitted."
    )
    horizon: int = Field(14, ge=1, le=28)
    # Optional escape hatch: callers may pass features directly instead of having the
    # service pull them from the online store (useful for what-if pricing scenarios).
    features: dict[str, float] | None = None

    @field_validator("store_id", "item_id")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v.strip()


class BatchForecastRequest(BaseModel):
    items: list[ForecastRequest] = Field(..., min_length=1, max_length=1000)


class DayForecast(BaseModel):
    date: date
    p50: float
    p90: float


class ForecastResponse(BaseModel):
    store_id: str
    item_id: str
    horizon: int
    model_version: str
    forecasts: list[DayForecast]
    order_qty: int | None = None


class BatchForecastResponse(BaseModel):
    results: list[ForecastResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
    feature_store: str
