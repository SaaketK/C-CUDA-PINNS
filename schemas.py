"""Validated HTTP request and response contracts."""

from pydantic import BaseModel, Field

import config


class PredictRequest(BaseModel):
    a1: float = Field(default=1.0, ge=config.A1_MIN, le=config.A1_MAX)
    a2: float = Field(default=0.0, ge=config.A2_MIN, le=config.A2_MAX)
    alpha: float = Field(
        default=0.1, ge=config.ALPHA_MIN, le=config.ALPHA_MAX
    )


class PinnResponse(BaseModel):
    u: list[list[float]]
    pinn_ms: float


class ExactResponse(BaseModel):
    u: list[list[float]]
    exact_ms: float


class CompareResponse(BaseModel):
    pinn: list[list[float]]
    exact: list[list[float]]
    absolute_error: list[list[float]]
    relative_l2_error: float
    max_absolute_error: float
    pinn_ms: float
    exact_ms: float
    x_grid: list[float]
    t_grid: list[float]
