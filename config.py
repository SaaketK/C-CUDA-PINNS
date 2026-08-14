"""Deployment configuration for the parametric Heat 1D surrogate."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent

L = 1.0
T = 1.0
ALPHA_MIN = 0.01
ALPHA_MAX = 0.50
A1_MIN = -1.0
A1_MAX = 1.0
A2_MIN = -1.0
A2_MAX = 1.0

# Keep the public API fixed-size like the reference deployment. This bounds
# response size and makes cached calls identical for identical parameters.
GRID_NX = 100
GRID_NT = 100
# The interactive UI uses a lighter grid so live parameter changes stay
# responsive on a free single-CPU deployment. Public API responses remain
# fixed at GRID_NX x GRID_NT.
UI_GRID_NX = 50
UI_GRID_NT = 50
CACHE_SIZE = 256

MODEL_PATH = Path(
    os.environ.get(
        "PINN_HEAT1D_MODEL",
        ROOT / "models" / "heat1d" / "v1" / "model.pinn",
    )
)
METADATA_PATH = Path(
    os.environ.get(
        "PINN_HEAT1D_METADATA",
        ROOT / "models" / "heat1d" / "v1" / "metadata.json",
    )
)
LIBRARY_PATH = Path(
    os.environ.get(
        "PINN_HEAT1D_LIBRARY",
        ROOT / "lib" / "libpinn_heat1d_backend.so",
    )
)
