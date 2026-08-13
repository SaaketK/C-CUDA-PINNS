"""Tests for the Python/native deployment boundary."""

import numpy as np
import pytest

from src.inference import compare, load_model


@pytest.fixture(scope="module")
def model():
    loaded = load_model()
    yield loaded
    loaded.close()


@pytest.mark.parametrize(
    "a1,a2,alpha",
    [
        (1.0, 0.0, 0.10),
        (0.0, 1.0, 0.50),
        (0.7, -0.4, 0.05),
        (-0.6, 0.25, 0.20),
        (-1.0, 1.0, 0.50),
    ],
)
def test_full_field_accuracy(model, a1, a2, alpha):
    result = compare(model, a1, a2, alpha)
    assert result["pinn"].shape == (100, 100)
    assert np.isfinite(result["pinn"]).all()
    assert result["relative_l2_error"] < 0.01


def test_boundary_and_initial_conditions(model):
    result = compare(model, 0.7, -0.4, 0.10)
    prediction = result["pinn"]
    assert np.max(np.abs(prediction[[0, -1], :])) < 1e-6
    assert np.max(np.abs(prediction[:, 0] - result["exact"][:, 0])) < 2e-6
