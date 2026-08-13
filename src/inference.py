"""Python interface to the native Heat 1D modal-surrogate library."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import threading
import time

import numpy as np

import config


_FLOAT_POINTER = ctypes.POINTER(ctypes.c_float)


def _first_existing_path(candidates: list[Path], description: str) -> Path:
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    checked = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {description}. Checked:\n  {checked}")


def _resolve_library_path(path: str | os.PathLike[str] | None) -> Path:
    if path is not None:
        return _first_existing_path([Path(path).expanduser()], "native library")
    return _first_existing_path(
        [
            config.LIBRARY_PATH,
            config.ROOT / "build" / "libpinn_heat1d_backend.so",
            Path("/app/lib/libpinn_heat1d_backend.so"),
        ],
        "native Heat 1D backend library",
    )


class Heat1DModel:
    """Own a native model handle and expose full-field inference."""

    input_dim = 5

    def __init__(
        self,
        model_path: str | os.PathLike[str] = config.MODEL_PATH,
        library_path: str | os.PathLike[str] | None = None,
        metadata_path: str | os.PathLike[str] = config.METADATA_PATH,
    ) -> None:
        self.model_path = _first_existing_path(
            [Path(model_path).expanduser()], "Heat 1D model artifact"
        )
        self.library_path = _resolve_library_path(library_path)
        self.metadata_path = _first_existing_path(
            [Path(metadata_path).expanduser()], "Heat 1D model metadata"
        )
        with self.metadata_path.open("r", encoding="utf-8") as metadata_file:
            self.metadata = json.load(metadata_file)

        self._library = ctypes.CDLL(str(self.library_path))
        self._configure_abi()
        self._handle = self._library.heat1d_inference_load(
            os.fsencode(self.model_path)
        )
        if not self._handle:
            raise RuntimeError(
                f"Native backend rejected model artifact: {self.model_path}"
            )
        if self._library.heat1d_inference_input_dim() != self.input_dim:
            self.close()
            raise RuntimeError("Native Heat 1D input contract is not five-dimensional")

        # The C autograd tape is process-global, so concurrent FastAPI calls
        # must not enter native inference at the same time.
        self._lock = threading.Lock()

    def _configure_abi(self) -> None:
        self._library.heat1d_inference_load.argtypes = [ctypes.c_char_p]
        self._library.heat1d_inference_load.restype = ctypes.c_void_p
        self._library.heat1d_inference_free.argtypes = [ctypes.c_void_p]
        self._library.heat1d_inference_free.restype = None
        self._library.heat1d_inference_predict.argtypes = [
            ctypes.c_void_p,
            _FLOAT_POINTER,
            ctypes.c_int,
            _FLOAT_POINTER,
        ]
        self._library.heat1d_inference_predict.restype = ctypes.c_int
        self._library.heat1d_inference_input_dim.argtypes = []
        self._library.heat1d_inference_input_dim.restype = ctypes.c_int

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            self._library.heat1d_inference_free(handle)
            self._handle = None

    def __enter__(self) -> "Heat1DModel":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def predict_points(self, physical_points: np.ndarray) -> np.ndarray:
        """Predict rows in ``[t, x, alpha, a1, a2]`` order."""
        if not self._handle:
            raise RuntimeError("Heat 1D model has been closed")
        points = np.asarray(physical_points, dtype=np.float32)
        if points.ndim < 2 or points.shape[-1] != self.input_dim:
            raise ValueError("physical_points must have shape (..., 5)")
        output_shape = points.shape[:-1]
        flat_points = np.ascontiguousarray(points.reshape(-1, self.input_dim))
        if flat_points.shape[0] == 0:
            return np.empty(output_shape, dtype=np.float32)
        output = np.empty(flat_points.shape[0], dtype=np.float32)
        with self._lock:
            succeeded = self._library.heat1d_inference_predict(
                self._handle,
                flat_points.ctypes.data_as(_FLOAT_POINTER),
                flat_points.shape[0],
                output.ctypes.data_as(_FLOAT_POINTER),
            )
        if not succeeded:
            raise ValueError("Native inference rejected one or more input values")
        return output.reshape(output_shape)

    def predict_grid(
        self, a1: float, a2: float, alpha: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_grid = np.linspace(0.0, config.L, config.GRID_NX, dtype=np.float32)
        t_grid = np.linspace(0.0, config.T, config.GRID_NT, dtype=np.float32)
        xx, tt = np.meshgrid(x_grid, t_grid, indexing="ij")
        points = np.empty((config.GRID_NX, config.GRID_NT, 5), dtype=np.float32)
        points[..., 0] = tt
        points[..., 1] = xx
        points[..., 2] = alpha
        points[..., 3] = a1
        points[..., 4] = a2
        return self.predict_points(points), x_grid, t_grid


def load_model(
    path: str | os.PathLike[str] = config.MODEL_PATH,
    device: str = "cpu",
) -> Heat1DModel:
    """Load the native model; ``device`` mirrors the reference API."""
    if device != "cpu":
        raise ValueError("The deployed inference library currently targets CPU")
    return Heat1DModel(model_path=path)


def pinn_predict(
    model: Heat1DModel, a1: float, a2: float, alpha: float
) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    prediction, _, _ = model.predict_grid(a1, a2, alpha)
    return prediction, (time.perf_counter() - start) * 1000.0


def exact_predict(
    a1: float, a2: float, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    start = time.perf_counter()
    x_grid = np.linspace(0.0, config.L, config.GRID_NX, dtype=np.float32)
    t_grid = np.linspace(0.0, config.T, config.GRID_NT, dtype=np.float32)
    xx, tt = np.meshgrid(x_grid, t_grid, indexing="ij")
    exact = (
        a1 * np.sin(np.pi * xx) * np.exp(-alpha * np.pi**2 * tt)
        + a2
        * np.sin(2.0 * np.pi * xx)
        * np.exp(-4.0 * alpha * np.pi**2 * tt)
    )
    return exact.astype(np.float32), x_grid, t_grid, (
        time.perf_counter() - start
    ) * 1000.0


def compare(model: Heat1DModel, a1: float, a2: float, alpha: float) -> dict:
    pinn, pinn_ms = pinn_predict(model, a1, a2, alpha)
    exact, x_grid, t_grid, exact_ms = exact_predict(a1, a2, alpha)
    absolute_error = np.abs(pinn - exact)
    exact_norm = float(np.linalg.norm(exact))
    relative_l2 = (
        float(np.linalg.norm(pinn - exact) / exact_norm)
        if exact_norm > 0.0
        else float(np.linalg.norm(pinn))
    )
    return {
        "pinn": pinn,
        "exact": exact,
        "absolute_error": absolute_error,
        "relative_l2_error": relative_l2,
        "max_absolute_error": float(np.max(absolute_error, initial=0.0)),
        "pinn_ms": pinn_ms,
        "exact_ms": exact_ms,
        "x_grid": x_grid,
        "t_grid": t_grid,
    }
