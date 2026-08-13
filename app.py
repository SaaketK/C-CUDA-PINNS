"""Hugging Face Space: FastAPI API plus Gradio user interface."""

from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache
import os

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware

import config
from schemas import CompareResponse, ExactResponse, PinnResponse, PredictRequest
from src.inference import Heat1DModel, compare, exact_predict, load_model, pinn_predict


_model: Heat1DModel | None = None


def _require_model() -> Heat1DModel:
    if _model is None:
        raise RuntimeError("The native Heat 1D model is not initialized")
    return _model


@lru_cache(maxsize=config.CACHE_SIZE)
def _cached_compare(a1: float, a2: float, alpha: float) -> dict:
    return compare(_require_model(), a1, a2, alpha)


def _request_result(request: PredictRequest) -> dict:
    # Mirroring the reference Space, nearby slider requests share cached work.
    return _cached_compare(
        round(request.a1, 3), round(request.a2, 3), round(request.alpha, 3)
    )


@asynccontextmanager
async def lifespan(application: FastAPI):
    global _model
    _model = load_model(config.MODEL_PATH, device="cpu")
    application.state.model = _model
    try:
        yield
    finally:
        _cached_compare.cache_clear()
        _model.close()
        _model = None


app = FastAPI(
    title="C/CUDA PINN Heat 1D Surrogate",
    description="Native C inference for a parametric one-dimensional heat PINN.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if _model is not None else "starting",
        "backend": "native-c-modal",
        "model_loaded": _model is not None,
    }


@app.post("/api/v1/predict/pinn", response_model=PinnResponse)
def predict_pinn(request: PredictRequest) -> PinnResponse:
    try:
        prediction, elapsed_ms = pinn_predict(
            _require_model(), request.a1, request.a2, request.alpha
        )
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return PinnResponse(u=prediction.tolist(), pinn_ms=elapsed_ms)


@app.post("/api/v1/predict/exact", response_model=ExactResponse)
def predict_exact(request: PredictRequest) -> ExactResponse:
    solution, _, _, elapsed_ms = exact_predict(
        request.a1, request.a2, request.alpha
    )
    return ExactResponse(u=solution.tolist(), exact_ms=elapsed_ms)


@app.post("/api/v1/predict/compare", response_model=CompareResponse)
def predict_compare(request: PredictRequest) -> CompareResponse:
    try:
        result = _request_result(request)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return CompareResponse(
        pinn=result["pinn"].tolist(),
        exact=result["exact"].tolist(),
        absolute_error=result["absolute_error"].tolist(),
        relative_l2_error=result["relative_l2_error"],
        max_absolute_error=result["max_absolute_error"],
        pinn_ms=result["pinn_ms"],
        exact_ms=result["exact_ms"],
        x_grid=result["x_grid"].tolist(),
        t_grid=result["t_grid"].tolist(),
    )


@app.get("/api/v1/model/info")
def model_info() -> dict:
    model = _require_model()
    return {
        "backend": "native-c-modal",
        "model_file": model.model_path.name,
        "library_file": model.library_path.name,
        "grid": [config.GRID_NX, config.GRID_NT],
        "metadata": model.metadata,
    }


def _heatmap(
    values: np.ndarray,
    title: str,
    *,
    cmap: str = "viridis",
    value_range: tuple[float, float] | None = None,
):
    figure, axis = plt.subplots(figsize=(5.6, 4.0))
    limits = (
        {}
        if value_range is None
        else {"vmin": value_range[0], "vmax": value_range[1]}
    )
    image = axis.imshow(
        values,
        origin="lower",
        aspect="auto",
        extent=[0.0, config.T, 0.0, config.L],
        cmap=cmap,
        **limits,
    )
    axis.set_title(title)
    axis.set_xlabel("time t")
    axis.set_ylabel("position x")
    figure.colorbar(image, ax=axis, label="|error|" if cmap == "magma" else "u")
    figure.tight_layout()
    plt.close(figure)
    return figure


def _gradio_predict(a1: float, a2: float, alpha: float):
    result = _cached_compare(
        round(float(a1), 3), round(float(a2), 3), round(float(alpha), 3)
    )
    shared_range = (
        float(min(result["pinn"].min(), result["exact"].min())),
        float(max(result["pinn"].max(), result["exact"].max())),
    )
    metrics = (
        f"**Relative L2 error:** `{result['relative_l2_error']:.4e}`  \n"
        f"**Maximum absolute error:** `{result['max_absolute_error']:.4e}`  \n"
        f"**Native PINN inference:** `{result['pinn_ms']:.2f} ms`"
    )
    return (
        _heatmap(result["pinn"], "C PINN surrogate", value_range=shared_range),
        _heatmap(result["exact"], "Exact modal solution", value_range=shared_range),
        _heatmap(result["absolute_error"], "Absolute error", cmap="magma"),
        metrics,
    )


with gr.Blocks(title="C/CUDA PINN Heat 1D") as demo:
    gr.Markdown(
        "# Parametric 1D heat equation with a native C PINN\n"
        "Explore diffusivity and the first two Fourier-mode amplitudes. "
        "The same trained modal surrogate is reused for every selection."
    )
    with gr.Row():
        a1_input = gr.Slider(-1.0, 1.0, value=1.0, step=0.05, label="Mode 1: a₁")
        a2_input = gr.Slider(-1.0, 1.0, value=0.0, step=0.05, label="Mode 2: a₂")
        alpha_input = gr.Slider(
            config.ALPHA_MIN,
            config.ALPHA_MAX,
            value=0.1,
            step=0.01,
            label="Diffusivity: α",
        )
    predict_button = gr.Button("Solve", variant="primary")
    with gr.Row():
        pinn_output = gr.Plot(label="PINN")
        exact_output = gr.Plot(label="Exact")
        error_output = gr.Plot(label="Absolute error")
    metrics_output = gr.Markdown()
    predict_button.click(
        _gradio_predict,
        inputs=[a1_input, a2_input, alpha_input],
        outputs=[pinn_output, exact_output, error_output, metrics_output],
    )
    gr.Examples(
        examples=[
            [1.0, 0.0, 0.10],
            [0.0, 1.0, 0.50],
            [0.7, -0.4, 0.05],
            [-0.6, 0.25, 0.20],
            [-1.0, 1.0, 0.50],
        ],
        inputs=[a1_input, a2_input, alpha_input],
    )


app = gr.mount_gradio_app(
    app,
    demo,
    path="/",
    root_path=os.environ.get("GRADIO_ROOT_PATH"),
)
