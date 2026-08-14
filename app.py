"""Hugging Face Space: FastAPI API plus Gradio user interface."""

from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache
import os

import gradio as gr
import numpy as np
import plotly.graph_objects as go
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


@lru_cache(maxsize=config.CACHE_SIZE)
def _cached_ui_compare(a1: float, a2: float, alpha: float) -> dict:
    return compare(
        _require_model(),
        a1,
        a2,
        alpha,
        nx=config.UI_GRID_NX,
        nt=config.UI_GRID_NT,
    )


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
        _cached_ui_compare.cache_clear()
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
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    *,
    colorscale: str,
    colorbar_title: str,
    value_range: tuple[float, float] | None = None,
) -> go.Figure:
    limits = (
        {}
        if value_range is None
        else {"zmin": value_range[0], "zmax": value_range[1]}
    )
    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=t_grid,
            y=x_grid,
            colorscale=colorscale,
            zsmooth="best",
            colorbar={
                "title": {"text": colorbar_title, "side": "right"},
                "thickness": 14,
                "outlinewidth": 0,
            },
            hovertemplate=(
                "t=%{x:.3f}<br>x=%{y:.3f}<br>"
                + colorbar_title
                + "=%{z:.5f}<extra></extra>"
            ),
            **limits,
        )
    )
    _style_figure(figure, bottom=55)
    figure.update_layout(
        xaxis_title="time t",
        yaxis_title="position x",
        uirevision="heat1d-ui",
    )
    return figure


def _style_figure(figure: go.Figure, *, bottom: int) -> None:
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(24,24,27,0.72)",
        font={"color": "#e4e4e7", "family": "Inter, ui-sans-serif, sans-serif"},
        margin={"l": 58, "r": 28, "t": 18, "b": bottom},
        height=410,
        hoverlabel={
            "bgcolor": "#18181b",
            "bordercolor": "#52525b",
            "font": {"color": "#fafafa"},
        },
    )
    figure.update_xaxes(
        gridcolor="rgba(113,113,122,0.28)",
        zerolinecolor="rgba(161,161,170,0.45)",
        linecolor="#52525b",
    )
    figure.update_yaxes(
        gridcolor="rgba(113,113,122,0.28)",
        zerolinecolor="rgba(161,161,170,0.45)",
        linecolor="#52525b",
    )


def _animated_profile(result: dict) -> tuple[go.Figure, list[dict]]:
    """Build a browser-side animation of the C-PINN solution through time."""
    x_grid = np.asarray(result["x_grid"])
    t_grid = np.asarray(result["t_grid"])
    prediction = np.asarray(result["pinn"])
    frame_indices = list(range(0, len(t_grid), 2))
    if frame_indices[-1] != len(t_grid) - 1:
        frame_indices.append(len(t_grid) - 1)

    frames = [
        {
            "name": str(index),
            "data": [
                {
                    "type": "scatter",
                    "x": x_grid.tolist(),
                    "y": prediction[:, index].tolist(),
                    "mode": "lines",
                    "line": {"color": "#f97316", "width": 4},
                }
            ],
        }
        for index in frame_indices
    ]
    y_bound = max(float(np.max(np.abs(prediction), initial=0.0)) * 1.08, 0.1)
    figure = go.Figure(
        data=[
            go.Scatter(
                x=x_grid,
                y=prediction[:, frame_indices[0]],
                mode="lines",
                line={"color": "#f97316", "width": 4},
                name="C PINN",
            )
        ]
    )
    _style_figure(figure, bottom=92)
    figure.update_layout(
        xaxis={"title": "position x", "range": [0.0, config.L]},
        yaxis={"title": "u(x,t)", "range": [-y_bound, y_bound]},
        showlegend=False,
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "active": -1,
                "showactive": False,
                "x": 0.0,
                "y": -0.17,
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "frame": {"duration": 85, "redraw": False},
                                "transition": {
                                    "duration": 70,
                                    "easing": "cubic-in-out",
                                },
                            },
                        ],
                    },
                    {
                        "label": "❚❚ Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.25,
                "len": 0.75,
                "y": -0.12,
                "currentvalue": {"prefix": "time t = "},
                "steps": [
                    {
                        "label": f"{float(t_grid[index]):.2f}",
                        "method": "animate",
                        "args": [
                            [str(index)],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 55},
                            },
                        ],
                    }
                    for index in frame_indices
                ],
            }
        ],
    )
    return figure, frames


def _equation_markdown(a1: float, a2: float, alpha: float) -> str:
    a1_text = f"{a1:.2f}".replace("-", "−")
    a2_operator = "+" if a2 >= 0.0 else "−"
    return (
        "### Evolving solution\n"
        f"<span class=\"equation\">u(x,t) ≈ {a1_text}"
        f"e<sup>−{alpha:.2f}π²t</sup> sin(πx) {a2_operator} "
        f"{abs(a2):.2f}e<sup>−4·{alpha:.2f}π²t</sup> sin(2πx)</span>"
    )


def _gradio_predict(a1: float, a2: float, alpha: float):
    a1 = round(float(a1), 3)
    a2 = round(float(a2), 3)
    alpha = round(float(alpha), 3)
    result = _cached_ui_compare(a1, a2, alpha)
    shared_range = (
        float(min(result["pinn"].min(), result["exact"].min())),
        float(max(result["pinn"].max(), result["exact"].max())),
    )
    profile, animation_frames = _animated_profile(result)
    metrics = (
        f"**Relative L2 error:** `{result['relative_l2_error']:.4e}`  \n"
        f"**Maximum absolute error:** `{result['max_absolute_error']:.4e}`  \n"
        f"**Native PINN inference ({config.UI_GRID_NX}×{config.UI_GRID_NT}):** "
        f"`{result['pinn_ms']:.2f} ms`"
    )
    return (
        _equation_markdown(a1, a2, alpha),
        profile,
        _heatmap(
            result["pinn"],
            result["x_grid"],
            result["t_grid"],
            colorscale="Viridis",
            colorbar_title="u",
            value_range=shared_range,
        ),
        _heatmap(
            result["absolute_error"],
            result["x_grid"],
            result["t_grid"],
            colorscale="Magma",
            colorbar_title="|error|",
        ),
        metrics,
        animation_frames,
    )


APP_CSS = """
.plot-panel {
    background: rgba(24, 24, 27, 0.68);
    border: 1px solid rgba(82, 82, 91, 0.65);
    border-radius: 12px;
    padding: 14px 14px 4px;
}
.plot-heading {
    min-height: 66px;
    padding: 0 4px;
}
.plot-heading h3 {
    margin: 0 0 5px;
}
.plot-heading p {
    color: #a1a1aa;
    margin: 0;
}
.plot-heading .equation {
    color: #d4d4d8;
    font-family: "STIX Two Text", Georgia, serif;
    font-size: 0.98rem;
}
.plot-panel .js-plotly-plot,
.plot-panel .plot-container {
    background: transparent !important;
}
"""

INSTALL_ANIMATION_JS = """
(frames) => {
    const install = (attempt = 0) => {
        const plot = document.querySelector("#profile-plot .js-plotly-plot");
        const plotly = window.Plotly;
        if (!plot || !plotly || !plot.data?.length) {
            if (attempt < 30) {
                window.setTimeout(() => install(attempt + 1), 35);
            }
            return;
        }
        const oldCount = plot._transitionData?._frames?.length ?? 0;
        const addFrames = () => plotly.addFrames(plot, frames ?? []);
        if (oldCount > 0) {
            const indices = Array.from({length: oldCount}, (_, index) => index);
            Promise.resolve(plotly.deleteFrames(plot, indices)).then(addFrames);
        } else {
            addFrames();
        }
    };
    window.requestAnimationFrame(() => install());
}
"""


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
        with gr.Column(elem_classes="plot-panel"):
            profile_heading = gr.Markdown(
                "### Evolving solution\nLoading equation…",
                elem_classes="plot-heading",
            )
            profile_output = gr.Plot(
                show_label=False,
                container=False,
                elem_id="profile-plot",
            )
        with gr.Column(elem_classes="plot-panel"):
            gr.Markdown(
                "### C PINN heat map\nNative surrogate over space and time.",
                elem_classes="plot-heading",
            )
            heatmap_output = gr.Plot(show_label=False, container=False)
        with gr.Column(elem_classes="plot-panel"):
            gr.Markdown(
                "### Absolute error\nPointwise difference from the exact solution.",
                elem_classes="plot-heading",
            )
            error_output = gr.Plot(show_label=False, container=False)
    metrics_output = gr.Markdown("Loading the default solution…")
    animation_frames_output = gr.JSON(visible=False)
    parameter_inputs = [a1_input, a2_input, alpha_input]
    plot_outputs = [
        profile_heading,
        profile_output,
        heatmap_output,
        error_output,
        metrics_output,
        animation_frames_output,
    ]
    predict_event = predict_button.click(
        _gradio_predict,
        inputs=parameter_inputs,
        outputs=plot_outputs,
        queue=False,
        show_progress="minimal",
    )
    predict_event.then(
        fn=None,
        inputs=animation_frames_output,
        outputs=None,
        js=INSTALL_ANIMATION_JS,
        queue=False,
    )
    load_event = demo.load(
        _gradio_predict,
        inputs=parameter_inputs,
        outputs=plot_outputs,
        queue=False,
        show_progress="minimal",
    )
    load_event.then(
        fn=None,
        inputs=animation_frames_output,
        outputs=None,
        js=INSTALL_ANIMATION_JS,
        queue=False,
    )
    for parameter_input in parameter_inputs:
        live_event = parameter_input.input(
            _gradio_predict,
            inputs=parameter_inputs,
            outputs=plot_outputs,
            queue=True,
            show_progress="hidden",
            trigger_mode="always_last",
            concurrency_limit=1,
            concurrency_id="live-parameter-update",
        )
        live_event.then(
            fn=None,
            inputs=animation_frames_output,
            outputs=None,
            js=INSTALL_ANIMATION_JS,
            queue=False,
        )
    gr.Examples(
        examples=[
            [1.0, 0.0, 0.10],
            [0.0, 1.0, 0.50],
            [0.7, -0.4, 0.05],
            [-0.6, 0.25, 0.20],
            [-1.0, 1.0, 0.50],
        ],
        inputs=parameter_inputs,
    )


app = gr.mount_gradio_app(
    app,
    demo,
    path="/",
    root_path=os.environ.get("GRADIO_ROOT_PATH"),
    css=APP_CSS,
)
