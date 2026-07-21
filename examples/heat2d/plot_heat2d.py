from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


ALPHA_X = 0.1
ALPHA_Y = 0.02

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PRED_PATH = BASE_DIR / "files/heat2d_predictions.csv"
DEFAULT_LOSS_PATH = BASE_DIR / "files/heat2d_loss.csv"
DEFAULT_METRICS_PATH = BASE_DIR / "files/heat2d_metrics.csv"
DEFAULT_OUTPUT_PATH = BASE_DIR / "anisotropic_heat2d_results.png"


def analytical_solution(
    t: np.ndarray, x: np.ndarray, y: np.ndarray, alpha_x: float, alpha_y: float
) -> np.ndarray:
    return (
        np.exp(-(alpha_x + alpha_y) * np.pi**2 * t)
        * np.sin(np.pi * x)
        * np.sin(np.pi * y)
        + 0.5
        * np.exp(-(4.0 * alpha_x + alpha_y) * np.pi**2 * t)
        * np.sin(2.0 * np.pi * x)
        * np.sin(np.pi * y)
    )


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction CSV: {path}")

    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    required = {"t", "x", "y", "u_pred"}
    if data.dtype.names is None or not required.issubset(data.dtype.names):
        raise ValueError(f"{path} must contain columns: t,x,y,u_pred")

    return data["t"], data["x"], data["y"], data["u_pred"]


def load_loss_history(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    if data.dtype.names is None or "step" not in data.dtype.names:
        return None
    return {name: np.atleast_1d(data[name]) for name in data.dtype.names}


def load_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    if data.dtype.names is None or not {"metric", "value"}.issubset(data.dtype.names):
        return {}
    return {
        str(metric): float(value)
        for metric, value in zip(np.atleast_1d(data["metric"]), np.atleast_1d(data["value"]))
    }


def grid_from_scattered(
    t: np.ndarray, x: np.ndarray, y: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t_grid = np.unique(t)
    x_grid = np.unique(x)
    y_grid = np.unique(y)
    field = np.full((t_grid.size, x_grid.size, y_grid.size), np.nan, dtype=np.float64)

    t_index = {value: i for i, value in enumerate(t_grid)}
    x_index = {value: i for i, value in enumerate(x_grid)}
    y_index = {value: i for i, value in enumerate(y_grid)}
    for ti, xi, yi, value in zip(t, x, y, values):
        field[t_index[ti], x_index[xi], y_index[yi]] = value

    if np.isnan(field).any():
        raise ValueError("Prediction CSV must contain a complete rectangular t-x-y grid.")
    return t_grid, x_grid, y_grid, field


def solve_heat_fdm(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_eval: np.ndarray,
    alpha_x: float,
    alpha_y: float,
    safety: float = 0.45,
) -> tuple[np.ndarray, float]:
    """Explicit finite-difference reference on the evaluation grid."""
    t0 = time.perf_counter()
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    dt_stable = safety / (alpha_x / dx**2 + alpha_y / dy**2)
    n_steps = max(1, int(np.ceil(float(t_eval[-1]) / dt_stable)))
    dt = float(t_eval[-1]) / n_steps
    rx = alpha_x * dt / dx**2
    ry = alpha_y * dt / dy**2

    xx, yy = np.meshgrid(x_grid, y_grid, indexing="ij")
    u = (
        np.sin(np.pi * xx) * np.sin(np.pi * yy)
        + 0.5 * np.sin(2.0 * np.pi * xx) * np.sin(np.pi * yy)
    )
    fdm = np.empty((t_eval.size, x_grid.size, y_grid.size), dtype=np.float64)
    next_eval = 0

    for step in range(n_steps + 1):
        current_t = step * dt
        while next_eval < t_eval.size and t_eval[next_eval] <= current_t + 0.5 * dt:
            fdm[next_eval] = u
            next_eval += 1
        if step == n_steps:
            break
        u_next = u.copy()
        u_next[1:-1, 1:-1] = (
            u[1:-1, 1:-1]
            + rx * (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1])
            + ry * (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2])
        )
        u = u_next

    while next_eval < t_eval.size:
        fdm[next_eval] = u
        next_eval += 1
    return fdm, time.perf_counter() - t0


def nearest_index(grid: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(grid - target)))


def plot_heat2d(
    pred_path: Path,
    loss_path: Path,
    metrics_path: Path,
    output_path: Path,
    alpha_x: float,
    alpha_y: float,
    slice_time: float,
) -> None:
    t, x, y, u_pred_flat = load_predictions(pred_path)
    t_grid, x_grid, y_grid, u_pred = grid_from_scattered(t, x, y, u_pred_flat)
    tt, xx, yy = np.meshgrid(t_grid, x_grid, y_grid, indexing="ij")
    u_true = analytical_solution(tt, xx, yy, alpha_x, alpha_y)
    err_pinn = np.abs(u_pred - u_true)
    u_fdm, fdm_time = solve_heat_fdm(x_grid, y_grid, t_grid, alpha_x, alpha_y)
    err_fdm = np.abs(u_fdm - u_true)
    loss_history = load_loss_history(loss_path)
    metrics = load_metrics(metrics_path)

    t_idx = nearest_index(t_grid, slice_time)
    y_idx = nearest_index(y_grid, 0.5)
    t_label = t_grid[t_idx]
    xx_slice, yy_slice = np.meshgrid(x_grid, y_grid, indexing="ij")

    print(f"\n{'-' * 56}")
    print("  2D heat comparison")
    print(f"{'-' * 56}")
    print(f"  grid            : nt={t_grid.size}, nx={x_grid.size}, ny={y_grid.size}")
    print(f"  alpha_x, alpha_y: {alpha_x:.6f}, {alpha_y:.6f}")
    print(f"  PINN error      : max={err_pinn.max():.3e}, mean={err_pinn.mean():.3e}")
    print(f"  FDM error       : max={err_fdm.max():.3e}, mean={err_fdm.mean():.3e}")
    if "training_seconds" in metrics:
        print(f"  PINN training   : {metrics['training_seconds']:.3f} s")
    if "pinn_inference_seconds" in metrics:
        print(f"  PINN inference  : {metrics['pinn_inference_seconds'] * 1000:.3f} ms")
    print(f"  FDM solve       : {fdm_time * 1000:.3f} ms")
    print(f"{'-' * 56}\n")

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1.0, 1.0, 0.9])

    for ax, field, title, cmap, z_label in [
        (fig.add_subplot(gs[0, 0], projection="3d"), u_pred[t_idx], f"PINN 2D spatial field, t={t_label:.2f}", "viridis", "u"),
        (fig.add_subplot(gs[0, 1], projection="3d"), u_true[t_idx], f"Analytical 2D spatial field, t={t_label:.2f}", "viridis", "u"),
        (fig.add_subplot(gs[1, 0], projection="3d"), err_pinn[t_idx], "2D spatial absolute error", "magma", "|error|"),
    ]:
        surface = ax.plot_surface(xx_slice, yy_slice, field, cmap=cmap, linewidth=0, antialiased=True)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel(z_label)
        ax.view_init(elev=30, azim=-135)
        fig.colorbar(surface, ax=ax, fraction=0.046, pad=0.08)

    ax = fig.add_subplot(gs[1, 1])
    for target_t in [0.0, 0.25, 0.5, 1.0]:
        idx = nearest_index(t_grid, target_t)
        ax.plot(x_grid, u_true[idx, :, y_idx], lw=2.0, label=f"exact t={t_grid[idx]:.2f}")
        ax.plot(x_grid, u_pred[idx, :, y_idx], "--", lw=1.6, color=ax.lines[-1].get_color())
    ax.plot([], [], color="black", lw=2.0, label="analytical")
    ax.plot([], [], color="black", linestyle="--", lw=1.6, label="PINN")
    ax.set_title(f"Centerline slices at y={y_grid[y_idx]:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    ax = fig.add_subplot(gs[0, 2])
    if loss_history and "total" in loss_history:
        steps = loss_history["step"].astype(float)
        phases = loss_history.get("phase")
        if phases is not None and np.any(phases == "adam"):
            adam_mask = phases == "adam"
            ax.semilogy(
                steps[adam_mask],
                np.maximum(loss_history["physics"][adam_mask], 1e-16),
                color="tab:purple",
                label="Adam physics",
            )
            if np.any(phases == "lbfgs"):
                lbfgs_mask = phases == "lbfgs"
                lbfgs_steps = steps[lbfgs_mask] + steps[adam_mask].max()
                ax.semilogy(
                    lbfgs_steps,
                    np.maximum(loss_history["physics"][lbfgs_mask], 1e-16),
                    color="tab:orange",
                    marker="o",
                    label="L-BFGS physics",
                )
                ax.axvline(steps[adam_mask].max(), color="gray", linestyle="--", linewidth=1.0)
        else:
            ax.semilogy(steps, np.maximum(loss_history["physics"], 1e-16), color="tab:purple", label="physics")
        ax.set_title("Training loss")
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.axis("off")

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    summary = [
        "2D heat comparison",
        "-" * 30,
        f"grid       : {t_grid.size} x {x_grid.size} x {y_grid.size}",
        f"alpha_x   : {alpha_x:.4f}",
        f"alpha_y   : {alpha_y:.4f}",
        "",
        "Error vs analytical",
        f"PINN max  : {err_pinn.max():.2e}",
        f"PINN mean : {err_pinn.mean():.2e}",
        f"FDM max   : {err_fdm.max():.2e}",
        f"FDM mean  : {err_fdm.mean():.2e}",
    ]
    if "training_seconds" in metrics:
        summary.append(f"PINN train : {metrics['training_seconds']:.3f} s")
    if "pinn_inference_seconds" in metrics:
        summary.append(f"PINN infer : {metrics['pinn_inference_seconds'] * 1000:.3f} ms")
    summary.append(f"FDM solve  : {fdm_time * 1000:.3f} ms")
    ax.text(0.04, 0.96, "\n".join(summary), transform=ax.transAxes, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    fig.suptitle("2D Heat Equation PINN", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"saved plot: {output_path}")
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot C PINN 2D heat results.")
    parser.add_argument("--pred", type=Path, default=DEFAULT_PRED_PATH)
    parser.add_argument("--loss", type=Path, default=DEFAULT_LOSS_PATH)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--alpha-x", type=float, default=ALPHA_X)
    parser.add_argument("--alpha-y", type=float, default=ALPHA_Y)
    parser.add_argument("--time", type=float, default=0.5, help="Time slice to show in the heatmaps.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_heat2d(args.pred, args.loss, args.metrics, args.out, args.alpha_x, args.alpha_y, args.time)
