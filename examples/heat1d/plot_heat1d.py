from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


ALPHA = 0.1
T_START = 0.0
T_END = 1.0
X_START = 0.0
X_END = 1.0

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PRED_PATH = BASE_DIR / "files/heat1d_predictions.csv"
DEFAULT_LOSS_PATH = BASE_DIR / "files/heat1d_loss.csv"
DEFAULT_METRICS_PATH = BASE_DIR / "files/heat1d_metrics.csv"
DEFAULT_OUTPUT_PATH = BASE_DIR / "heat1d_results.png"


def analytical_solution(t: np.ndarray, x: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    return np.exp(-alpha * np.pi**2 * t) * np.sin(np.pi * x)


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing prediction CSV: {path}\n"
            "Expected columns: t,x,u_pred or t,x,u_pred,u_exact"
        )

    data = np.genfromtxt(path, delimiter=",", names=True)
    required = {"t", "x", "u_pred"}
    if data.dtype.names is None or not required.issubset(data.dtype.names):
        raise ValueError(f"{path} must contain columns: t,x,u_pred")

    return data["t"], data["x"], data["u_pred"]


def load_loss_history(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None

    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.dtype.names is None or "step" not in data.dtype.names:
        return None

    return {name: data[name] for name in data.dtype.names}


def load_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}

    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    if data.dtype.names is None or not {"metric", "value"}.issubset(data.dtype.names):
        return {}

    metrics = {}
    for metric, value in zip(np.atleast_1d(data["metric"]), np.atleast_1d(data["value"])):
        metrics[str(metric)] = float(value)
    return metrics


def grid_from_scattered(
    t: np.ndarray,
    x: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_unique = np.unique(t)
    x_unique = np.unique(x)
    z = np.full((t_unique.size, x_unique.size), np.nan, dtype=np.float64)

    t_index = {value: i for i, value in enumerate(t_unique)}
    x_index = {value: i for i, value in enumerate(x_unique)}
    for ti, xi, value in zip(t, x, values):
        z[t_index[ti], x_index[xi]] = value

    if np.isnan(z).any():
        raise ValueError(
            "Prediction CSV must contain a complete rectangular t-x grid "
            "for heatmap plotting."
        )

    return t_unique, x_unique, z


def solve_heat_fdm(
    x_grid: np.ndarray,
    t_eval: np.ndarray,
    alpha: float = ALPHA,
    safety: float = 0.45,
) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()

    nx = x_grid.size
    dx = x_grid[1] - x_grid[0]
    dt_stable = safety * dx * dx / alpha
    t_final = float(np.max(t_eval))
    n_steps = max(1, int(np.ceil(t_final / dt_stable)))
    dt = t_final / n_steps if n_steps > 0 else dt_stable
    r = alpha * dt / (dx * dx)

    u = np.sin(np.pi * x_grid)
    u[0] = 0.0
    u[-1] = 0.0

    fdm = np.empty((t_eval.size, nx), dtype=np.float64)
    eval_order = np.argsort(t_eval)
    next_eval = 0

    for n in range(n_steps + 1):
        current_t = n * dt
        while next_eval < t_eval.size and t_eval[eval_order[next_eval]] <= current_t + 0.5 * dt:
            fdm[eval_order[next_eval]] = u
            next_eval += 1

        if n == n_steps:
            break

        u_next = u.copy()
        u_next[1:-1] = u[1:-1] + r * (u[:-2] - 2.0 * u[1:-1] + u[2:])
        u_next[0] = 0.0
        u_next[-1] = 0.0
        u = u_next

    while next_eval < t_eval.size:
        fdm[eval_order[next_eval]] = u
        next_eval += 1

    elapsed = time.perf_counter() - t0
    return fdm, elapsed


def nearest_time_index(t_grid: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(t_grid - target)))


def plot_heat1d(
    pred_path: Path,
    loss_path: Path,
    metrics_path: Path,
    output_path: Path,
    alpha: float,
) -> None:
    t, x, u_pred_flat = load_predictions(pred_path)
    loss_history = load_loss_history(loss_path)
    metrics = load_metrics(metrics_path)

    t_grid, x_grid, u_pred = grid_from_scattered(t, x, u_pred_flat)
    tt, xx = np.meshgrid(t_grid, x_grid, indexing="ij")
    u_true = analytical_solution(tt, xx, alpha)
    err_pinn = np.abs(u_pred - u_true)

    u_fdm, fdm_time = solve_heat_fdm(x_grid, t_grid, alpha)
    err_fdm = np.abs(u_fdm - u_true)
    pinn_train_time = metrics.get("training_seconds")
    pinn_infer_time = metrics.get("pinn_inference_seconds")

    print(f"\n{'-' * 52}")
    print("  1D heat comparison")
    print(f"{'-' * 52}")
    print(f"  prediction file : {pred_path}")
    print(f"  grid            : nt={t_grid.size}, nx={x_grid.size}")
    print(f"  alpha           : {alpha:.6f}")
    if pinn_train_time is not None:
        print(f"  PINN training   : {pinn_train_time:8.3f} s")
    if pinn_infer_time is not None:
        print(f"  PINN inference  : {pinn_infer_time * 1000:8.3f} ms")
    print(f"  FDM solve       : {fdm_time * 1000:8.3f} ms")
    print(f"{'-' * 52}")
    print("  Error vs analytical  (max | mean)")
    print(f"  PINN            : {err_pinn.max():.3e} | {err_pinn.mean():.3e}")
    print(f"  FDM             : {err_fdm.max():.3e} | {err_fdm.mean():.3e}")
    print(f"{'-' * 52}\n")

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1.1, 1.1, 0.9])

    extent = [x_grid.min(), x_grid.max(), t_grid.min(), t_grid.max()]

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(u_pred, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    ax.set_title("PINN u(t,x)")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(u_true, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    ax.set_title("Analytical u(t,x)")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(err_pinn, origin="lower", aspect="auto", extent=extent, cmap="magma")
    ax.set_title("PINN Absolute Error")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[1, 1])
    for target_t in [0.0, 0.25, 0.5, 1.0]:
        idx = nearest_time_index(t_grid, target_t)
        label_t = t_grid[idx]
        ax.plot(x_grid, u_true[idx], color="steelblue", lw=2.0, alpha=0.9)
        ax.plot(x_grid, u_pred[idx], color="tab:orange", linestyle="--", lw=1.8)
        ax.plot(x_grid, u_fdm[idx], color="forestgreen", linestyle=":", lw=1.8)
        ax.text(
            x_grid[-1],
            u_true[idx, -1],
            f"t={label_t:.2f}",
            fontsize=8,
            ha="right",
            va="bottom",
        )
    ax.plot([], [], color="steelblue", lw=2.0, label="Analytical")
    ax.plot([], [], color="tab:orange", linestyle="--", lw=1.8, label="PINN")
    ax.plot([], [], color="forestgreen", linestyle=":", lw=1.8, label="FDM")
    ax.set_title("Time Slices")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    if loss_history and "total" in loss_history:
        ax.semilogy(loss_history["step"], np.maximum(loss_history["total"], 1e-16), label="total", color="black")
        for name, color in [("physics", "tab:purple"), ("ic", "tab:blue"), ("bc", "tab:green")]:
            if name in loss_history:
                ax.semilogy(loss_history["step"], np.maximum(loss_history[name], 1e-16), label=name, color=color)
        ax.set_title("Training Loss")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
    else:
        ax.axis("off")
        ax.text(0.05, 0.95, "No loss CSV found", transform=ax.transAxes, va="top")

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    summary_lines = [
        "1D heat comparison",
        "-" * 32,
        f"alpha          : {alpha:.6f}",
        f"t domain       : [{t_grid.min():.2f}, {t_grid.max():.2f}]",
        f"x domain       : [{x_grid.min():.2f}, {x_grid.max():.2f}]",
        f"grid           : {t_grid.size} x {x_grid.size}",
    ]
    if pinn_train_time is not None:
        summary_lines.append(f"PINN train     : {pinn_train_time:.3f} s")
    if pinn_infer_time is not None:
        summary_lines.append(f"PINN inference : {pinn_infer_time * 1000:.3f} ms")
    summary_lines.extend([
        f"FDM solve      : {fdm_time * 1000:.3f} ms",
        "",
        "Error vs analytical",
        "-" * 32,
        "          max err    mean err",
        f"PINN  : {err_pinn.max():.2e}  {err_pinn.mean():.2e}",
        f"FDM   : {err_fdm.max():.2e}  {err_fdm.mean():.2e}",
    ])
    summary = "\n".join(summary_lines)
    ax.text(
        0.04,
        0.96,
        summary,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    fig.suptitle("1D Heat Equation PINN", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"saved plot: {output_path}")
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot C PINN 1D heat results.")
    parser.add_argument("--pred", type=Path, default=DEFAULT_PRED_PATH)
    parser.add_argument("--loss", type=Path, default=DEFAULT_LOSS_PATH)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_heat1d(args.pred, args.loss, args.metrics, args.out, args.alpha)
