from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


R = 0.05
K = 100.0
T = 1.0
SIGMA = 0.2
S_MAX = 2.0 * K

BASE_DIR = Path(__file__).resolve().parent
FILES_DIR = BASE_DIR
DEFAULT_PRED_PATH = FILES_DIR / "blackscholes1d_predictions.csv"
DEFAULT_LOSS_PATH = FILES_DIR / "blackscholes1d_loss.csv"
DEFAULT_METRICS_PATH = FILES_DIR / "blackscholes1d_metrics.csv"
DEFAULT_OUTPUT_PATH = FILES_DIR / "blackscholes1d_results.png"


def normal_cdf(x: np.ndarray) -> np.ndarray:
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(x / np.sqrt(2.0)))


def analytical_solution(
    t: np.ndarray,
    S: np.ndarray,
    r: float = R,
    K_: float = K,
    T_: float = T,
    sigma: float = SIGMA,
) -> np.ndarray:
    t_arr = np.asarray(t, dtype=np.float64)
    S_arr = np.asarray(S, dtype=np.float64)
    tau = np.maximum(T_ - t_arr, 0.0)
    payoff = np.maximum(S_arr - K_, 0.0)

    out = np.zeros(np.broadcast(t_arr, S_arr).shape, dtype=np.float64)
    active = (tau > 0.0) & (S_arr > 0.0)
    out[~active] = payoff[~active]

    sqrt_tau = np.sqrt(tau[active])
    d1 = (np.log(S_arr[active] / K_) + (r + 0.5 * sigma * sigma) * tau[active]) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    out[active] = S_arr[active] * normal_cdf(d1) - K_ * np.exp(-r * tau[active]) * normal_cdf(d2)
    return out


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing prediction CSV: {path}\n"
            "Run ./build/black_scholes1d_train from C-CUDA-PINNs first."
        )

    data = np.genfromtxt(path, delimiter=",", names=True)
    required = {"t", "S", "V_pred"}
    if data.dtype.names is None or not required.issubset(data.dtype.names):
        raise ValueError(f"{path} must contain columns: t,S,V_pred")

    return data["t"], data["S"], data["V_pred"]


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
    S: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_unique = np.unique(t)
    S_unique = np.unique(S)
    z = np.full((t_unique.size, S_unique.size), np.nan, dtype=np.float64)

    t_index = {value: i for i, value in enumerate(t_unique)}
    S_index = {value: i for i, value in enumerate(S_unique)}
    for ti, Si, value in zip(t, S, values):
        z[t_index[ti], S_index[Si]] = value

    if np.isnan(z).any():
        raise ValueError("Prediction CSV must contain a complete rectangular t-S grid.")

    return t_unique, S_unique, z


def solve_black_scholes_fdm(
    S_grid: np.ndarray,
    t_eval: np.ndarray,
    r: float,
    K_: float,
    T_: float,
    sigma: float,
    safety: float = 0.35,
) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()

    nS = S_grid.size
    dS = S_grid[1] - S_grid[0]
    S_max = float(S_grid[-1])
    tau_eval = T_ - t_eval
    tau_final = float(np.max(tau_eval))

    max_S = max(float(np.max(S_grid)), 1.0)
    diffusion_dt = dS * dS / (sigma * sigma * max_S * max_S + 1e-12)
    drift_dt = dS / (r * max_S + 1e-12)
    dt_stable = safety * min(diffusion_dt, drift_dt)
    n_steps = max(1, int(np.ceil(tau_final / dt_stable)))
    dt = tau_final / n_steps if n_steps > 0 else dt_stable

    V = np.maximum(S_grid - K_, 0.0)
    V[0] = 0.0
    V[-1] = S_max - K_

    fdm_tau = np.empty((t_eval.size, nS), dtype=np.float64)
    eval_order = np.argsort(tau_eval)
    next_eval = 0

    for n in range(n_steps + 1):
        tau = n * dt
        while next_eval < t_eval.size and tau_eval[eval_order[next_eval]] <= tau + 0.5 * dt:
            fdm_tau[eval_order[next_eval]] = V
            next_eval += 1

        if n == n_steps:
            break

        tau_next = (n + 1) * dt
        V_next = V.copy()
        Si = S_grid[1:-1]
        V_SS = (V[2:] - 2.0 * V[1:-1] + V[:-2]) / (dS * dS)
        V_S = (V[2:] - V[:-2]) / (2.0 * dS)
        V_next[1:-1] = V[1:-1] + dt * (
            0.5 * sigma * sigma * Si * Si * V_SS + r * Si * V_S - r * V[1:-1]
        )
        V_next[0] = 0.0
        V_next[-1] = S_max - K_ * np.exp(-r * tau_next)
        V = V_next

    while next_eval < t_eval.size:
        fdm_tau[eval_order[next_eval]] = V
        next_eval += 1

    elapsed = time.perf_counter() - t0
    return fdm_tau, elapsed


def nearest_time_index(t_grid: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(t_grid - target)))


def plot_blackscholes1d(
    pred_path: Path,
    loss_path: Path,
    metrics_path: Path,
    output_path: Path,
    r: float,
    K_: float,
    T_: float,
    sigma: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t, S, V_pred_flat = load_predictions(pred_path)
    loss_history = load_loss_history(loss_path)
    metrics = load_metrics(metrics_path)

    r = metrics.get("rate", r)
    K_ = metrics.get("strike", K_)
    T_ = metrics.get("maturity", T_)
    sigma = metrics.get("sigma", sigma)

    t_grid, S_grid, V_pred = grid_from_scattered(t, S, V_pred_flat)
    tt, ss = np.meshgrid(t_grid, S_grid, indexing="ij")
    V_true = analytical_solution(tt, ss, r, K_, T_, sigma)
    err_pinn = np.abs(V_pred - V_true)

    V_fdm, fdm_time = solve_black_scholes_fdm(S_grid, t_grid, r, K_, T_, sigma)
    err_fdm = np.abs(V_fdm - V_true)
    pinn_train_time = metrics.get("training_seconds")
    pinn_infer_time = metrics.get("pinn_inference_seconds")

    print(f"\n{'-' * 58}")
    print("  1D Black-Scholes comparison")
    print(f"{'-' * 58}")
    print(f"  prediction file : {pred_path}")
    print(f"  grid            : nt={t_grid.size}, nS={S_grid.size}")
    print(f"  r, K, T, sigma  : {r:.4f}, {K_:.4f}, {T_:.4f}, {sigma:.4f}")
    if pinn_train_time is not None:
        print(f"  PINN training   : {pinn_train_time:8.3f} s")
    if pinn_infer_time is not None:
        print(f"  PINN inference  : {pinn_infer_time * 1000:8.3f} ms")
    print(f"  FDM solve       : {fdm_time * 1000:8.3f} ms")
    print(f"{'-' * 58}")
    print("  Error vs analytical  (max | mean)")
    print(f"  PINN            : {err_pinn.max():.3e} | {err_pinn.mean():.3e}")
    print(f"  FDM             : {err_fdm.max():.3e} | {err_fdm.mean():.3e}")
    print(f"{'-' * 58}\n")

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1.1, 1.1, 0.9])

    extent = [S_grid.min(), S_grid.max(), t_grid.min(), t_grid.max()]

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(V_pred, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    ax.set_title("PINN V(t,S)")
    ax.set_xlabel("S")
    ax.set_ylabel("t")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(V_true, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    ax.set_title("Analytical V(t,S)")
    ax.set_xlabel("S")
    ax.set_ylabel("t")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(err_pinn, origin="lower", aspect="auto", extent=extent, cmap="magma")
    ax.set_title("PINN Absolute Error")
    ax.set_xlabel("S")
    ax.set_ylabel("t")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[1, 1])
    for target_t in [0.0, 0.25 * T_, 0.5 * T_, T_]:
        idx = nearest_time_index(t_grid, target_t)
        label_t = t_grid[idx]
        ax.plot(S_grid, V_true[idx], color="steelblue", lw=2.0, alpha=0.9)
        ax.plot(S_grid, V_pred[idx], color="tab:orange", linestyle="--", lw=1.8)
        ax.plot(S_grid, V_fdm[idx], color="forestgreen", linestyle=":", lw=1.8)
        ax.text(S_grid[-1], V_true[idx, -1], f"t={label_t:.2f}", fontsize=8, ha="right", va="bottom")
    ax.plot([], [], color="steelblue", lw=2.0, label="Analytical")
    ax.plot([], [], color="tab:orange", linestyle="--", lw=1.8, label="PINN")
    ax.plot([], [], color="forestgreen", linestyle=":", lw=1.8, label="FDM")
    ax.set_title("Time Slices")
    ax.set_xlabel("S")
    ax.set_ylabel("V")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    if loss_history and "total" in loss_history:
        ax.semilogy(loss_history["step"], np.maximum(loss_history["total"], 1e-16), label="total", color="black")
        for name, color in [("physics", "tab:purple"), ("terminal", "tab:blue"), ("bc", "tab:green")]:
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
        "1D Black-Scholes comparison",
        "-" * 34,
        f"r              : {r:.6f}",
        f"K              : {K_:.6f}",
        f"T              : {T_:.6f}",
        f"sigma          : {sigma:.6f}",
        f"S domain       : [{S_grid.min():.2f}, {S_grid.max():.2f}]",
        f"grid           : {t_grid.size} x {S_grid.size}",
    ]
    if pinn_train_time is not None:
        summary_lines.append(f"PINN train     : {pinn_train_time:.3f} s")
    if pinn_infer_time is not None:
        summary_lines.append(f"PINN inference : {pinn_infer_time * 1000:.3f} ms")
    summary_lines.extend([
        f"FDM solve      : {fdm_time * 1000:.3f} ms",
        "",
        "Error vs analytical",
        "-" * 34,
        "          max err    mean err",
        f"PINN  : {err_pinn.max():.2e}  {err_pinn.mean():.2e}",
        f"FDM   : {err_fdm.max():.2e}  {err_fdm.mean():.2e}",
    ])
    ax.text(
        0.04,
        0.96,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    fig.suptitle("1D Black-Scholes PINN", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"saved plot: {output_path}")
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot C PINN 1D Black-Scholes results.")
    parser.add_argument("--pred", type=Path, default=DEFAULT_PRED_PATH)
    parser.add_argument("--loss", type=Path, default=DEFAULT_LOSS_PATH)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--r", type=float, default=R)
    parser.add_argument("--K", type=float, default=K)
    parser.add_argument("--T", type=float, default=T)
    parser.add_argument("--sigma", type=float, default=SIGMA)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_blackscholes1d(args.pred, args.loss, args.metrics, args.out, args.r, args.K, args.T, args.sigma)
