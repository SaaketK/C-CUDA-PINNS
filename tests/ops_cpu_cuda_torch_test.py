#!/usr/bin/env python3
"""Build and run the native CPU/CUDA autograd test, then check PyTorch parity."""

from __future__ import annotations

import pathlib
import subprocess
import sys

import torch


ROOT = pathlib.Path(__file__).resolve().parents[2]
BUILD = pathlib.Path("/tmp/pinn-ops-parity")
BINARY = BUILD / "ops-backward-parity-test"
FORWARD_BINARY = BUILD / "ops-forward-test"


def build_native_test(test_source: str, binary: pathlib.Path) -> None:
    BUILD.mkdir(exist_ok=True)
    c_sources = [
        test_source,
        "src/core/tensor.c",
        "src/core/ops_cpu.c",
        "src/core/ops.c",
        "src/core/autograd.c",
        "src/core/backend.c",
    ]
    objects = []
    for source in c_sources:
        output = BUILD / (pathlib.Path(source).stem + ".o")
        subprocess.run(
            ["gcc", "-std=c11", "-O2", "-DPINN_USE_CUDA", "-Iinclude", "-fopenmp", "-c", source, "-o", str(output)],
            cwd=ROOT,
            check=True,
        )
        objects.append(str(output))
    cuda_object = BUILD / "ops_cuda.o"
    subprocess.run(
        ["nvcc", "-std=c++14", "-O2", "-Iinclude", "-c", "src/core/ops_cuda.cu", "-o", str(cuda_object)],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        ["nvcc", "-Xcompiler", "-fopenmp", *objects, str(cuda_object), "-lcublas", "-o", str(binary)],
        cwd=ROOT,
        check=True,
    )


def native_values() -> dict[str, torch.Tensor]:
    completed = subprocess.run([str(BINARY)], cwd=ROOT, text=True, capture_output=True)
    print(completed.stdout, end="")
    if completed.returncode:
        print(completed.stderr, file=sys.stderr, end="")
        raise RuntimeError("native CPU/CUDA parity test failed")
    values = {}
    for line in completed.stdout.splitlines():
        fields = line.split(",")
        if len(fields) == 23 and fields[0] == "BASIC":
            values[fields[1]] = torch.tensor([float(value) for value in fields[2:]])
    if set(values) != {"CPU", "CUDA"}:
        raise RuntimeError("native test did not produce both CPU and CUDA result rows")
    return values


def run_forward_suite() -> None:
    completed = subprocess.run([str(FORWARD_BINARY)], cwd=ROOT, text=True, capture_output=True)
    print(completed.stdout, end="")
    if completed.returncode:
        print(completed.stderr, file=sys.stderr, end="")
        raise RuntimeError("native CPU/CUDA forward suite failed")


def pytorch_values() -> torch.Tensor:
    x = torch.tensor([[-1.0, -0.5, 0.25], [0.75, 1.25, 1.5]], requires_grad=True)
    y = torch.tensor([[0.5, -1.5, 2.0], [-0.25, 0.75, -1.0]], requires_grad=True)
    w = torch.tensor([[0.2, -0.3], [0.4, 0.1], [-0.5, 0.6]], requires_grad=True)
    b = torch.tensor([0.15, -0.2], requires_grad=True)
    target = torch.tensor([[0.1, 0.9], [0.3, 0.7]])
    prediction = torch.sigmoid(torch.relu(torch.tanh(x * y) @ w + b))
    loss = torch.mean((prediction - target) ** 2)
    loss.backward()
    return torch.cat((loss.detach().reshape(-1), x.grad.reshape(-1), y.grad.reshape(-1), w.grad.reshape(-1), b.grad.reshape(-1)))


def check(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    error = torch.max(torch.abs(actual - expected)).item()
    print(f"{name} vs PyTorch max_abs_error={error:.3e}")
    if error > 2e-4:
        raise AssertionError(f"{name} differs from PyTorch by {error:.3e}")


def main() -> None:
    build_native_test("src/tests/ops-forward-test.c", FORWARD_BINARY)
    build_native_test("src/tests/ops-backward-parity-test.c", BINARY)
    run_forward_suite()
    native = native_values()
    reference = pytorch_values()
    check("CPU", native["CPU"], reference)
    check("CUDA", native["CUDA"], reference)
    print("CPU vs CUDA vs PyTorch autograd comparison passed")


if __name__ == "__main__":
    main()
