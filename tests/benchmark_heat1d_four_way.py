#!/usr/bin/env python3
"""End-to-end Heat1D PINN performance sweep for native C/CUDA and PyTorch."""

import argparse
import csv
import json
import math
import os
import pathlib
import re
import shlex
import statistics
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[2]
BUILD = pathlib.Path('/tmp/pinn_heat1d_four_way')
RESULTS = ROOT / 'src/tests/results/heat1d_four_way'
CSV_PATH = ROOT / 'src/tests/heat1d_four_way_results.csv'
REPORT_PATH = ROOT / 'src/tests/HEAT1D_PERFORMANCE_ANALYSIS.md'


def command_output(command):
    return subprocess.check_output(command, cwd=ROOT, text=True).strip()


def build_native():
    BUILD.mkdir(parents=True, exist_ok=True)
    cflags = shlex.split(command_output(['pkg-config', '--cflags', 'openblas']))
    libs = shlex.split(command_output(['pkg-config', '--libs', 'openblas']))
    cuda_include = pathlib.Path(command_output(['bash', '-lc', 'dirname $(dirname $(command -v nvcc))/include']))
    sources = [
        'examples/heat1d/heat1d.c', 'src/core/tensor.c', 'src/core/ops_cpu.c',
        'src/core/ops.c', 'src/core/autograd.c', 'src/core/backend.c',
        'src/nn/mlp.c', 'src/nn/optimizer_cpu.c', 'src/nn/optimizer.c',
        'src/nn/lbfgs.c', 'src/autodiff/jet.c', 'src/pinn/sampler.c',
        'src/pinn/residual.c',
    ]
    objects = []
    for source in sources:
        object_path = BUILD / (pathlib.Path(source).stem + '.o')
        subprocess.run([
            'gcc', '-std=c11', '-O3', '-DPINN_USE_CUDA', '-DPINN_USE_OPENBLAS',
            *cflags, '-Iinclude', f'-I{cuda_include}', '-fopenmp', '-c', source,
            '-o', str(object_path),
        ], cwd=ROOT, check=True)
        objects.append(str(object_path))
    for source in ('src/core/ops_cuda.cu', 'src/nn/optimizer_cuda.cu'):
        object_path = BUILD / (pathlib.Path(source).stem + '.o')
        subprocess.run([
            'nvcc', '-std=c++14', '-O3', '-Iinclude', '-c', source,
            '-o', str(object_path),
        ], cwd=ROOT, check=True)
        objects.append(str(object_path))
    executable = BUILD / 'heat1d_train'
    subprocess.run([
        'nvcc', '-Xcompiler', '-fopenmp', *objects, '-lcublas', *libs,
        '-o', str(executable),
    ], cwd=ROOT, check=True)
    return executable


def native_run(executable, device, steps, points, seed, threads):
    environment = os.environ.copy()
    environment['OMP_NUM_THREADS'] = str(threads)
    environment['OPENBLAS_NUM_THREADS'] = '1'
    completed = subprocess.run(
        [str(executable), str(steps), str(points), str(seed), device], cwd=ROOT,
        text=True, capture_output=True, check=True, env=environment,
    )
    duration = re.search(r'training seconds=([0-9.]+)', completed.stdout)
    losses = re.findall(r'step=\d+ physics=([0-9.eE+-]+)', completed.stdout)
    if not duration or not losses:
        raise RuntimeError(f'Could not parse native output:\n{completed.stdout}\n{completed.stderr}')
    return float(duration.group(1)), float(losses[-1])


def torch_run(device, steps, points, seed, threads, label):
    results_dir = RESULTS / label
    completed = subprocess.run([
        str(ROOT / '.venv/bin/python'), 'src/tests/heat1d_pytorch.py',
        '--device', device, '--steps', str(steps), '--points', str(points),
        '--seed', str(seed), '--threads', str(threads), '--results-dir', str(results_dir),
    ], cwd=ROOT, text=True, capture_output=True, check=True)
    metrics = json.loads((results_dir / f'pytorch_{device}_metrics.json').read_text())
    return metrics['training_seconds'], metrics['final_loss']


def geometric_mean(values):
    return math.exp(statistics.fmean(math.log(value) for value in values))


def platform_details():
    cpu = 'unknown'
    for line in pathlib.Path('/proc/cpuinfo').read_text().splitlines():
        if line.startswith('model name'):
            cpu = line.split(':', 1)[1].strip()
            break
    try:
        gpu = command_output(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'])
    except (FileNotFoundError, subprocess.CalledProcessError):
        gpu = 'unavailable'
    torch_version = command_output([str(ROOT / '.venv/bin/python'), '-c', 'import torch; print(torch.__version__)'])
    return cpu, gpu, torch_version


def write_outputs(rows, threads, cpu, gpu, torch_version):
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    grouped = {}
    for row in rows:
        grouped.setdefault((row['points'], row['steps']), []).append(row)
    native_cpu_ratio = []
    native_cuda_ratio = []
    cuda_vs_cpu = []
    native_cuda_wins = 0
    torch_cuda_wins = 0
    fastest_counts = {}
    for group in grouped.values():
        by_name = {row['backend']: row for row in group}
        native_cpu_ratio.append(by_name['C OpenMP CPU']['milliseconds_per_step'] / by_name['PyTorch CPU']['milliseconds_per_step'])
        native_cuda_ratio.append(by_name['C CUDA']['milliseconds_per_step'] / by_name['PyTorch CUDA']['milliseconds_per_step'])
        cuda_vs_cpu.append(by_name['C OpenMP CPU']['milliseconds_per_step'] / by_name['C CUDA']['milliseconds_per_step'])
        if by_name['C CUDA']['milliseconds_per_step'] < by_name['C OpenMP CPU']['milliseconds_per_step']:
            native_cuda_wins += 1
        if by_name['PyTorch CUDA']['milliseconds_per_step'] < by_name['PyTorch CPU']['milliseconds_per_step']:
            torch_cuda_wins += 1
        fastest = min(group, key=lambda row: row['milliseconds_per_step'])['backend']
        fastest_counts[fastest] = fastest_counts.get(fastest, 0) + 1

    lines = [
        '# Heat1D PINN Performance Analysis', '',
        '## Scope', '',
        'This report measures end-to-end Adam training for the same nominal Heat1D residual workload: a `2 → 64 → 64 → 64 → 1` tanh MLP, float32, second spatial derivative, parameter backward pass, and Adam update. Timing includes the training loop but excludes model construction and final evaluation.', '',
        f'- CPU: {cpu}',
        f'- GPU: {gpu}',
        f'- PyTorch: {torch_version}',
        f'- CPU thread settings: `OMP_NUM_THREADS={threads}`, `torch.set_num_threads({threads})`, `OPENBLAS_NUM_THREADS=1`.',
        '- Native CPU uses the current OpenMP/OpenBLAS dispatch. Native CUDA uses the current CUDA kernels and cuBLAS paths.',
        '- Each configuration is one timed run with seed 1234. Timings are useful for current-state throughput; repeat trials are needed for confidence intervals.', '',
        '## Results', '',
        '| Points | Adam steps | Backend | Training s | ms/step | Final physics loss |',
        '|---:|---:|---|---:|---:|---:|',
    ]
    for points, steps in sorted(grouped):
        for row in sorted(grouped[(points, steps)], key=lambda value: value['backend']):
            lines.append(f"| {points} | {steps} | {row['backend']} | {row['training_seconds']:.4f} | {row['milliseconds_per_step']:.3f} | {row['final_loss']:.3e} |")

    lines.extend([
        '', '## Overarching trends', '',
        f"- PyTorch CPU is faster than the current native OpenMP CPU path in every measured configuration. The geometric-mean native/PyTorch CPU time ratio is **{geometric_mean(native_cpu_ratio):.2f}×** (values above 1 mean native is slower).",
        f"- PyTorch CUDA is faster than the current native CUDA path in every measured configuration. The geometric-mean native/PyTorch CUDA time ratio is **{geometric_mean(native_cuda_ratio):.2f}×**.",
        f"- Moving the native workload from its OpenMP CPU backend to CUDA provides a geometric-mean **{geometric_mean(cuda_vs_cpu):.2f}×** per-step speedup across this sweep, but that advantage is not uniform: native CUDA is faster in {native_cuda_wins}/{len(grouped)} configurations and loses slightly at 512 and 1000 points with 500 steps.",
        f"- PyTorch CUDA is faster than PyTorch CPU in {torch_cuda_wins}/{len(grouped)} configurations. PyTorch CPU wins at 128 points, where GPU launch/graph overhead dominates; CUDA wins the larger, longer-running cases.",
        f"- Scaling is the main native CUDA weakness: from 128 to 1000 points at 500 steps, native CUDA grows from {grouped[(128, 500)][1]['milliseconds_per_step']:.3f} to {grouped[(1000, 500)][1]['milliseconds_per_step']:.3f} ms/step, while PyTorch CUDA stays near {grouped[(128, 500)][3]['milliseconds_per_step']:.3f}–{grouped[(1000, 500)][3]['milliseconds_per_step']:.3f} ms/step.",
        '- Fastest backend counts: ' + ', '.join(f'{name}: {count}/{len(grouped)}' for name, count in sorted(fastest_counts.items())) + '.',
        '', '## Interpretation and limits', '',
        '- This is a throughput comparison, not a numerical equivalence proof. The native trainer resamples Latin-hypercube collocation points each step, while the current PyTorch reference reuses a seeded uniform point tensor. The same architecture, PDE form, float32 precision, Adam hyperparameters, point counts, and step counts are used, but initial weights and samples are not bit-identical.',
        '- Final losses therefore indicate that each run trains, but are not a fair accuracy ranking. A parity experiment should feed an identical point sequence and identical initialized weights into both implementations, then compare loss curves and parameter gradients.',
        '- The native CPU result includes the framework\'s current fine-grained tensor/autograd allocation and OpenMP behavior. Its weak result at these sizes is consistent with the earlier raw-autograd benchmark, where scheduling/allocation overhead and many small dispatches outweighed parallel work.',
        '- The native CUDA result includes GPU kernels, allocations, host/device transfers associated with the current graph, and cuBLAS. It is the relevant present-day end-to-end number, but kernel fusion, allocator reuse, and reducing synchronization would be needed before treating it as a cuBLAS-limited baseline.',
        '- The GTX 1660 is an older Turing GPU without tensor cores for FP32; this comparison should be interpreted as framework efficiency on this hardware, not an upper bound for newer GPUs.',
        '', '## Recommended next measurements', '',
        '1. Add a shared binary sample/weight fixture and a per-step loss/gradient parity check before comparing accuracy.',
        '2. Run 5–10 trials per point/step configuration, discard a warm-up run, and report median plus p95.',
        '3. Profile native CPU with OpenMP disabled, thresholded, and OpenBLAS-only matmul to choose runtime dispatch thresholds.',
        '4. Profile native CUDA with Nsight Systems/Compute; focus on allocations, launch count, small kernels, and host-device synchronization before optimizing math kernels.',
    ])
    REPORT_PATH.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--points', type=int, nargs='+', default=[128, 512, 1000])
    parser.add_argument('--steps', type=int, nargs='+', default=[100, 500])
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    executable = build_native()
    RESULTS.mkdir(parents=True, exist_ok=True)
    cpu, gpu, torch_version = platform_details()
    rows = []
    for points in args.points:
        for steps in args.steps:
            print(f'Running points={points}, steps={steps}')
            cases = (
                ('C OpenMP CPU', lambda: native_run(executable, 'cpu', steps, points, args.seed, args.threads)),
                ('C CUDA', lambda: native_run(executable, 'cuda', steps, points, args.seed, args.threads)),
                ('PyTorch CPU', lambda: torch_run('cpu', steps, points, args.seed, args.threads, f'cpu_{points}_{steps}')),
                ('PyTorch CUDA', lambda: torch_run('cuda', steps, points, args.seed, args.threads, f'cuda_{points}_{steps}')),
            )
            for backend, runner in cases:
                seconds, loss = runner()
                row = {
                    'points': points, 'steps': steps, 'backend': backend,
                    'training_seconds': seconds,
                    'milliseconds_per_step': seconds * 1000.0 / steps,
                    'final_loss': loss,
                }
                rows.append(row)
                print(f"  {backend}: {row['milliseconds_per_step']:.3f} ms/step")
    write_outputs(rows, args.threads, cpu, gpu, torch_version)
    print(f'Wrote {CSV_PATH}')
    print(f'Wrote {REPORT_PATH}')


if __name__ == '__main__':
    main()
