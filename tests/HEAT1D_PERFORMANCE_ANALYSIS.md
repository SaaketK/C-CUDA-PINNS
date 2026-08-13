# Heat1D PINN Performance Analysis

## Scope

This report measures end-to-end Adam training for the same nominal Heat1D residual workload: a `2 → 64 → 64 → 64 → 1` tanh MLP, float32, second spatial derivative, parameter backward pass, and Adam update. Timing includes the training loop but excludes model construction and final evaluation.

- CPU: AMD Ryzen 7 3700X 8-Core Processor
- GPU: NVIDIA GeForce GTX 1660, 580.173.02
- PyTorch: 2.5.1+cu121
- CPU thread settings: `OMP_NUM_THREADS=16`, `torch.set_num_threads(16)`, `OPENBLAS_NUM_THREADS=1`.
- Native CPU uses the current OpenMP/OpenBLAS dispatch. Native CUDA uses the current CUDA kernels and cuBLAS paths.
- Each configuration is one timed run with seed 1234. Timings are useful for current-state throughput; repeat trials are needed for confidence intervals.

## Results

| Points | Adam steps | Backend | Training s | ms/step | Final physics loss |
|---:|---:|---|---:|---:|---:|
| 128 | 100 | C CUDA | 0.7169 | 7.169 | 9.330e-04 |
| 128 | 100 | C OpenMP CPU | 1.5625 | 15.625 | 9.330e-04 |
| 128 | 100 | PyTorch CPU | 0.4978 | 4.978 | 3.127e-03 |
| 128 | 100 | PyTorch CUDA | 0.7366 | 7.366 | 2.330e-03 |
| 128 | 500 | C CUDA | 3.3263 | 6.653 | 1.040e-04 |
| 128 | 500 | C OpenMP CPU | 5.6824 | 11.365 | 1.040e-04 |
| 128 | 500 | PyTorch CPU | 2.1855 | 4.371 | 6.506e-05 |
| 128 | 500 | PyTorch CUDA | 2.7598 | 5.520 | 7.588e-05 |
| 512 | 100 | C CUDA | 2.1225 | 21.225 | 1.029e-03 |
| 512 | 100 | C OpenMP CPU | 4.6418 | 46.418 | 1.029e-03 |
| 512 | 100 | PyTorch CPU | 0.5744 | 5.744 | 2.857e-03 |
| 512 | 100 | PyTorch CUDA | 0.7274 | 7.274 | 3.021e-03 |
| 512 | 500 | C CUDA | 10.7289 | 21.458 | 9.300e-05 |
| 512 | 500 | C OpenMP CPU | 10.2264 | 20.453 | 9.300e-05 |
| 512 | 500 | PyTorch CPU | 3.1751 | 6.350 | 6.825e-05 |
| 512 | 500 | PyTorch CUDA | 2.8085 | 5.617 | 7.909e-05 |
| 1000 | 100 | C CUDA | 4.0254 | 40.254 | 1.081e-03 |
| 1000 | 100 | C OpenMP CPU | 5.9015 | 59.015 | 1.081e-03 |
| 1000 | 100 | PyTorch CPU | 0.9533 | 9.533 | 3.186e-03 |
| 1000 | 100 | PyTorch CUDA | 0.7761 | 7.761 | 3.196e-03 |
| 1000 | 500 | C CUDA | 19.9292 | 39.858 | 1.040e-04 |
| 1000 | 500 | C OpenMP CPU | 19.3817 | 38.763 | 1.040e-04 |
| 1000 | 500 | PyTorch CPU | 4.4773 | 8.955 | 7.450e-05 |
| 1000 | 500 | PyTorch CUDA | 2.7743 | 5.549 | 8.014e-05 |

## Overarching trends

- PyTorch CPU is faster than the current native OpenMP CPU path in every measured configuration. The geometric-mean native/PyTorch CPU time ratio is **4.23×** (values above 1 mean native is slower).
- PyTorch CUDA is faster than the current native CUDA path in every measured configuration. The geometric-mean native/PyTorch CUDA time ratio is **2.81×**.
- Moving the native workload from its OpenMP CPU backend to CUDA provides a geometric-mean **1.49×** per-step speedup across this sweep, but that advantage is not uniform: native CUDA is faster in 4/6 configurations and loses slightly at 512 and 1000 points with 500 steps.
- PyTorch CUDA is faster than PyTorch CPU in 3/6 configurations. PyTorch CPU wins at 128 points, where GPU launch/graph overhead dominates; CUDA wins the larger, longer-running cases.
- Scaling is the main native CUDA weakness: from 128 to 1000 points at 500 steps, native CUDA grows from 6.653 to 39.858 ms/step, while PyTorch CUDA stays near 5.520–5.549 ms/step.
- Fastest backend counts: PyTorch CPU: 3/6, PyTorch CUDA: 3/6.

## Interpretation and limits

- This is a throughput comparison, not a numerical equivalence proof. The native trainer resamples Latin-hypercube collocation points each step, while the current PyTorch reference reuses a seeded uniform point tensor. The same architecture, PDE form, float32 precision, Adam hyperparameters, point counts, and step counts are used, but initial weights and samples are not bit-identical.
- Final losses therefore indicate that each run trains, but are not a fair accuracy ranking. A parity experiment should feed an identical point sequence and identical initialized weights into both implementations, then compare loss curves and parameter gradients.
- The native CPU result includes the framework's current fine-grained tensor/autograd allocation and OpenMP behavior. Its weak result at these sizes is consistent with the earlier raw-autograd benchmark, where scheduling/allocation overhead and many small dispatches outweighed parallel work.
- The native CUDA result includes GPU kernels, allocations, host/device transfers associated with the current graph, and cuBLAS. It is the relevant present-day end-to-end number, but kernel fusion, allocator reuse, and reducing synchronization would be needed before treating it as a cuBLAS-limited baseline.
- The GTX 1660 is an older Turing GPU without tensor cores for FP32; this comparison should be interpreted as framework efficiency on this hardware, not an upper bound for newer GPUs.

## Recommended next measurements

1. Add a shared binary sample/weight fixture and a per-step loss/gradient parity check before comparing accuracy.
2. Run 5–10 trials per point/step configuration, discard a warm-up run, and report median plus p95.
3. Profile native CPU with OpenMP disabled, thresholded, and OpenBLAS-only matmul to choose runtime dispatch thresholds.
4. Profile native CUDA with Nsight Systems/Compute; focus on allocations, launch count, small kernels, and host-device synchronization before optimizing math kernels.
