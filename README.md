---
title: C CUDA PINN Heat Equation
emoji: 🌡️
colorFrom: orange
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# C/CUDA PINN Heat Equation

A parametric physics-informed surrogate for the one-dimensional heat equation,
trained and evaluated by the native C/CUDA PINN framework in this repository.
One model covers a continuous range of diffusivities and initial conditions; no
retraining is performed when the sliders change.

## What the deployed model solves

The surrogate solves

```text
u_t = alpha * u_xx,   x in [0, 1], t in [0, 1]
u(t, 0) = u(t, 1) = 0
u(0, x) = a1*sin(pi*x) + a2*sin(2*pi*x)
```

The shared network learns a dimensionless modal decay function `q(tau)`. The
deployed field is reconstructed as

```text
u(t,x) = a1*sin(pi*x)*q(alpha*pi^2*t)
       + a2*sin(2*pi*x)*q(4*alpha*pi^2*t)
```

for `alpha` in `[0.01, 0.50]` and `a1`, `a2` in `[-1, 1]`. The exported model's
validation suite reports a worst-case relative L2 error of `5.18e-3` over its
parameter cases.

## Deployment architecture

- FastAPI serves the JSON API.
- Gradio is mounted at `/` for the interactive demo.
- The trained model is loaded once during application startup.
- Python calls the native shared library through a small `ctypes` interface.
- The compare endpoint is LRU-cached with parameters rounded to three decimals.
- The Docker Space compiles the CPU backend in a build stage. CUDA remains
  available for local training and benchmarking, but is not required to serve.

## API

```text
GET  /health
POST /api/v1/predict/pinn
POST /api/v1/predict/exact
POST /api/v1/predict/compare
GET  /api/v1/model/info
```

```bash
curl -X POST http://localhost:7860/api/v1/predict/compare \
  -H "Content-Type: application/json" \
  -d '{"a1": 0.7, "a2": -0.4, "alpha": 0.1}'
```

Each solution is returned on a fixed `100 x 100` `(x, t)` grid.

## Run locally with Docker

```bash
docker build -t c-cuda-pinn-space .
docker run --rm -p 7860:7860 c-cuda-pinn-space
```

Open <http://localhost:7860>. To deploy, create a Hugging Face Docker Space and
push this repository to the Space's Git remote; the root `README.md` and
`Dockerfile` are already the Space entrypoints.

## Run locally without Docker

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target pinn_heat1d_backend --parallel
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 7860
```

## Tests

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
pip install -r requirements-dev.txt
pytest -q tests/test_deployment_inference.py tests/test_deployment_api.py
```
