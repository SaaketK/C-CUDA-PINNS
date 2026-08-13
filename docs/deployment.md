# Hugging Face deployment

This repository follows the same direct Docker Space layout as
`pinn-heat-equation`: the Space metadata, application, Dockerfile, dependency
files, and model artifact all live in the repository that is pushed to Hugging
Face.

The deployment-specific root files are:

```text
README.md
Dockerfile
app.py
config.py
schemas.py
requirements.txt
requirements-dev.txt
src/inference.py
models/heat1d/v1/model.pinn
models/heat1d/v1/metadata.json
```

The Docker build uses two stages. The first compiles
`libpinn_heat1d_backend.so` from the C sources without CUDA. The second contains
only the runtime libraries, Python serving layer, native library, and trained
model. FastAPI loads the model once at startup, and Gradio is mounted at `/`.

## Local verification

```bash
docker build -t c-cuda-pinn-space .
docker run --rm -p 7860:7860 c-cuda-pinn-space
curl http://localhost:7860/health
```

## Push to a Space

Create a new Hugging Face Space with the Docker SDK, then add its Git URL as a
remote for this repository and push `main`. Do not upload a separately prepared
deployment directory: Hugging Face builds the root `Dockerfile` directly.

The deployed image intentionally serves on CPU. The model is small, a full
`100 x 100` reconstruction takes only tens of milliseconds, and this keeps the
Space compatible with CPU hardware. CUDA remains part of the research and
training repository.
