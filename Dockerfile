FROM python:3.12-slim-bookworm AS native-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libopenblas-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /source
COPY CMakeLists.txt ./
COPY include ./include
COPY src ./src

RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DPINN_BUILD_DEVELOPMENT_TARGETS=OFF \
    && cmake --build build --target pinn_heat1d_backend --parallel


FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libopenblas0-pthread \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 user

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=native-builder /source/build/libpinn_heat1d_backend.so /app/lib/
COPY --chown=user:user app.py config.py schemas.py ./
COPY --chown=user:user src/__init__.py src/inference.py ./src/
COPY --chown=user:user models/heat1d/v1/model.pinn models/heat1d/v1/metadata.json ./models/heat1d/v1/

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PINN_HEAT1D_LIBRARY=/app/lib/libpinn_heat1d_backend.so \
    PINN_HEAT1D_MODEL=/app/models/heat1d/v1/model.pinn \
    PINN_HEAT1D_METADATA=/app/models/heat1d/v1/metadata.json \
    OPENBLAS_NUM_THREADS=2 \
    OMP_NUM_THREADS=2 \
    MPLCONFIGDIR=/tmp/matplotlib

USER user
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:7860/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
