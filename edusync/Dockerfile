# syntax=docker/dockerfile:1
# Build context: ./edusync  (uv workspace root)

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Layer 1: third-party dependencies only (cached until manifests change).
COPY pyproject.toml uv.lock .python-version* ./
COPY packages/core/pyproject.toml packages/core/pyproject.toml
COPY packages/c1/pyproject.toml packages/c1/pyproject.toml
COPY packages/c2/pyproject.toml packages/c2/pyproject.toml
COPY packages/c3/pyproject.toml packages/c3/pyproject.toml
COPY packages/c4/pyproject.toml packages/c4/pyproject.toml
COPY apps/api/pyproject.toml apps/api/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace --package edusync-api --extra ml

# Layer 2: workspace sources.
COPY packages packages
COPY apps apps
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --package edusync-api --extra ml

FROM python:3.14-slim-bookworm AS runtime
RUN useradd --system --uid 10001 --create-home --home-dir /home/edusync edusync \
    && mkdir -p /data && chown edusync:edusync /data
COPY --from=builder /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    EDUSYNC_DATA_DIR=/data \
    HF_HOME=/data/models
USER 10001
WORKDIR /app
VOLUME /data
EXPOSE 8100
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8100/health', timeout=4).status == 200 else 1)"]
CMD ["uvicorn", "edusync_api.asgi:app", "--host", "0.0.0.0", "--port", "8100"]
