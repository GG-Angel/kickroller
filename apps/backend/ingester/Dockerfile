# Build context: repo root (needs the workspace uv.lock).

# --- Build stage ---
FROM python:3.14-slim-trixie AS builder

# 1. Install uv and configure it for an image build.
COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_DEV=1 UV_PYTHON_DOWNLOADS=0
WORKDIR /app

# 2. Install dependencies only, so code changes reuse this layer.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=apps/backend/ingester/pyproject.toml,target=apps/backend/ingester/pyproject.toml \
    --mount=type=bind,source=libs/kickroller/pyproject.toml,target=libs/kickroller/pyproject.toml \
    uv sync --frozen --package ingester --no-install-workspace --no-editable

# 3. Copy the source and install the app into the venv.
COPY pyproject.toml uv.lock ./
COPY apps/backend/ingester/pyproject.toml apps/backend/ingester/README.md apps/backend/ingester/
COPY apps/backend/ingester/src apps/backend/ingester/src
COPY libs/kickroller/pyproject.toml libs/kickroller/README.md libs/kickroller/
COPY libs/kickroller/src libs/kickroller/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --package ingester --no-editable

# --- Runtime stage ---
FROM python:3.14-slim-trixie

# 4. Copy only the venv; it holds the installed app, so no source is needed.
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# 5. Run as a non-root user (nobody:nogroup in the base image).
USER 65534:65534

# 6. Start the worker through main.
CMD ["ingester"]
