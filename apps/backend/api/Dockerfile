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
    --mount=type=bind,source=apps/backend/api/pyproject.toml,target=apps/backend/api/pyproject.toml \
    uv sync --frozen --package api --no-install-workspace --no-editable

# 3. Copy the source and install the app into the venv.
COPY pyproject.toml uv.lock ./
COPY apps/backend/api/pyproject.toml apps/backend/api/README.md apps/backend/api/
COPY apps/backend/api/src apps/backend/api/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --package api --no-editable

# --- Runtime stage ---
FROM python:3.14-slim-trixie

# 4. Copy only the venv; it holds the installed app, so no source is needed.
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# 5. Copy the migrations; `alembic upgrade head` reads its config from the cwd.
WORKDIR /app
COPY apps/backend/api/pyproject.toml apps/backend/api/alembic.ini ./
COPY apps/backend/api/alembic alembic

# 6. Run as a non-root user (nobody:nogroup in the base image).
USER 65534:65534

# 7. Start the app through main.
ENV SERVER_HOST=0.0.0.0
# API
EXPOSE 8000
# Prometheus metrics
EXPOSE 9100
CMD ["api"]
