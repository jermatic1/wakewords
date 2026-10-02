FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY external ./external
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

# The container runs as the host user, so caches must not point at /root.
ENV PATH="/app/.venv/bin:${PATH}" \
    HOME=/tmp \
    XDG_CACHE_HOME=/work/data/.cache \
    NUMBA_CACHE_DIR=/tmp/numba-cache

WORKDIR /work
