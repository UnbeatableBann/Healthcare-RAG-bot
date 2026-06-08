# Builder stage
FROM ghcr.io/astral-sh/uv:0.8.15 AS uv
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 
ENV PYTHONUNBUFFERED=1 
ENV UV_SYSTEM_PYTHON=1

WORKDIR /app

COPY --from=uv /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project || \
    (sleep 5 && uv sync --frozen --no-dev --no-install-project) || \
    (sleep 10 && uv sync --frozen --no-dev --no-install-project)

COPY . .

RUN uv sync --frozen --no-dev

# Runtime stage
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 
ENV PYTHONUNBUFFERED=1 
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app

EXPOSE 8000

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
