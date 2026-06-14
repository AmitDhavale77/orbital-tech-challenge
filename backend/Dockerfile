FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# libatomic1 is required by the Node runtime pyright downloads at check time.
RUN apt-get update && apt-get install -y --no-install-recommends libatomic1 \
    && rm -rf /var/lib/apt/lists/*

# Python needs to find the takehome package inside backend/src
ENV PYTHONPATH=/app/backend/src:$PYTHONPATH

# Copy dependency files first (layer caching)
COPY pyproject.toml uv.lock* ./

# Install dependencies
RUN uv sync --group dev

# Copy source (will be overridden by volume mount in dev)
COPY backend/src ./backend/src
COPY alembic ./alembic
COPY alembic.ini ./alembic.ini

# Create uploads directory
RUN mkdir -p uploads

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "takehome.web.app:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
