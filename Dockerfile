FROM python:3.14-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml main.py ./

# Install dependencies (including dev for runtime)
RUN uv sync --no-dev --system

# Create data directory
RUN mkdir -p /app/data

EXPOSE 8090

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8090"]
