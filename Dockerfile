FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml ./pyproject.toml
RUN pip install --no-cache-dir \
    "openenv-core>=0.2.0" \
    "pydantic>=2.0" \
    "fastapi>=0.104.0" \
    "uvicorn>=0.24.0"

# Copy application code
COPY models.py ./models.py
COPY openenv.yaml ./openenv.yaml
COPY client.py ./client.py
COPY __init__.py ./__init__.py
COPY server/ ./server/

# Expose port 8000 (OpenEnv standard)
EXPOSE 8000

# Enable web interface
ENV ENABLE_WEB_INTERFACE=true
ENV PYTHONPATH="/app:$PYTHONPATH"

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
