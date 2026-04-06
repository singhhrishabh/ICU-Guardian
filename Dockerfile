FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies (minimal — no openenv-core needed)
RUN pip install --no-cache-dir \
    "fastapi>=0.104.0" \
    "uvicorn>=0.24.0" \
    "pydantic>=2.0"

# Copy application code
COPY models.py ./models.py
COPY openenv.yaml ./openenv.yaml
COPY server/ ./server/

# Port
EXPOSE 8000

ENV PYTHONPATH="/app:$PYTHONPATH"

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
