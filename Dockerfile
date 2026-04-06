FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for Docker layer caching
COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY models.py ./models.py
COPY openenv.yaml ./openenv.yaml
COPY pyproject.toml ./pyproject.toml
COPY server/ ./server/

# Expose port for HF Spaces (default 7860)
EXPOSE 7860

# Health check for HF Spaces auto-ping
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run the FastAPI server
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
