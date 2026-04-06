# ============================================================
# SalesOps OpenEnv — Dockerfile
# Multi-stage build for minimal final image size.
# Compatible with Hugging Face Spaces (Docker SDK).
# ============================================================

# ---- Stage 1: dependency builder ----
FROM python:3.11-slim AS builder

WORKDIR /build

# Copy only requirements first for effective layer caching
COPY requirements.txt .

RUN pip install --upgrade pip --quiet && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Stage 2: runtime image ----
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="SalesOps OpenEnv v1" \
      org.opencontainers.image.description="Enterprise CRM workflow RL environment" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.licenses="MIT"

# Non-root user for security best practices
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source code
COPY --chown=appuser:appuser . .

# Environment variable defaults (can be overridden at runtime)
ENV API_BASE_URL="https://api.openai.com/v1" \
    MODEL_NAME="gpt-4o-mini" \
    HF_TOKEN="" \
    IMAGE_NAME="salesops-openenv:latest" \
    PORT=7860 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose HF Spaces default port
EXPOSE 7860

USER appuser

# Health check — HF automated ping must return 200
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Default: run the FastAPI health+inference server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
