# ─────────────────────────────────────────────────────────────────────────────
# Backend image - FastAPI (ChatService) served by uvicorn.
#
# Lightweight: no local ML deps (embeddings/LLMs are all API-based), so a slim
# Python base keeps the image small and cold/warm starts fast.
#
# Build:  docker build -t chatclone-backend .
# Run:    docker run -p 8000:8000 --env-file .env chatclone-backend
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# - PYTHONUNBUFFERED: stream logs immediately (important for container log tailing)
# - PYTHONDONTWRITEBYTECODE: no .pyc clutter in the image
# - PIP_NO_CACHE_DIR: smaller image
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    RELOAD=false

WORKDIR /app

# System deps: build-essential is occasionally needed for wheels that lack a
# prebuilt slim binary; curl is used by the container healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (own layer) so code changes don't bust the pip cache.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the application code. .dockerignore keeps venv/tests/.env/etc. out.
COPY . .

# Run as a non-root user for safety.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Liveness: the app exposes /health (see ChatService/Server/main.py).
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

# Launch from the backend root so `ChatService` resolves as a package (the app
# uses relative imports). RELOAD must be off in production.
CMD ["sh", "-c", "uvicorn ChatService.Server.main:app --host ${HOST} --port ${PORT}"]
