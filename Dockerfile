# ── Build stage ────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install only what is needed to build the Python wheel
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Runtime libraries required by OpenCV and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the installed packages from the builder stage
COPY --from=builder /install /usr/local

# Copy project source
COPY src/ src/
COPY pyproject.toml .
COPY requirements.txt .

# Install the project itself in editable/direct mode
RUN pip install --no-cache-dir -e .

# ── Configuration ────────────────────────────────────────────────────────────────
# Ollama runs on the HOST machine (already installed on the Mac).
# When the container starts, point the app at the host's Ollama service.
# 'host.docker.internal' resolves to the host from inside a Docker container on
# macOS/Windows.  On Linux you may need --add-host=host.docker.internal:host-gateway.
ENV PHOTO_ANALYZER_LLM__OLLAMA_URL="http://host.docker.internal:11434"

# Store the SQLite database and all app data under /data (mount an external volume here)
ENV PHOTO_ANALYZER_DATA_DIR="/data"
ENV PHOTO_ANALYZER_CONFIG_DIR="/config"
ENV PHOTO_ANALYZER_CACHE_DIR="/cache"
ENV PHOTO_ANALYZER_LOG_DIR="/data/logs"

# Expose the FastAPI web interface port
EXPOSE 8080

# Create mount-point directories so Docker can bind volumes
RUN mkdir -p /data /config /cache /media

# ── Healthcheck ─────────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import httpx, sys; r = httpx.get('http://localhost:8080/health', timeout=5); sys.exit(0 if r.status_code == 200 else 1)"

# ── Default command ─────────────────────────────────────────────────────────────
# Start the FastAPI web service on port 8080
CMD ["uvicorn", "photo_analyzer.web.app:app", "--host", "0.0.0.0", "--port", "8080"]
