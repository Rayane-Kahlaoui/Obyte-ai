# ── Stage 1: build deps ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools needed by some Python wheels (e.g. chromadb, sentence-transformers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install \
        fastapi \
        uvicorn[standard] \
        pydantic \
        requests \
        -r requirements.txt

# ── Stage 2: runtime ───────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy project source
COPY api/     ./api/
COPY rag/     ./rag/
COPY data/    ./data/
COPY database/ ./database/

# Expose FastAPI port
EXPOSE 8000

# Environment variables (override at runtime via -e or .env file)
ENV LLM_API_TYPE=openrouter \
    OPENROUTER_API_KEY="" \
    HF_TOKEN="" \
    LOCAL_LLM_API_URL="http://localhost:11434/v1" \
    LOCAL_LLM_MODEL="llama3.2:1b" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Run the API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
