# CareSight Risk Engine - Production Docker Image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Create necessary directories
RUN mkdir -p /app/logs /app/mlruns /app/data/reports

# Set Python path
ENV PYTHONPATH=/app/src

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash caresight && \
    chown -R caresight:caresight /app
USER caresight

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command - run FastAPI server
CMD ["python", "-m", "uvicorn", "src.serving.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
