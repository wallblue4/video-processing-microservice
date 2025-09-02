# video-processing-microservice/Dockerfile
FROM python:3.11-slim

# Metadatos
LABEL maintainer="TuStockYa Team"
LABEL version="1.0.0"
LABEL description="Video Processing Microservice with AI"

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalar dependencias del sistema para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    curl \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Copiar código
COPY . .

# Crear directorio para videos temporales
RUN mkdir -p /tmp/videos && \
    chmod 755 /tmp/videos && \
    mkdir -p /app/logs && \
    chmod 755 /app/logs


# Crear usuario no-root
RUN adduser --disabled-password --gecos '' --no-create-home appuser && \
    chown -R appuser:appuser /app && \
    chown -R appuser:appuser /tmp/videos

USER appuser

# Exponer puerto
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Comando por defecto
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--loop", "uvloop", "--http", "h11", \
     "--log-level", "info", "--no-access-log"]