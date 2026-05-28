FROM python:3.13-slim

# Copiar 'uv' directamente desde la imagen oficial
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Variables de entorno para optimizar Python y uv
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
# Agregamos el entorno virtual al PATH por defecto
ENV PATH="/workspace/api/.venv/bin:$PATH"

WORKDIR /workspace

# Instalar dependencias del SO necesarias para compilar paquetes
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copiar SOLO los archivos de dependencias de uv primero
COPY api/pyproject.toml api/uv.lock /workspace/api/

WORKDIR /workspace/api

# Instalar dependencias con uv (creará el .venv limpio dentro del contenedor)
RUN uv sync --no-dev --no-install-project

WORKDIR /workspace

# copiamos tu código (el .dockerignore evitará que se copie el .venv local)
COPY api/ /workspace/api/
COPY ml_research/predictive_poc/model_artifacts/ /workspace/ml_research/predictive_poc/model_artifacts/

WORKDIR /workspace/api

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]