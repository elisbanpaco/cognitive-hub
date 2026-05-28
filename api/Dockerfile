FROM python:3.13-slim

# 1. Traer uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 2. Configurar variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV PATH="/workspace/api/.venv/bin:$PATH"

WORKDIR /workspace

# 3. Instalar dependencias del sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# 4. Copiar SOLO los archivos de uv (¡Adiós requirements.txt!)
COPY api/pyproject.toml api/uv.lock* /workspace/api/

WORKDIR /workspace/api

# 5. Instalar dependencias de Python con uv
RUN uv sync --no-dev --no-install-project

WORKDIR /workspace

# 6. Copiar el código fuente y modelos (Asegúrate de tener el .dockerignore listo)
COPY api/ /workspace/api/
COPY ml_research/predictive_poc/model_artifacts/ /workspace/ml_research/predictive_poc/model_artifacts/

WORKDIR /workspace/api

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]