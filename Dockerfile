FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.docker.txt .
RUN pip install --no-cache-dir -r requirements.docker.txt

COPY api ./api
COPY providers ./providers
COPY config ./config
COPY ui ./ui
COPY scripts/__init__.py ./scripts/
COPY scripts/ingest ./scripts/ingest

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
