FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV CHROMA_DB_PATH=/app/backend/chroma_db

COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    PIP_CONFIG_FILE=/dev/null pip install --no-cache-dir --default-timeout=300 -r requirements.txt

COPY . .

RUN mkdir -p /app/backend/uploads /app/backend/chroma_db

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.main_cloud:app --host 0.0.0.0 --port ${PORT}"]
