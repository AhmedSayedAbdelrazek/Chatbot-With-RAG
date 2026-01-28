FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/backend/uploads /app/backend/chroma_db

ENV PORT=8000
ENV CHROMA_DB_PATH=/app/backend/chroma_db

CMD ["sh", "-c", "uvicorn backend.main_cloud:app --host 0.0.0.0 --port ${PORT}"]
