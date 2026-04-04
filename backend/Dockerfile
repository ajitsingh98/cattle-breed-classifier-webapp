# Backend Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy project files
COPY backend/ ./backend/
COPY ml/ ./ml/
COPY data/ ./data/
COPY models/ ./models/

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/vit_best.pth
ENV METADATA_PATH=/app/data/breed_metadata.csv
ENV CLASSES_PATH=/app/models/classes.txt
ENV MODEL_NAME=vit
ENV MODEL_VERSION=v2.0

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
