# Unified Dockerfile for Hugging Face Spaces (or Render)
# Builds the Vite React Frontend and serves it statically via FastAPI Backend

# Stage 1: Build Frontend
FROM node:20-alpine as frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Build Backend & Serve
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend source code and config
COPY backend/ ./backend/
COPY ml/src/ ./ml/src/
COPY data/breed_metadata.csv ./data/
# Copy the built frontend
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist
# Copy the exported model artifact
COPY models/vit_best.pth ./models/vit_best.pth

# Set runtime environments
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/vit_best.pth
ENV METADATA_PATH=/app/data/breed_metadata.csv
ENV MODEL_NAME=vit
ENV MODEL_VERSION=v2.0

# Hugging Face Spaces run explicitly on 7860
EXPOSE 7860

# Run Uvicorn via HTTP (Hugging Face / Render friendly)
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
