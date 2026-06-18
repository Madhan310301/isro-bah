FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# System deps for rasterio + faiss
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    gdal-bin \
    libgeos-dev \
    libproj-dev \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/       ./app/
COPY ml/        ./ml/
COPY scripts/   ./scripts/

# Copy pre-built artefacts (must be built before docker build)
COPY checkpoints/best_model.pt  ./checkpoints/best_model.pt
COPY faiss_index/               ./faiss_index/

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

ENV MODEL_CHECKPOINT_PATH=/app/checkpoints/best_model.pt
ENV FAISS_INDEX_PATH=/app/faiss_index
ENV MODEL_BACKBONE=resnet50
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
