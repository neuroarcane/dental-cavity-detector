# Dental Cavity Detector — serving API container (AASD 4016, SCRUM-43)
FROM python:3.11-slim

# System libraries needed by ultralytics/opencv/torch at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Run as a non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Cloud platforms inject $PORT; default to 8000 locally
ENV PORT=8000
EXPOSE 8000

# Production server. -w 1 keeps one model copy in memory; raise timeout for
# the first request (it downloads the weights from Hugging Face on cold start).
CMD gunicorn -b 0.0.0.0:${PORT:-8000} -w 1 --timeout 180 app:app
