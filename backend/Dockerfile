FROM python:3.11-slim
# -----------------------------
# System dependencies (IMPORTANT)
# -----------------------------
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# -----------------------------
# Working directory
# -----------------------------
WORKDIR /app
# -----------------------------
# Python dependencies
# -----------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# -----------------------------
# NLTK data (REQUIRED)
# -----------------------------
RUN python -m nltk.downloader \
    stopwords \
    wordnet \
    omw-1.4 \
    punkt \
    averaged_perceptron_tagger \
    averaged_perceptron_tagger_eng
# -----------------------------
# Copy application code
# -----------------------------
COPY api ./api
COPY models ./models
# -----------------------------
# Expose & run
# -----------------------------
EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
