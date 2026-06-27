FROM python:3.11-slim

WORKDIR /app

# System libs required by pdfplumber / poppler
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libpoppler-cpp-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Pre-create data dirs (volumes will override at runtime)
RUN mkdir -p data/uploads data/chroma_db logs

EXPOSE 8000

CMD ["python", "main.py"]
