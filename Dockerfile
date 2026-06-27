FROM python:3.13-slim

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

# HF Spaces runs containers as a non-root user (uid=1000, gid=1000)
# Create user and fix permissions on writable dirs
RUN useradd -m -u 1000 user && \
    mkdir -p data/uploads data/chroma_db logs && \
    chown -R user:user /app

USER user

# HF Spaces mandatory port
EXPOSE 7860

CMD ["python", "main.py"]
