FROM python:3.13-slim

WORKDIR /app

# ── System libs ────────────────────────────────────────────────────────────────
# poppler  : required by pdfplumber to extract text from PDFs
# gcc/g++  : required to compile chroma-hnswlib from source
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libpoppler-cpp-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ── CPU-only PyTorch ───────────────────────────────────────────────────────────
# Install torch BEFORE requirements.txt so pip picks the lightweight CPU wheel
# (~220 MB) instead of the default GPU build (~1.1 GB with CUDA/cuDNN/NCCL).
RUN pip install --no-cache-dir \
    torch==2.5.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ── Python dependencies ────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Remove build tools to shrink final image ───────────────────────────────────
RUN apt-get purge -y --auto-remove gcc g++ libpoppler-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Source code ────────────────────────────────────────────────────────────────
COPY . .

# ── Non-root user (required by Hugging Face Spaces) ────────────────────────────
RUN useradd -m -u 1000 user && \
    mkdir -p data/uploads data/chroma_db logs && \
    chown -R user:user /app

USER user

# HF Spaces mandatory port
EXPOSE 7860

CMD ["python", "main.py"]
