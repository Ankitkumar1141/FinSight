# FinSight

> A production-grade Retrieval-Augmented Generation (RAG) system engineered for deep financial document intelligence. Combines hybrid sparse-dense retrieval, cross-encoder re-ranking, and domain-aware prompt routing over SEC filings, earnings transcripts, and investor presentations.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        CLIENT  (HTTP / REST)                          │
└───────────────────┬──────────────────────────┬───────────────────────┘
                    │                          │
             POST /upload                 POST /query
                    │                          │
┌───────────────────▼──────────────────────────▼───────────────────────┐
│                      FastAPI  —  routes.py                            │
│              /upload   /query   /documents   /health                  │
└───────────────────┬──────────────────────────┬───────────────────────┘
                    │                          │
        ┌───────────▼────────┐     ┌───────────▼──────────────────────┐
        │  INGESTION PIPELINE│     │         QUERY PIPELINE            │
        │                    │     │                                    │
        │  loader.py         │     │  embedder.py                      │
        │  ─ PDF/DOCX/TXT    │     │  ─ Query → 384-dim vector         │
        │  ─ Table extract   │     │                                    │
        │  ─ Metadata tag    │     │  retriever.py                     │
        │                    │     │  ┌─────────────┬───────────────┐  │
        │  chunker.py        │     │  │ Vector      │ BM25 Keyword  │  │
        │  ─ Section-aware   │     │  │ Search      │ Search        │  │
        │  ─ 1000-char split │     │  │ (ChromaDB)  │ (rank-bm25)   │  │
        │  ─ Overlap 200     │     │  └──────┬──────┴───────┬───────┘  │
        │                    │     │         │               │          │
        │  embedder.py       │     │         └──────┬────────┘          │
        │  ─ all-MiniLM-L6   │     │                │                   │
        │  ─ 384-dim vectors │     │     Reciprocal Rank Fusion (RRF)   │
        │                    │     │                │                   │
        │  vector_store.py   │     │     Cross-Encoder Re-ranking       │
        │  ─ ChromaDB upsert │     │     (ms-marco-MiniLM-L-6-v2)      │
        │  ─ Persist to disk │     │                │                   │
        └────────────────────┘     │     prompt_templates.py            │
                                   │     ─ Intent routing               │
                 ┌─────────────────│─────  Risk / Revenue / Mgmt / Gen  │
                 │  ChromaDB Index │                │                   │
                 │  ./data/chroma  │     llm_client.py                  │
                 └─────────────────┘     ─ Mistral mistral-small-latest │
                                         ─ Cited structured response    │
                                   └──────────────────────────────────┘
```

---

## Retrieval Architecture — Hybrid Search

The retrieval layer fuses two complementary search strategies before applying a neural re-ranker:

```
Query
  │
  ├──► Dense Retrieval (ChromaDB cosine similarity)
  │    └── Captures semantic meaning, paraphrasing, conceptual similarity
  │
  └──► Sparse Retrieval (BM25 — Okapi BM25)
       └── Captures exact financial terms, tickers, metric names, acronyms
  │
  ▼
Reciprocal Rank Fusion  [score = Σ weight / (k + rank)]
  └── Merges ranked lists without score normalisation issues
  │
  ▼
Cross-Encoder Re-ranking  (ms-marco-MiniLM-L-6-v2)
  └── Full query-document attention — produces calibrated relevance scores
  │
  ▼
Top-K chunks  →  Prompt construction  →  LLM generation
```

**Why hybrid over pure vector search:**  
BM25 is precision-critical for financial text — exact matches on terms like `EBITDA`, `Regulation S-K Item 1A`, or `FY2023Q4` are lost in embedding space but retrieved perfectly by BM25. RRF fuses both without requiring score normalisation.

---

## Prompt Intent Router

Incoming queries are classified and dispatched to domain-specific prompt templates before reaching the LLM:

| Intent Signal | Template Applied | Structured Output |
|---|---|---|
| `risk / threat / exposure` | Risk Analyst | Severity tiers (High/Med/Low) + citations |
| `revenue / growth / margin / profit` | Financial Analyst | YoY tables + segment breakdown |
| `management / CEO / guidance / strategy` | Earnings Analyst | Direct quotes + tone analysis |
| *(default)* | General RAG | Cited prose response |

---

## Component Reference

| Module | Responsibility |
|---|---|
| `src/ingestion/loader.py` | Multi-format document loading (PDF, DOCX, TXT, MD); table extraction via `pdfplumber`; auto-detects `doc_type`, `year`, `company` from filename and content |
| `src/chunking/chunker.py` | Financial-aware recursive splitter; hard section boundaries at SEC Item headers, Risk Factors, Financial Statements; 1000-char chunks with 200-char overlap |
| `src/embeddings/embedder.py` | `sentence-transformers` wrapper; batched inference; L2-normalised output for cosine similarity |
| `src/vectordb/vector_store.py` | ChromaDB persistent client; upsert, cosine search, metadata filtering, per-source deletion |
| `src/retrieval/retriever.py` | Orchestrates BM25 + vector search; RRF fusion; cross-encoder re-ranking via `CrossEncoder.predict()` |
| `src/prompts/prompt_templates.py` | Regex-based intent router; four domain-specific system+user prompt pairs |
| `src/llm/llm_client.py` | Mistral AI SDK wrapper; synchronous and streaming generation modes |
| `src/api/routes.py` | FastAPI router; Pydantic request/response schemas; multipart file upload handling |
| `src/utils/helpers.py` | YAML config loader, rotating file logger, directory bootstrapper |

---

## API Reference

**Base URL:** `http://localhost:8000/api/v1`  
**Interactive docs:** `/docs` (Swagger UI) · `/redoc` (ReDoc)

### `POST /upload`
Ingest a financial document into the vector index.

```http
Content-Type: multipart/form-data

file: <binary>   # PDF, DOCX, TXT, or MD
```

```json
// 200 OK
{
  "filename": "aapl_10k_2023.pdf",
  "chunks_created": 312,
  "doc_type": "annual_report",
  "year": "2023",
  "message": "Document successfully uploaded and indexed."
}
```

### `POST /query`
Execute a financial research query against the indexed corpus.

```json
// Request
{
  "query": "What did management say about AI investments?",
  "top_k": 5,
  "filter_doc_type": "annual_report",   // optional
  "filter_year": "2023",                // optional
  "stream": false                       // optional — SSE streaming
}

// Response
{
  "query": "What did management say about AI investments?",
  "answer": "## Management Commentary on AI Investments\n\n...[Source 1: aapl_10k_2023.pdf, Page 24]",
  "sources": [
    {
      "document": "aapl_10k_2023.pdf",
      "page": 24,
      "doc_type": "annual_report",
      "year": "2023",
      "company": "AAPL",
      "relevance_score": 0.9412
    }
  ]
}
```

### `GET /documents`
Returns metadata for all indexed source documents.

### `DELETE /documents/{source_name}`
Removes all chunks associated with a document from the index.

### `GET /health`
Returns index statistics: `total_chunks`, `total_sources`, `sources[]`.

---

## Configuration

All runtime parameters are controlled via `config.yaml`. No code changes required to swap models or tune retrieval.

```yaml
embedding:
  model: "all-MiniLM-L6-v2"      # swap → BAAI/bge-large-en-v1.5 for +accuracy
  batch_size: 32

chunking:
  chunk_size: 1000                # characters per chunk
  chunk_overlap: 200

retrieval:
  top_k: 10                       # candidates before re-ranking
  bm25_weight: 0.4                # RRF weight for sparse retrieval
  vector_weight: 0.6              # RRF weight for dense retrieval
  rerank_top_k: 5                 # final chunks sent to LLM
  reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"

llm:
  model: "mistral-small-latest"
  max_tokens: 4096
  temperature: 0.1
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Mistral AI (`mistral-small-latest`) — free tier via La Plateforme |
| **Embeddings** | `sentence-transformers` — `all-MiniLM-L6-v2` (384-dim) |
| **Vector Store** | ChromaDB — persistent local HNSW index |
| **Sparse Retrieval** | `rank-bm25` — Okapi BM25 |
| **Re-ranking** | `sentence-transformers` CrossEncoder — `ms-marco-MiniLM-L-6-v2` |
| **Document Parsing** | `pdfplumber` (text + tables), `python-docx` |
| **API Framework** | FastAPI + Uvicorn |
| **Containerisation** | Docker + Docker Compose |
| **Cloud Deployment** | GCP Cloud Run + Artifact Registry + Secret Manager + Cloud Storage |

---

## Repository Structure

```
.
├── main.py                        # Application entrypoint — service wiring + FastAPI init
├── config.yaml                    # Unified runtime configuration
├── Dockerfile                     # Multi-stage container build
├── docker-compose.yml             # Local container orchestration with volume mounts
├── requirements.txt
├── src/
│   ├── ingestion/loader.py
│   ├── chunking/chunker.py
│   ├── embeddings/embedder.py
│   ├── vectordb/vector_store.py
│   ├── retrieval/retriever.py
│   ├── prompts/prompt_templates.py
│   ├── llm/llm_client.py
│   ├── api/routes.py
│   └── utils/helpers.py
├── tests/
│   └── test_app.py
├── data/
│   ├── uploads/                   # Ingested source documents
│   └── chroma_db/                 # Persisted HNSW vector index
└── logs/
    └── app.log
```

---

## Setup

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Configure credentials
echo "MISTRAL_API_KEY=your_mistral_key_here" > .env

# 3. Run
python main.py                    # local
docker-compose up -d              # docker
```

---

## Testing

```bash
pytest tests/ -v
```

Unit tests cover chunker logic, document loader utilities, prompt routing, config loading, and mocked vector store operations — no API keys or network access required.
