import logging
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from src.utils.helpers import ensure_directories, load_config, setup_logging

config = load_config("config.yaml")
setup_logging(log_file=config["logging"]["file"], level=config["logging"]["level"])

logger = logging.getLogger(__name__)

# Ensure required directories exist before importing heavy deps
ensure_directories(
    config["api"]["upload_dir"],
    config["vectordb"]["persist_directory"],
    "./logs",
)

from src.chunking.chunker import SemanticFinancialChunker
from src.embeddings.embedder import Embedder
from src.llm.llm_client import LLMClient
from src.retrieval.retriever import HybridRetriever
from src.vectordb.vector_store import VectorStore
from src.api.routes import create_router

# ------------------------------------------------------------------ #
# Service initialisation (runs once at startup)                        #
# ------------------------------------------------------------------ #

embedder = Embedder(
    model_name=config["embedding"]["model"],
    batch_size=config["embedding"]["batch_size"],
)

vector_store = VectorStore(
    collection_name=config["vectordb"]["collection_name"],
    persist_directory=config["vectordb"]["persist_directory"],
)

chunker = SemanticFinancialChunker(
    embedder=embedder,
    breakpoint_percentile=config["chunking"]["breakpoint_percentile"],
    min_chunk_size=config["chunking"]["min_chunk_size"],
    max_chunk_size=config["chunking"]["max_chunk_size"],
    buffer_size=config["chunking"]["buffer_size"],
)

llm_client = LLMClient(
    model=config["llm"]["model"],
    max_tokens=config["llm"]["max_tokens"],
    temperature=config["llm"]["temperature"],
)

retriever = HybridRetriever(
    vector_store=vector_store,
    embedder=embedder,
    bm25_weight=config["retrieval"]["bm25_weight"],
    vector_weight=config["retrieval"]["vector_weight"],
    reranker_model=config["retrieval"]["reranker_model"],
)

# ------------------------------------------------------------------ #
# FastAPI app                                                          #
# ------------------------------------------------------------------ #

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FinSight starting up...")
    logger.info(f"Indexed: {vector_store.count()} chunks across {len(vector_store.list_sources())} documents")
    yield
    logger.info("FinSight shut down.")


app = FastAPI(
    title="FinSight",
    description=(
        "AI-powered RAG system for analysing annual reports, earnings transcripts, "
        "investor presentations, and financial news. "
        "Features hybrid BM25 + vector search with cross-encoder re-ranking and source citations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = create_router(
    retriever=retriever,
    llm_client=llm_client,
    chunker=chunker,
    embedder=embedder,
    vector_store=vector_store,
    upload_dir=config["api"]["upload_dir"],
    config=config,
)

app.include_router(router, prefix="/api/v1")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config["api"]["host"],
        port=config["api"]["port"],
        reload=True,
        reload_dirs=["src"],
        log_level=config["logging"]["level"].lower(),
    )
