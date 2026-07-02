import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md"}


# ------------------------------------------------------------------ #
# Request / Response schemas                                         #
# ------------------------------------------------------------------ #

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    filter_doc_type: Optional[str] = None
    filter_year: Optional[str] = None
    stream: bool = False


class SourceInfo(BaseModel):
    document: str
    page: Any
    doc_type: str
    year: str
    company: str
    relevance_score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceInfo]


class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_created: int
    doc_type: str
    year: str


class DocumentInfo(BaseModel):
    source: str
    doc_type: str
    year: str
    company: str


class HealthResponse(BaseModel):
    status: str
    total_chunks: int
    total_sources: int
    sources: List[str]


# ------------------------------------------------------------------ #
# Router factory                                                     #
# ------------------------------------------------------------------ #

def create_router(
    retriever,
    llm_client,
    chunker,
    embedder,
    vector_store,
    upload_dir: str,
    config: Dict[str, Any],
) -> APIRouter:
    from src.ingestion.loader import load_document
    from src.prompts.prompt_templates import get_specialized_prompt

    router = APIRouter()

    # ---- Upload ---------------------------------------------------- #

    @router.post("/upload", response_model=UploadResponse, tags=["Documents"])
    async def upload_document(file: UploadFile = File(...)):
        import asyncio

        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
            )

        os.makedirs(upload_dir, exist_ok=True)
        dest = os.path.join(upload_dir, file.filename)

        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)

        def _process() -> UploadResponse:
            """CPU-bound processing — runs in a thread pool to avoid blocking uvicorn."""
            pages = load_document(dest)
            chunks = chunker.chunk_documents(pages)
            if not chunks:
                raise ValueError("No text content could be extracted from the file.")
            embeddings = embedder.embed([c["content"] for c in chunks])
            vector_store.add_chunks(chunks, embeddings)
            sample_meta = chunks[0]["metadata"]
            return UploadResponse(
                message="Document successfully uploaded and indexed.",
                filename=file.filename,
                chunks_created=len(chunks),
                doc_type=sample_meta.get("doc_type", "unknown"),
                year=sample_meta.get("year", "unknown"),
            )

        try:
            # Allow up to 8 minutes for large PDFs on CPU
            result = await asyncio.wait_for(
                asyncio.to_thread(_process),
                timeout=480,
            )
            return result
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail="Processing timed out (>8 min). Try a smaller file or split the PDF.",
            )
        except Exception as exc:
            logger.error(f"Failed to process '{file.filename}': {exc}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))


    # ---- Query ----------------------------------------------------- #

    @router.post("/query", response_model=QueryResponse, tags=["Query"])
    async def query(request: QueryRequest):
        """Ask a financial research question over the indexed documents."""
        if vector_store.count() == 0:
            raise HTTPException(
                status_code=400,
                detail="No documents indexed yet. Please upload financial documents first.",
            )

        filter_by: Optional[Dict] = None
        if request.filter_doc_type:
            filter_by = {"doc_type": request.filter_doc_type}
        if request.filter_year:
            filter_by = filter_by or {}
            filter_by["year"] = request.filter_year

        try:
            chunks = retriever.retrieve(
                query=request.query,
                top_k=config["retrieval"]["top_k"],
                rerank_top_k=request.top_k,
                filter_by=filter_by,
            )

            if not chunks:
                return QueryResponse(
                    query=request.query,
                    answer=(
                        "No relevant content found for your query in the indexed documents. "
                        "Try uploading more relevant files or rephrasing your question."
                    ),
                    sources=[],
                )

            system_prompt, user_prompt = get_specialized_prompt(request.query, chunks)

            if request.stream:
                def event_stream():
                    for token in llm_client.stream(system_prompt, user_prompt):
                        yield token

                return StreamingResponse(event_stream(), media_type="text/plain")

            answer = llm_client.generate(system_prompt, user_prompt)

            # Deduplicate sources by (document, page)
            seen: set = set()
            sources: List[SourceInfo] = []
            for chunk in chunks:
                meta = chunk.get("metadata", {})
                key = (meta.get("source"), meta.get("page"))
                if key in seen:
                    continue
                seen.add(key)
                sources.append(
                    SourceInfo(
                        document=meta.get("source", "Unknown"),
                        page=meta.get("page", "N/A"),
                        doc_type=meta.get("doc_type", "unknown"),
                        year=meta.get("year", "unknown"),
                        company=meta.get("company", "unknown"),
                        relevance_score=round(
                            chunk.get("rerank_score", chunk.get("hybrid_score", 0.0)), 4
                        ),
                    )
                )

            return QueryResponse(query=request.query, answer=answer, sources=sources)

        except Exception as exc:
            logger.error(f"Query failed: {exc}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # ---- Documents ------------------------------------------------- #

    @router.get("/documents", response_model=List[DocumentInfo], tags=["Documents"])
    async def list_documents():
        """List all indexed source documents."""
        sources = vector_store.list_sources()
        result = []
        for source in sources:
            meta = vector_store.get_metadata_for_source(source)
            result.append(
                DocumentInfo(
                    source=source,
                    doc_type=meta.get("doc_type", "unknown"),
                    year=meta.get("year", "unknown"),
                    company=meta.get("company", "unknown"),
                )
            )
        return result

    @router.delete("/documents/{source_name}", tags=["Documents"])
    async def delete_document(source_name: str):
        """Remove a document and all its chunks from the index."""
        count = vector_store.delete_by_source(source_name)
        if count == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{source_name}' not found in the index.",
            )
        return {"message": f"Removed {count} chunks for '{source_name}'."}

    # ---- Health ---------------------------------------------------- #

    @router.get("/health", response_model=HealthResponse, tags=["System"])
    async def health():
        """Service health check."""
        sources = vector_store.list_sources()
        return HealthResponse(
            status="healthy",
            total_chunks=vector_store.count(),
            total_sources=len(sources),
            sources=sources,
        )

    return router
