import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Combines BM25 keyword search and dense vector search via Reciprocal Rank Fusion,
    then re-ranks candidates with a cross-encoder for precision.
    """

    def __init__(
        self,
        vector_store,
        embedder,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

        self._bm25 = None
        self._bm25_docs: List[Dict[str, Any]] = []

        logger.info(f"Loading cross-encoder re-ranker: {reranker_model}")
        from sentence_transformers import CrossEncoder
        self.reranker = CrossEncoder(reranker_model, max_length=512)

    # ------------------------------------------------------------------ #
    # BM25                                                                 #
    # ------------------------------------------------------------------ #

    def _rebuild_bm25(self):
        docs = self.vector_store.get_all_documents()
        self._bm25_docs = docs
        if not docs:
            self._bm25 = None
            return
        from rank_bm25 import BM25Okapi
        tokenized = [d["content"].lower().split() for d in docs]
        self._bm25 = BM25Okapi(tokenized)
        logger.debug(f"BM25 index built with {len(docs)} docs")

    def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if self._bm25 is None or len(self._bm25_docs) == 0:
            return []
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        indices = np.argsort(scores)[::-1][:top_k]
        return [
            {**self._bm25_docs[i], "bm25_score": float(scores[i])}
            for i in indices
            if scores[i] > 0
        ]

    # ------------------------------------------------------------------ #
    # Reciprocal Rank Fusion                                               #
    # ------------------------------------------------------------------ #

    def _rrf(
        self,
        vector_hits: List[Dict],
        bm25_hits: List[Dict],
        k: int = 60,
    ) -> List[Dict]:
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict] = {}

        for rank, doc in enumerate(vector_hits):
            key = doc["content"][:120]
            scores[key] = scores.get(key, 0.0) + self.vector_weight / (k + rank + 1)
            doc_map[key] = doc

        for rank, doc in enumerate(bm25_hits):
            key = doc["content"][:120]
            scores[key] = scores.get(key, 0.0) + self.bm25_weight / (k + rank + 1)
            doc_map.setdefault(key, doc)

        sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
        return [{**doc_map[k], "hybrid_score": scores[k]} for k in sorted_keys]

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        rerank_top_k: int = 5,
        filter_by: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        # Refresh BM25 whenever we retrieve (index is cheap to rebuild at this scale)
        self._rebuild_bm25()

        # Dense search
        query_emb = self.embedder.embed_single(query)
        vector_hits = self.vector_store.search(query_emb, top_k=top_k, where=filter_by)

        # Sparse search (BM25 cannot filter by metadata, so filter post-hoc)
        bm25_hits = self._bm25_search(query, top_k=top_k)
        if filter_by:
            bm25_hits = [
                h for h in bm25_hits
                if all(h.get("metadata", {}).get(k) == v for k, v in filter_by.items())
            ]

        # Fusion
        fused = self._rrf(vector_hits, bm25_hits)
        candidates = fused[: max(top_k, rerank_top_k * 2)]

        if not candidates:
            return []

        # Cross-encoder re-ranking
        pairs = [(query, doc["content"]) for doc in candidates]
        rerank_scores = self.reranker.predict(pairs)

        for doc, score in zip(candidates, rerank_scores):
            doc["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        logger.info(
            f"Retrieved {len(reranked[:rerank_top_k])} chunks for query "
            f"(vector={len(vector_hits)}, bm25={len(bm25_hits)})"
        )
        return reranked[:rerank_top_k]
