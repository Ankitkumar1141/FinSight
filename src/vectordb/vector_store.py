import logging
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, collection_name: str, persist_directory: str):
        import chromadb
        from chromadb.config import Settings

        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"Vector store ready — collection '{collection_name}' "
            f"at '{persist_directory}' ({self.count()} chunks)"
        )

    def add_chunks(
        self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]
    ) -> List[str]:
        ids = [str(uuid.uuid4()) for _ in chunks]
        documents = [c["content"] for c in chunks]

        metadatas = []
        for c in chunks:
            # ChromaDB only accepts str / int / float / bool metadata values
            meta: Dict[str, Any] = {}
            for k, v in c["metadata"].items():
                if isinstance(v, (int, float, bool)):
                    meta[k] = v
                else:
                    meta[k] = str(v)
            metadatas.append(meta)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(f"Added {len(chunks)} chunks — total: {self.count()}")
        return ids

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        where: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        n = min(top_k, self.count())
        if n == 0:
            return []

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            # ChromaDB 0.4+ requires explicit operator syntax for where filters.
            # Wrap each value in {"$eq": value} and combine multiple filters with $and.
            chroma_where = {k: {"$eq": v} for k, v in where.items()}
            if len(chroma_where) > 1:
                kwargs["where"] = {"$and": [{k: v} for k, v in chroma_where.items()]}
            else:
                kwargs["where"] = chroma_where

        results = self.collection.query(**kwargs)

        hits = []
        for i, doc in enumerate(results["documents"][0]):
            hits.append({
                "content": doc,
                "metadata": results["metadatas"][0][i],
                # ChromaDB cosine distance → similarity
                "score": 1.0 - results["distances"][0][i],
            })
        return hits

    def get_all_documents(self) -> List[Dict[str, Any]]:
        if self.count() == 0:
            return []
        results = self.collection.get(include=["documents", "metadatas"])
        return [
            {
                "id": results["ids"][i],
                "content": results["documents"][i],
                "metadata": results["metadatas"][i],
            }
            for i in range(len(results["ids"]))
        ]

    def delete_by_source(self, source_name: str) -> int:
        results = self.collection.get(
            where={"source": {"$eq": source_name}},
            include=["documents"],
        )
        ids = results["ids"]
        if ids:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} chunks for source '{source_name}'")
        return len(ids)

    def list_sources(self) -> List[str]:
        if self.count() == 0:
            return []
        results = self.collection.get(include=["metadatas"])
        sources = {m.get("source", "unknown") for m in results["metadatas"]}
        return sorted(sources)

    def get_metadata_for_source(self, source: str) -> Dict[str, Any]:
        results = self.collection.get(
            where={"source": {"$eq": source}},
            limit=1,
            include=["metadatas"],
        )
        return results["metadatas"][0] if results["metadatas"] else {}

    def count(self) -> int:
        return self.collection.count()
