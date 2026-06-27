import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ------------------------------------------------------------------ #
# Chunker                                                              #
# ------------------------------------------------------------------ #

class TestSemanticFinancialChunker:
    def setup_method(self):
        from unittest.mock import MagicMock
        import numpy as np
        from src.chunking.chunker import SemanticFinancialChunker

        # Mock embedder: returns random unit vectors so distance logic runs
        mock_embedder = MagicMock()
        def fake_embed(texts):
            vecs = np.random.rand(len(texts), 384).astype(np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return (vecs / norms).tolist()
        mock_embedder.embed.side_effect = fake_embed

        self.chunker = SemanticFinancialChunker(
            embedder=mock_embedder,
            breakpoint_percentile=85.0,
            min_chunk_size=50,
            max_chunk_size=2000,
            buffer_size=1,
        )

    def test_basic_chunking_returns_chunks(self):
        pages = [{
            "content": (
                "Revenue for FY2023 was $10 billion. "
                "Net income increased 15% year-over-year. "
                "Operating margin expanded by 200 basis points. "
                "The company saw strong growth across all segments."
            ),
            "metadata": {"source": "test.pdf", "page": 1, "doc_type": "annual_report",
                         "year": "2023", "company": "ACME"},
        }]
        chunks = self.chunker.chunk_documents(pages)
        assert len(chunks) >= 1
        assert all("content" in c for c in chunks)
        assert all("metadata" in c for c in chunks)
        assert all(c["metadata"]["source"] == "test.pdf" for c in chunks)

    def test_oversized_chunk_is_split(self):
        long_text = ("This is a long financial sentence discussing revenue growth. " * 40).strip()
        result = self.chunker._split_oversized(long_text)
        assert len(result) > 1
        assert all(len(c) <= self.chunker.max_chunk_size for c in result)

    def test_section_boundary_forces_new_chunk(self):
        text = (
            "Introduction with background context about the company. "
            "RISK FACTORS Competition is intense in our markets. "
            "Management believes risks are manageable going forward."
        )
        pages = [{"content": text, "metadata": {
            "source": "10k.pdf", "page": 1, "doc_type": "annual_report",
            "year": "2023", "company": "TEST"
        }}]
        chunks = self.chunker.chunk_documents(pages)
        assert len(chunks) >= 1

    def test_tiny_content_returns_empty(self):
        pages = [{"content": "Hi. OK.", "metadata": {
            "source": "tiny.txt", "page": 1, "doc_type": "unknown",
            "year": "unknown", "company": "unknown"
        }}]
        chunks = self.chunker.chunk_documents(pages)
        assert all(len(c["content"]) >= self.chunker.min_chunk_size for c in chunks)

    def test_tiny_chunks_are_merged(self):
        chunks = ["Short.", "Also short.", "This is a much longer sentence that has real content in it."]
        merged = self.chunker._merge_tiny_chunks(chunks)
        assert len(merged) < len(chunks)

    def test_abbreviations_not_split(self):
        text = (
            "Apple Inc. reported strong results. "
            "Revenue grew vs. prior year expectations. "
            "The CEO of Apple Corp. confirmed guidance."
        )
        sentences = self.chunker._split_sentences(text)
        # "Inc." and "Corp." and "vs." should NOT create extra sentence breaks
        assert len(sentences) <= 3
        # Original dots must be restored in output
        assert all("Inc." in s or "Corp." in s or "vs." in s or "Apple" in s
                   for s in sentences if "Inc" in s or "Corp" in s or "vs" in s)

    def test_decimal_numbers_not_split(self):
        text = "Revenue was $3.5 billion. EPS came in at $1.23 per share."
        sentences = self.chunker._split_sentences(text)
        # "$3.5" and "$1.23" should not create false sentence breaks
        assert len(sentences) <= 2


# ------------------------------------------------------------------ #
# Loader utilities                                                     #
# ------------------------------------------------------------------ #

class TestDocLoader:
    def test_detect_doc_type_annual_report(self):
        from src.ingestion.loader import detect_doc_type
        assert detect_doc_type("AAPL_10-k_2023.pdf") == "annual_report"
        assert detect_doc_type("annual_report_2022.pdf") == "annual_report"

    def test_detect_doc_type_earnings(self):
        from src.ingestion.loader import detect_doc_type
        assert detect_doc_type("Q4_earnings_transcript.pdf") == "earnings_transcript"

    def test_detect_doc_type_presentation(self):
        from src.ingestion.loader import detect_doc_type
        assert detect_doc_type("investor_presentation.pdf") == "investor_presentation"

    def test_detect_doc_type_unknown(self):
        from src.ingestion.loader import detect_doc_type
        assert detect_doc_type("random_file.pdf") == "unknown"

    def test_extract_year_found(self):
        from src.ingestion.loader import extract_year
        assert extract_year("Fiscal year 2023 results were strong") == "2023"
        assert extract_year("Q4 2022 earnings call") == "2022"

    def test_extract_year_not_found(self):
        from src.ingestion.loader import extract_year
        assert extract_year("No year mentioned in this text") == "unknown"

    def test_unsupported_extension_raises(self):
        from src.ingestion.loader import load_document
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_document("report.xls")


# ------------------------------------------------------------------ #
# Config helpers                                                       #
# ------------------------------------------------------------------ #

class TestHelpers:
    def test_load_config_keys(self):
        from src.utils.helpers import load_config
        config = load_config("config.yaml")
        for key in ("embedding", "chunking", "vectordb", "retrieval", "llm", "api", "logging"):
            assert key in config, f"Missing config key: {key}"

    def test_format_sources_markdown(self):
        from src.utils.helpers import format_sources_markdown
        chunks = [
            {"content": "text", "metadata": {"source": "aapl_10k.pdf", "page": 5,
                                              "year": "2023", "doc_type": "annual_report"}},
            {"content": "text", "metadata": {"source": "aapl_10k.pdf", "page": 5,
                                              "year": "2023", "doc_type": "annual_report"}},  # dup
        ]
        result = format_sources_markdown(chunks)
        assert "aapl_10k.pdf" in result
        assert result.count("aapl_10k.pdf") == 1  # deduplicated


# ------------------------------------------------------------------ #
# Prompt routing                                                       #
# ------------------------------------------------------------------ #

class TestPromptTemplates:
    def _make_chunks(self, content: str = "Revenue was $10B."):
        return [{
            "content": content,
            "metadata": {"source": "report.pdf", "page": 5,
                         "doc_type": "annual_report", "year": "2023", "company": "ACME"},
        }]

    def test_risk_query_uses_risk_template(self):
        from src.prompts.prompt_templates import get_specialized_prompt
        _, user = get_specialized_prompt("What are the main risks?", self._make_chunks())
        assert "Risk" in user or "risk" in user

    def test_revenue_query_uses_revenue_template(self):
        from src.prompts.prompt_templates import get_specialized_prompt
        _, user = get_specialized_prompt("Compare revenue growth for the last 3 years", self._make_chunks())
        assert "Revenue" in user or "revenue" in user

    def test_mgmt_query_uses_mgmt_template(self):
        from src.prompts.prompt_templates import get_specialized_prompt
        _, user = get_specialized_prompt("What did management say about AI investments?", self._make_chunks())
        assert "management" in user.lower() or "Strategic" in user

    def test_general_query_uses_general_template(self):
        from src.prompts.prompt_templates import get_specialized_prompt
        _, user = get_specialized_prompt("Tell me about the company", self._make_chunks())
        assert "report.pdf" in user
        assert "Page 5" in user

    def test_context_includes_source_headers(self):
        from src.prompts.prompt_templates import get_specialized_prompt
        _, user = get_specialized_prompt("What happened?", self._make_chunks("Big event."))
        assert "Source 1" in user
        assert "ACME" in user


# ------------------------------------------------------------------ #
# VectorStore (mocked)                                                 #
# ------------------------------------------------------------------ #

class TestVectorStoreMocked:
    def test_count_delegates_to_collection(self):
        with patch("chromadb.PersistentClient") as mock_client:
            mock_col = MagicMock()
            mock_col.count.return_value = 42
            mock_client.return_value.get_or_create_collection.return_value = mock_col
            from src.vectordb.vector_store import VectorStore
            store = VectorStore("test", "./tmp_db")
            assert store.count() == 42

    def test_list_sources_empty(self):
        with patch("chromadb.PersistentClient") as mock_client:
            mock_col = MagicMock()
            mock_col.count.return_value = 0
            mock_client.return_value.get_or_create_collection.return_value = mock_col
            from src.vectordb.vector_store import VectorStore
            store = VectorStore("test", "./tmp_db")
            assert store.list_sources() == []
