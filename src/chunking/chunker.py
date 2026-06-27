import logging
import re
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# Hard section boundaries in SEC filings and financial reports
FINANCIAL_SECTION_MARKERS = [
    "ITEM ", "Item ",
    "SECTION ", "Section ",
    "RISK FACTORS", "Risk Factors",
    "MANAGEMENT'S DISCUSSION", "Management's Discussion",
    "FINANCIAL STATEMENTS", "Financial Statements",
    "NOTES TO", "Notes to",
    "CONSOLIDATED BALANCE", "Consolidated Balance",
    "CONSOLIDATED STATEMENTS", "Consolidated Statements",
    "QUANTITATIVE AND QUALITATIVE", "Quantitative and Qualitative",
    "CRITICAL ACCOUNTING", "Critical Accounting",
    "FORWARD-LOOKING", "Forward-Looking",
]

# Abbreviations common in financial documents that end in "." but are NOT
# sentence endings — naive splitters break on these.
_FINANCIAL_ABBREVIATIONS = [
    # Corporate suffixes
    "Inc", "Corp", "Ltd", "Co", "LLC", "LLP", "PLC", "AG", "SA",
    # Academic / formal
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr",
    # Common abbreviations
    "vs", "etc", "approx", "est", "avg", "yr", "mo", "dept",
    "No", "Vol", "Fig", "Sec", "Ref", "Para",
    # Fiscal / quarter markers
    "FY", "YTD", "Q1", "Q2", "Q3", "Q4",
    # Months
    "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    # Financial terms
    "p", "pp", "cf", "i.e", "e.g", "et al", "viz",
]

# Pre-compile a single pattern that matches any abbreviation followed by a period
_ABBREV_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in _FINANCIAL_ABBREVIATIONS) + r")\."
)

# Matches decimal numbers (e.g. 3.5, $1.2B, 99.9%)
_DECIMAL_PATTERN = re.compile(r"(\d+)\.(\d+)")

_PLACEHOLDER = "__PERIOD__"


class SemanticFinancialChunker:
    """
    Semantic Chunking for financial documents.

    Why semantic over fixed-size (RecursiveTextSplitter):
      - Fixed-size splits produce chunks that span multiple unrelated topics.
        A single chunk about revenue trends may bleed into debt discussion,
        causing the retrieved chunk to pollute LLM context with noise.
      - Semantic chunking splits where the *meaning* shifts, producing
        topic-pure chunks that retrieve cleanly and answer precisely.

    Algorithm:
      1. Split text into sentences (abbreviation-aware for financial text).
      2. Build a sliding window of `buffer_size` sentences around each
         sentence and embed the window — gives neighbourhood context so
         distances reflect topic shifts, not local phrasing variation.
      3. Compute cosine distance between consecutive window embeddings.
      4. Split wherever distance exceeds the Nth percentile threshold.
      5. Force additional hard splits at SEC/financial section headers.
      6. Merge tiny chunks into neighbours; split oversized chunks by
         sentence boundary.

    Trade-off vs RecursiveTextSplitter:
      + Better retrieval precision (topic-pure chunks).
      + No redundant overlap data stored in the vector index.
      - Ingestion is slower: every upload embeds all sentences before storing.
      - Percentile threshold is a hyperparameter (default 85 works well for
        dense financial prose; lower it for sparser documents).
    """

    def __init__(
        self,
        embedder,
        breakpoint_percentile: float = 85.0,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
        buffer_size: int = 1,
    ):
        self.embedder = embedder
        self.breakpoint_percentile = breakpoint_percentile
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.buffer_size = buffer_size

    # ------------------------------------------------------------------ #
    # Sentence splitting — abbreviation-aware                              #
    # ------------------------------------------------------------------ #

    def _protect_non_boundaries(self, text: str) -> str:
        """Replace periods inside abbreviations and decimals with a placeholder
        so the sentence splitter does not split on them."""
        text = _ABBREV_PATTERN.sub(lambda m: m.group(1) + _PLACEHOLDER, text)
        text = _DECIMAL_PATTERN.sub(lambda m: m.group(1) + _PLACEHOLDER + m.group(2), text)
        return text

    def _split_sentences(self, text: str) -> List[str]:
        protected = self._protect_non_boundaries(text)

        # Split on sentence-ending punctuation followed by whitespace
        raw = re.split(r"(?<=[.!?])\s+", protected)

        sentences: List[str] = []
        for segment in raw:
            # Also split on newlines (common artefact in PDF-extracted text)
            for line in segment.split("\n"):
                line = line.replace(_PLACEHOLDER, ".").strip()
                if len(line) > 15:   # filter out noise, page numbers, stray chars
                    sentences.append(line)

        return sentences

    def _is_section_boundary(self, text: str) -> bool:
        stripped = text.strip()
        return any(stripped.startswith(m) for m in FINANCIAL_SECTION_MARKERS)

    # ------------------------------------------------------------------ #
    # Semantic breakpoint detection                                        #
    # ------------------------------------------------------------------ #

    def _build_windows(self, sentences: List[str]) -> List[str]:
        """
        Concatenate `buffer_size` neighbours on each side of every sentence.
        Embedding the window instead of the bare sentence reduces noise from
        short or ambiguous sentences at topic boundaries.
        """
        windows = []
        n = len(sentences)
        for i in range(n):
            start = max(0, i - self.buffer_size)
            end = min(n, i + self.buffer_size + 1)
            windows.append(" ".join(sentences[start:end]))
        return windows

    def _find_semantic_breakpoints(self, embeddings: np.ndarray) -> List[int]:
        if len(embeddings) < 2:
            return []

        # Cosine distance between consecutive windows
        # (embeddings are L2-normalised by the Embedder, so dot product = cosine sim)
        distances = [
            1.0 - float(np.dot(embeddings[i], embeddings[i + 1]))
            for i in range(len(embeddings) - 1)
        ]

        threshold = float(np.percentile(distances, self.breakpoint_percentile))
        return [i for i, d in enumerate(distances) if d >= threshold]

    # ------------------------------------------------------------------ #
    # Chunk assembly & size guards                                         #
    # ------------------------------------------------------------------ #

    def _sentences_to_chunks(
        self, sentences: List[str], breakpoints: List[int]
    ) -> List[str]:
        breakpoints = sorted(set(breakpoints))
        chunks: List[str] = []
        start = 0

        for bp in breakpoints:
            segment = " ".join(sentences[start : bp + 1]).strip()
            if segment:
                chunks.append(segment)
            start = bp + 1

        if start < len(sentences):
            segment = " ".join(sentences[start:]).strip()
            if segment:
                chunks.append(segment)

        return chunks

    def _merge_tiny_chunks(self, chunks: List[str]) -> List[str]:
        """Merge any chunk below min_chunk_size into its preceding neighbour."""
        merged: List[str] = []
        for chunk in chunks:
            if merged and len(chunk) < self.min_chunk_size:
                merged[-1] = merged[-1] + " " + chunk
            else:
                merged.append(chunk)
        return merged

    def _split_oversized(self, text: str) -> List[str]:
        """Sentence-boundary fallback for chunks that exceed max_chunk_size."""
        sentences = self._split_sentences(text)
        result: List[str] = []
        current = ""
        for s in sentences:
            if current and len(current) + len(s) + 1 > self.max_chunk_size:
                result.append(current.strip())
                current = s
            else:
                current = (current + " " + s).strip()
        if current:
            result.append(current)
        return result

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def split_text(self, text: str) -> List[str]:
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        # Too few sentences to compute distances — return whole text if large enough
        if len(sentences) <= 3:
            return [text.strip()] if len(text.strip()) >= self.min_chunk_size else []

        # Embed sentence windows
        windows = self._build_windows(sentences)
        embeddings = np.array(self.embedder.embed(windows))

        # Semantic breakpoints from cosine distance
        breakpoints = self._find_semantic_breakpoints(embeddings)

        # Hard-force breaks at known financial section headers
        for i, sent in enumerate(sentences):
            if self._is_section_boundary(sent) and i > 0:
                breakpoints.append(i - 1)

        # Assemble → merge tiny → split oversized
        raw_chunks = self._sentences_to_chunks(sentences, breakpoints)
        merged = self._merge_tiny_chunks(raw_chunks)

        final: List[str] = []
        for chunk in merged:
            if len(chunk) > self.max_chunk_size:
                final.extend(self._split_oversized(chunk))
            elif len(chunk) >= self.min_chunk_size:
                final.append(chunk)

        return final

    def chunk_documents(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_chunks: List[Dict[str, Any]] = []

        for page in pages:
            text_chunks = self.split_text(page["content"])
            for i, chunk_text in enumerate(text_chunks):
                all_chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        **page["metadata"],
                        "chunk_index": i,
                        "chunk_total": len(text_chunks),
                    },
                })

        logger.info(
            f"Created {len(all_chunks)} semantic chunks from {len(pages)} pages"
        )
        return all_chunks
