"""
Cross-encoder re-ranking for retrieved document chunks.

Uses sentence-transformers cross-encoder models to re-score retrieved
chunks against the exact query. Much more accurate than bi-encoder
similarity for relevance ranking.

Models (choose by speed vs accuracy tradeoff):
- Fast:     cross-encoder/ms-marco-MiniLM-L-2-v2  (~20ms/query)
- Balanced: cross-encoder/ms-marco-MiniLM-L-6-v2  (~50ms/query)  ← default
- Accurate: cross-encoder/ms-marco-MiniLM-L-12-v2 (~120ms/query)
"""

import logging
import time
from typing import List, Tuple, Optional
from langchain.schema import Document
import numpy as np

log = logging.getLogger(__name__)

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Re-ranks a list of retrieved documents using a cross-encoder model.

    The cross-encoder jointly encodes (query, document) pairs and produces
    a single relevance score per pair — significantly more accurate than
    the dot-product similarity used during initial retrieval.

    Usage:
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(query="What is the refund policy?", docs=retrieved_docs, top_k=5)
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None  # Lazy load

    @property
    def model(self):
        """Lazy-load the cross-encoder model on first use."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                log.info(f"Loading re-ranker: {self.model_name}")
                self._model = CrossEncoder(self.model_name, device=self.device)
                log.info("Re-ranker loaded successfully")
            except ImportError:
                log.warning("sentence-transformers not installed. Re-ranking disabled.")
                self._model = None
        return self._model

    def rerank(
        self,
        query: str,
        docs: List[Document],
        top_k: Optional[int] = None,
        score_threshold: float = -10.0
    ) -> List[Tuple[Document, float]]:
        """
        Re-rank documents by cross-encoder relevance score.

        Args:
            query: User query string
            docs: Candidate documents from initial retrieval
            top_k: Number of results to return (None = all)
            score_threshold: Minimum score to include a document

        Returns:
            List of (Document, score) tuples, sorted by score descending
        """
        if not docs:
            return []

        if self.model is None:
            # Fallback: return original order with placeholder scores
            return [(doc, float(i)) for i, doc in enumerate(reversed(docs))]

        t0 = time.time()
        pairs = [(query, doc.page_content[:512]) for doc in docs]
        scores = self.model.predict(pairs, show_progress_bar=False)
        elapsed = (time.time() - t0) * 1000
        log.debug(f"Re-ranked {len(docs)} docs in {elapsed:.0f}ms")

        # Pair docs with scores, filter, sort
        scored = [(doc, float(score)) for doc, score in zip(docs, scores)
                  if float(score) >= score_threshold]
        scored.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            scored = scored[:top_k]

        return scored

    def get_relevance_scores(self, query: str, docs: List[Document]) -> np.ndarray:
        """Return raw relevance scores array (useful for evaluation)."""
        if self.model is None or not docs:
            return np.zeros(len(docs))
        pairs = [(query, doc.page_content[:512]) for doc in docs]
        return self.model.predict(pairs, show_progress_bar=False)


def reciprocal_rank_fusion(
    result_lists: List[List[Document]],
    k: int = 60
) -> List[Document]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion (RRF).

    RRF score for document d = Σ 1/(k + rank(d)) across all lists.
    Documents appearing in multiple lists get boosted.

    Args:
        result_lists: List of ranked document lists (e.g., [semantic_results, bm25_results])
        k: RRF constant (default 60 per original RRF paper)

    Returns:
        Merged, deduplicated list sorted by RRF score
    """
    scores: dict = {}
    doc_map: dict = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list, start=1):
            # Use content hash as key for deduplication
            key = hash(doc.page_content[:200])
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            doc_map[key] = doc  # keep most recent metadata

    sorted_keys = sorted(scores, key=scores.get, reverse=True)
    return [doc_map[k] for k in sorted_keys]
