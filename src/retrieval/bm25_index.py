"""
BM25 index over Akkadian transliteration text.

Replaces FAISS semantic search for retrieval.
Matching on the Akkadian source fixes the fundamental problem of the old
approach (embedding English translations we haven't produced yet).
"""

import re
import logging
from typing import List, Tuple

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """
    Tokenize Akkadian transliteration.

    Splits on whitespace/punctuation, lowercases, keeps hyphens inside tokens
    (e.g. "a-na", "qí-bi-ma" are single tokens).
    """
    return re.findall(r'[^\s,;.()\[\]{}]+', text.lower())


class BM25Index:
    """BM25 index over a list of Akkadian texts."""

    def __init__(self):
        self.bm25 = None
        self.texts: List[str] = []

    def build(self, texts: List[str]) -> None:
        """
        Build BM25 index from a list of texts.

        Args:
            texts: Akkadian transliteration strings (one per corpus entry)
        """
        self.texts = texts
        tokenized = [_tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index built over {len(texts)} texts")

    def search(
        self,
        query: str,
        k: int = 5,
        indices: List[int] = None,
        min_score: float = 0.0,
    ) -> List[Tuple[int, float]]:
        """
        Search BM25 index for top-k results.

        Args:
            query: Akkadian query text
            k: Maximum number of results
            indices: Optional subset of corpus indices to restrict search to.
                     If None, searches the full index.
            min_score: Minimum BM25 score to include (threshold).

        Returns:
            List of (corpus_index, score) tuples, sorted by descending score.
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call build() first.")

        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # Restrict to genre-filtered subset if provided
        if indices is not None:
            candidates = [(i, float(scores[i])) for i in indices]
        else:
            candidates = list(enumerate(scores.tolist()))

        # Sort descending, apply threshold, cap at k
        candidates.sort(key=lambda x: -x[1])
        return [(i, s) for i, s in candidates[:k] if s >= min_score]
