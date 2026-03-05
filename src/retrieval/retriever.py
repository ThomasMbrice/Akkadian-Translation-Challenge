"""
Translation memory retriever.

Combines lexicon, embeddings, and FAISS index to retrieve relevant
Akkadian-English pairs given a query.

Usage:
    retriever = Retriever()
    retriever.load(
        corpus_path="data/processed/combined_corpus.csv",
        index_path="data/indices/faiss_index.bin",
    )

    results = retriever.retrieve("a-na A-šùr-i-mì-tí qí-bi-ma", k=5)
    # Returns: List of dicts with transliteration, translation, distance
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.retrieval.lexicon import Lexicon
from src.retrieval.embedder import Embedder
from src.retrieval.index_builder import IndexBuilder
from src.retrieval.bm25_index import BM25Index
from src.retrieval.genre import classify_genre

logger = logging.getLogger(__name__)


class Retriever:
    """
    Translation memory retriever.

    Retrieves k-nearest Akkadian-English pairs given a query.
    """

    def __init__(
        self,
        embedder_model: str = "all-MiniLM-L6-v2",
        lexicon_fuzzy_threshold: float = 0.8,
    ):
        """
        Args:
            embedder_model: Sentence-transformers model name
            lexicon_fuzzy_threshold: Threshold for fuzzy lexicon matching
        """
        self.embedder_model = embedder_model
        self.lexicon_fuzzy_threshold = lexicon_fuzzy_threshold

        # Components
        self.lexicon: Optional[Lexicon] = None
        self.embedder: Optional[Embedder] = None
        self.index: Optional[IndexBuilder] = None
        self.corpus: Optional[pd.DataFrame] = None

        # BM25 retrieval (Akkadian-side)
        self.bm25: Optional[BM25Index] = None
        self.genre_labels: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(
        self,
        corpus_path: str,
        index_path: str,
        load_lexicon: bool = True,
    ) -> None:
        """
        Load all retrieval components.

        Args:
            corpus_path: Path to corpus CSV (with transliteration, translation columns)
            index_path: Path to FAISS index
            load_lexicon: Whether to load lexicon (for proper noun/Sumerogram lookup)
        """
        logger.info("=" * 60)
        logger.info("LOADING RETRIEVAL SYSTEM")
        logger.info("=" * 60)

        # Load corpus
        logger.info(f"Loading corpus from {corpus_path}...")
        self.corpus = pd.read_csv(corpus_path)
        logger.info(f"Loaded {len(self.corpus)} parallel pairs")

        # Validate columns
        required_cols = ["transliteration", "translation"]
        missing = [col for col in required_cols if col not in self.corpus.columns]
        if missing:
            raise ValueError(f"Corpus missing required columns: {missing}")

        # Load lexicon
        if load_lexicon:
            logger.info("Loading lexicon...")
            self.lexicon = Lexicon(fuzzy_threshold=self.lexicon_fuzzy_threshold)
            self.lexicon.load()
            logger.info(f"Lexicon stats: {self.lexicon.get_stats()}")

        # Load embedder
        logger.info(f"Loading embedder model: {self.embedder_model}...")
        self.embedder = Embedder(model_name=self.embedder_model)

        # Load index
        logger.info(f"Loading FAISS index from {index_path}...")
        self.index = IndexBuilder(dimension=self.embedder.get_embedding_dim())
        self.index.load(index_path)
        logger.info(f"Index stats: {self.index.get_stats()}")

        # Validate sizes match
        if len(self.corpus) != self.index.n_vectors:
            raise ValueError(
                f"Corpus size ({len(self.corpus)}) does not match index size "
                f"({self.index.n_vectors})"
            )

        # Build BM25 index on Akkadian side
        logger.info("Building BM25 index on Akkadian transliterations...")
        self.bm25 = BM25Index()
        texts = self.corpus["transliteration"].fillna("").tolist()
        self.bm25.build(texts)

        # Lookup: transliteration text → corpus index (for self-exclusion)
        self._text_to_index = {t: i for i, t in enumerate(texts)}

        # Pre-classify corpus by genre
        logger.info("Classifying corpus by genre...")
        self.genre_labels = [
            classify_genre(t) for t in self.corpus["transliteration"].fillna("")
        ]
        from collections import Counter
        logger.info(f"Corpus genre distribution: {dict(Counter(self.genre_labels))}")

        logger.info("=" * 60)
        logger.info("RETRIEVAL SYSTEM READY")
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int = 5,
        use_lexicon: bool = True,
    ) -> List[Dict]:
        """
        Retrieve k most similar Akkadian-English pairs.

        Strategy:
        1. If query is Akkadian: extract lexicon context, embed as English proxy
        2. If query is English: embed directly
        3. Search FAISS index for k-nearest neighbors
        4. Return corpus entries with distances

        Args:
            query: Akkadian or English query string
            k: Number of results to return
            use_lexicon: Whether to extract lexicon context from Akkadian queries

        Returns:
            List of dicts with keys: transliteration, translation, distance, rank
        """
        if self.corpus is None or self.embedder is None or self.index is None:
            raise RuntimeError("Retriever not loaded. Call load() first.")

        # Build query context
        query_text = query
        if use_lexicon and self.lexicon is not None:
            query_text = self._build_query_context(query)

        # Embed query
        query_embedding = self.embedder.embed_single(query_text)

        # Search index
        distances, indices = self.index.search(query_embedding, k=k)

        # Build results
        results = []
        for rank, (idx, dist) in enumerate(zip(indices, distances), start=1):
            if idx < 0 or idx >= len(self.corpus):
                # Invalid index (can happen with approximate search)
                continue

            row = self.corpus.iloc[idx]
            results.append({
                "transliteration": row["transliteration"],
                "translation": row["translation"],
                "distance": float(dist),
                "rank": rank,
                "corpus_index": int(idx),
            })

        return results

    def retrieve_batch(
        self,
        queries: List[str],
        k: int = 5,
        use_lexicon: bool = True,
    ) -> List[List[Dict]]:
        """
        Retrieve for multiple queries in batch.

        Args:
            queries: List of query strings
            k: Number of results per query
            use_lexicon: Whether to use lexicon for Akkadian queries

        Returns:
            List of result lists (one per query)
        """
        if not queries:
            return []

        # Build query contexts
        query_texts = []
        for query in queries:
            if use_lexicon and self.lexicon is not None:
                query_text = self._build_query_context(query)
            else:
                query_text = query
            query_texts.append(query_text)

        # Embed all queries
        query_embeddings = self.embedder.embed(
            query_texts,
            show_progress=len(queries) > 10,
        )

        # Search index (batch)
        distances_batch, indices_batch = self.index.search_batch(query_embeddings, k=k)

        # Build results for each query
        all_results = []
        for distances, indices in zip(distances_batch, indices_batch):
            results = []
            for rank, (idx, dist) in enumerate(zip(indices, distances), start=1):
                if idx < 0 or idx >= len(self.corpus):
                    continue

                row = self.corpus.iloc[idx]
                results.append({
                    "transliteration": row["transliteration"],
                    "translation": row["translation"],
                    "distance": float(dist),
                    "rank": rank,
                    "corpus_index": int(idx),
                })
            all_results.append(results)

        return all_results

    def retrieve_bm25(
        self,
        query: str,
        k: int = 3,
        min_score: float = 0.5,
    ) -> List[Dict]:
        """
        Genre-filtered BM25 retrieval on the Akkadian side.

        Strategy:
        1. Classify query genre
        2. Filter corpus to same genre (hard filter)
        3. Rank filtered pool by BM25 score
        4. Return up to k examples above min_score threshold
        5. If genre is "unknown" or no genre match, fall back to unfiltered BM25

        Args:
            query: Akkadian transliteration query
            k: Maximum number of results
            min_score: Minimum BM25 score to include (skip weak matches)

        Returns:
            List of dicts with transliteration, translation, score, rank, corpus_index
        """
        if self.bm25 is None or self.genre_labels is None:
            raise RuntimeError("BM25 index not loaded. Call load() first.")

        # Exclude the query document itself (prevents self-retrieval during
        # training/eval when the query is in the index).
        self_idx = getattr(self, "_text_to_index", {}).get(query)
        exclude = [self_idx] if self_idx is not None else None

        query_genre = classify_genre(query)

        # Build genre-filtered candidate index list
        if query_genre != "unknown":
            genre_indices = [
                i for i, g in enumerate(self.genre_labels) if g == query_genre
            ]
        else:
            genre_indices = None  # unfiltered fallback

        hits = self.bm25.search(query, k=k, indices=genre_indices,
                                min_score=min_score, exclude_indices=exclude)

        # If filtered search returned nothing, fall back to unfiltered
        if not hits and genre_indices is not None:
            hits = self.bm25.search(query, k=k, indices=None,
                                    min_score=min_score, exclude_indices=exclude)

        results = []
        for rank, (idx, score) in enumerate(hits, start=1):
            row = self.corpus.iloc[idx]
            results.append({
                "transliteration": row["transliteration"],
                "translation": row["translation"],
                "score": score,
                "rank": rank,
                "corpus_index": int(idx),
                "genre": self.genre_labels[idx],
            })

        return results

    def retrieve_bm25_batch(
        self,
        queries: List[str],
        k: int = 3,
        min_score: float = 0.5,
    ) -> List[List[Dict]]:
        """retrieve_bm25 for multiple queries."""
        return [self.retrieve_bm25(q, k=k, min_score=min_score) for q in queries]

    # ------------------------------------------------------------------
    # Query Context Building
    # ------------------------------------------------------------------

    def _build_query_context(self, akkadian_query: str) -> str:
        """
        Build English context from Akkadian query using lexicon.

        Strategy:
        1. Extract proper nouns → get normalized forms
        2. Extract Sumerograms → get English definitions
        3. Combine into pseudo-English query for embedding

        This is a heuristic to approximate semantic similarity when we
        can't directly embed Akkadian.

        Args:
            akkadian_query: Akkadian transliteration

        Returns:
            English context string for embedding
        """
        if self.lexicon is None:
            return akkadian_query

        context_parts = []

        # Extract proper nouns
        proper_nouns = self.lexicon.extract_proper_nouns(akkadian_query)
        if proper_nouns:
            pn_context = " ".join([pn["norm"] for pn in proper_nouns])
            context_parts.append(pn_context)

        # Extract Sumerograms
        sumerograms = self.lexicon.extract_sumerograms(akkadian_query)
        if sumerograms:
            sg_context = " ".join([sg["definition"] for sg in sumerograms])
            context_parts.append(sg_context)

        # Fallback: use original query if no lexicon matches
        if not context_parts:
            return akkadian_query

        return " ".join(context_parts)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_corpus_entry(self, index: int) -> Dict:
        """
        Get a specific corpus entry by index.

        Args:
            index: Corpus index

        Returns:
            Dict with transliteration and translation
        """
        if self.corpus is None:
            raise RuntimeError("Corpus not loaded")

        if index < 0 or index >= len(self.corpus):
            raise ValueError(f"Index {index} out of range [0, {len(self.corpus)})")

        row = self.corpus.iloc[index]
        return {
            "transliteration": row["transliteration"],
            "translation": row["translation"],
        }
