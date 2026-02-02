"""
FAISS index builder for translation memory retrieval.

Builds a FAISS index over English embeddings for fast k-nearest neighbor
search. Given a query embedding, retrieves the k most similar translations
from the corpus.

Usage:
    builder = IndexBuilder(dimension=384)
    builder.build(embeddings)
    builder.save("data/indices/faiss_index.bin")

    # Later
    builder.load("data/indices/faiss_index.bin")
    distances, indices = builder.search(query_embedding, k=5)
"""

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import faiss

logger = logging.getLogger(__name__)


class IndexBuilder:
    """
    FAISS index builder for translation memory.

    Uses FAISS IndexFlatL2 for exact L2 distance search.
    For larger corpora (>100k), consider IndexIVFFlat or IndexHNSW.
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: Embedding dimension (must match embedder output)
        """
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.n_vectors = 0

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def build(
        self,
        embeddings: np.ndarray,
        index_type: str = "flat",
    ) -> None:
        """
        Build FAISS index from embeddings.

        Args:
            embeddings: Numpy array of shape (n_vectors, dimension)
            index_type: "flat" (exact search) or "ivf" (approximate, faster for large corpora)
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, "
                f"got {embeddings.shape[1]}"
            )

        n_vectors = embeddings.shape[0]
        logger.info(
            f"Building FAISS index ({index_type}) for {n_vectors} vectors "
            f"of dimension {self.dimension}..."
        )

        # Ensure embeddings are float32 (FAISS requirement)
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        # Build index
        if index_type == "flat":
            # Exact L2 search (good for small-medium corpora)
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings)

        elif index_type == "ivf":
            # Approximate search with IVF (faster for large corpora)
            # Use sqrt(n) clusters as heuristic
            n_clusters = min(int(np.sqrt(n_vectors)), 256)
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, n_clusters)

            # Train index
            logger.info(f"Training IVF index with {n_clusters} clusters...")
            self.index.train(embeddings)
            self.index.add(embeddings)

        else:
            raise ValueError(f"Unknown index type: {index_type}")

        self.n_vectors = n_vectors
        logger.info(f"Index built successfully. Total vectors: {self.n_vectors}")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.

        Args:
            query_embedding: Query vector of shape (dimension,) or (1, dimension)
            k: Number of nearest neighbors to return

        Returns:
            Tuple of (distances, indices)
            - distances: shape (k,) - L2 distances to neighbors
            - indices: shape (k,) - indices of neighbors in original corpus
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")

        # Reshape query to (1, dimension) if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Ensure float32
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        # Search
        distances, indices = self.index.search(query_embedding, k)

        # Return as 1D arrays
        return distances[0], indices[0]

    def search_batch(
        self,
        query_embeddings: np.ndarray,
        k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors for multiple queries.

        Args:
            query_embeddings: Query vectors of shape (n_queries, dimension)
            k: Number of nearest neighbors to return per query

        Returns:
            Tuple of (distances, indices)
            - distances: shape (n_queries, k)
            - indices: shape (n_queries, k)
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")

        # Ensure float32
        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype(np.float32)

        # Search
        distances, indices = self.index.search(query_embeddings, k)
        return distances, indices

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, output_path: str) -> None:
        """
        Save FAISS index to disk.

        Args:
            output_path: Path to save index (.bin or .faiss)
        """
        if self.index is None:
            raise RuntimeError("No index to save. Call build() first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(output_path))

        # Save metadata (dimension, n_vectors) alongside
        meta_path = output_path.with_suffix(".meta.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(
                {
                    "dimension": self.dimension,
                    "n_vectors": self.n_vectors,
                },
                f,
            )

        logger.info(f"Saved FAISS index to {output_path}")
        logger.info(f"Saved metadata to {meta_path}")

    def load(self, input_path: str) -> None:
        """
        Load FAISS index from disk.

        Args:
            input_path: Path to index file (.bin or .faiss)
        """
        input_path = Path(input_path)

        # Load FAISS index
        self.index = faiss.read_index(str(input_path))

        # Load metadata
        meta_path = input_path.with_suffix(".meta.pkl")
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
                self.dimension = meta["dimension"]
                self.n_vectors = meta["n_vectors"]
        else:
            # Infer from index
            self.dimension = self.index.d
            self.n_vectors = self.index.ntotal

        logger.info(f"Loaded FAISS index from {input_path}")
        logger.info(f"Dimension: {self.dimension}, Vectors: {self.n_vectors}")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return index statistics."""
        if self.index is None:
            return {"built": False}

        return {
            "built": True,
            "dimension": self.dimension,
            "n_vectors": self.n_vectors,
            "index_type": type(self.index).__name__,
        }
