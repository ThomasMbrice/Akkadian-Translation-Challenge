"""
English text embedder using sentence-transformers.

Since no pretrained Akkadian embedders exist, we embed the English side
of parallel pairs for retrieval. This is a heuristic: given an Akkadian
query, we can't directly embed it, but we can use lexicon lookups and
fuzzy matching to approximate semantic similarity.

Usage:
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    embeddings = embedder.embed(["This is a sentence.", "Another sentence."])
    # Returns: numpy array of shape (2, 384)
"""

import logging
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """
    Sentence embedder for English translations.

    Uses sentence-transformers for fast, high-quality embeddings.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            model_name: Sentence-transformers model name.
                Recommended: "all-MiniLM-L6-v2" (fast, 384-dim, good quality)
                Alternatives: "all-mpnet-base-v2" (slower, 768-dim, best quality)
            device: Device to use ("cuda", "cpu", or None for auto)
            cache_dir: Directory to cache model weights (default: ~/.cache)
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir

        logger.info(f"Loading sentence-transformers model: {model_name}")
        self.model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=cache_dir,
        )
        logger.info(
            f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}"
        )

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed a list of texts.

        Args:
            texts: List of English sentences/documents
            batch_size: Batch size for encoding (larger = faster, more memory)
            show_progress: Show progress bar

        Returns:
            Numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])

        logger.info(f"Embedding {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        logger.info(f"Embeddings shape: {embeddings.shape}")
        return embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """
        Embed a single text.

        Args:
            text: English sentence/document

        Returns:
            Numpy array of shape (embedding_dim,)
        """
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        output_path: str,
    ) -> None:
        """
        Save embeddings to disk.

        Args:
            embeddings: Numpy array of embeddings
            output_path: Path to save file (.npy or .pkl)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".npy":
            np.save(output_path, embeddings)
        elif output_path.suffix == ".pkl":
            with open(output_path, "wb") as f:
                pickle.dump(embeddings, f)
        else:
            raise ValueError(f"Unsupported file format: {output_path.suffix}")

        logger.info(f"Saved {embeddings.shape[0]} embeddings to {output_path}")

    def load_embeddings(self, input_path: str) -> np.ndarray:
        """
        Load embeddings from disk.

        Args:
            input_path: Path to embeddings file (.npy or .pkl)

        Returns:
            Numpy array of embeddings
        """
        input_path = Path(input_path)

        if input_path.suffix == ".npy":
            embeddings = np.load(input_path)
        elif input_path.suffix == ".pkl":
            with open(input_path, "rb") as f:
                embeddings = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")

        logger.info(f"Loaded {embeddings.shape[0]} embeddings from {input_path}")
        return embeddings

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension of the model."""
        return self.model.get_sentence_embedding_dimension()
