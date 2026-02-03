#!/usr/bin/env python3
"""
Build FAISS index for translation memory retrieval.

Loads the combined corpus, embeds English translations, and builds a FAISS
index for fast k-nearest neighbor search.

Usage:
    # Build index from combined corpus
    python scripts/build_index.py

    # Build from custom corpus
    python scripts/build_index.py --corpus data/processed/train_sentences.csv

    # Use different embedder model
    python scripts/build_index.py --model all-mpnet-base-v2
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.io import setup_logging
from src.retrieval import Embedder, IndexBuilder
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_CORPUS = "data/processed/combined_corpus.csv"
DEFAULT_OUTPUT_DIR = "data/indices"
DEFAULT_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def build_index(
    corpus_path: str,
    output_dir: str,
    embedder_model: str,
) -> None:
    """
    Build FAISS index from corpus.

    Args:
        corpus_path: Path to corpus CSV (with transliteration, translation columns)
        output_dir: Directory to save index and embeddings
        embedder_model: Sentence-transformers model name
    """
    setup_logging(level="INFO")

    logger.info("=" * 60)
    logger.info("BUILDING FAISS INDEX")
    logger.info("=" * 60)

    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load corpus
    # ------------------------------------------------------------------
    logger.info(f"Loading corpus from {corpus_path}...")
    corpus_df = pd.read_csv(corpus_path)
    logger.info(f"Loaded {len(corpus_df)} parallel pairs")

    # Validate columns
    if "translation" not in corpus_df.columns:
        raise ValueError(f"Corpus missing 'translation' column")

    # Extract English translations
    translations = corpus_df["translation"].fillna("").tolist()
    logger.info(f"Extracted {len(translations)} English translations")

    # ------------------------------------------------------------------
    # 2. Embed translations
    # ------------------------------------------------------------------
    logger.info(f"Loading embedder model: {embedder_model}...")
    embedder = Embedder(model_name=embedder_model)
    embedding_dim = embedder.get_embedding_dim()
    logger.info(f"Embedding dimension: {embedding_dim}")

    logger.info("Embedding translations...")
    embeddings = embedder.embed(translations, batch_size=32, show_progress=True)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Save embeddings
    embeddings_path = output_dir / "embeddings.npy"
    embedder.save_embeddings(embeddings, str(embeddings_path))

    # ------------------------------------------------------------------
    # 3. Build FAISS index
    # ------------------------------------------------------------------
    logger.info("Building FAISS index...")
    builder = IndexBuilder(dimension=embedding_dim)

    # Use flat index for small-medium corpora (<100k)
    # Use IVF for large corpora (>100k)
    index_type = "flat" if len(embeddings) < 100000 else "ivf"
    builder.build(embeddings, index_type=index_type)

    # Save index
    index_path = output_dir / "faiss_index.bin"
    builder.save(str(index_path))

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("INDEX BUILD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Corpus size:        {len(corpus_df)} pairs")
    logger.info(f"Embedding dim:      {embedding_dim}")
    logger.info(f"Index type:         {index_type}")
    logger.info(f"Embeddings saved:   {embeddings_path}")
    logger.info(f"Index saved:        {index_path}")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS index for translation memory retrieval"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=DEFAULT_CORPUS,
        help=f"Path to corpus CSV (default: {DEFAULT_CORPUS})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for index (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Sentence-transformers model (default: {DEFAULT_MODEL})",
    )

    args = parser.parse_args()

    try:
        build_index(
            corpus_path=args.corpus,
            output_dir=args.output,
            embedder_model=args.model,
        )
    except Exception as e:
        logger.error(f"Index build failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
