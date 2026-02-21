#!/usr/bin/env python3
"""
Retrieval Quality Diagnostic for Akkadian NMT RAG System.

Analyzes retrieval performance on the validation set:
1. Retrieval quality metrics (genre precision, content overlap)
2. Deduplication analysis (near-duplicate detection)
3. Per-query detailed diagnostics

Usage:
    python scripts/diagnose_retrieval.py
    python scripts/diagnose_retrieval.py --k 5  # change number of retrieved examples
    python scripts/diagnose_retrieval.py --verbose  # show detailed per-query output
"""

# Must be set before importing torch or faiss to prevent OpenMP runtime conflict
# (FAISS and PyTorch each ship their own OpenMP; duplicate libs crash on macOS/Linux)
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import logging
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Set
import re

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from difflib import SequenceMatcher

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import setup_logging, load_yaml
from src.retrieval import Retriever
from src.retrieval.genre import classify_genre as classify_genre_akkadian

logger = logging.getLogger(__name__)


# ============================================================================
# Genre Detection
# ============================================================================

def detect_genre(transliteration: str, translation: str) -> str:
    """
    Heuristically detect document genre from content.

    Old Assyrian texts typically fall into these categories:
    - Legal: contracts, seals, debt documents
    - Letters: correspondence (um-ma pattern, qí-bi-ma)
    - Commercial: inventories, price lists, transactions
    - Administrative: lists, records

    Returns:
        Genre label: "legal", "letter", "commercial", "administrative", "unknown"
    """
    trans_lower = transliteration.lower()
    transl_lower = translation.lower()

    # Legal documents: start with KIŠIB (seal), contain debt/payment terms
    if trans_lower.startswith("kišib") or "kišib" in trans_lower[:50]:
        return "legal"

    if any(word in transl_lower for word in ["seal of", "debt", "owes", "contract", "witnessed by"]):
        return "legal"

    # Letters: contain epistolary formula (um-ma ... qí-bi-ma)
    if "um-ma" in trans_lower and "qí-bi" in trans_lower:
        return "letter"

    if "from " in transl_lower[:100] and " to " in transl_lower[:100]:
        # Pattern: "From X to Y" at start
        return "letter"

    # Commercial: inventories, prices, quantities
    commercial_markers = ["price of", "minas", "shekels", "textiles", "copper", "silver", "received"]
    if sum(1 for marker in commercial_markers if marker in transl_lower) >= 3:
        return "commercial"

    # Administrative: lists, multiple entries
    if transl_lower.count(";") >= 3 or transl_lower.count("\n") >= 5:
        return "administrative"

    return "unknown"


# ============================================================================
# Content Word Extraction
# ============================================================================

def extract_content_words(translation: str, min_length: int = 4) -> Set[str]:
    """
    Extract content words from English translation.

    Filters out:
    - Stop words (common function words)
    - Very short words (likely articles, prepositions)
    - Punctuation

    Args:
        translation: English translation text
        min_length: Minimum word length to include

    Returns:
        Set of normalized content words
    """
    # Common stop words in Old Assyrian translations
    STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would",
        "should", "could", "may", "might", "must", "can", "shall",
        "he", "she", "it", "they", "we", "you", "i", "me", "him", "her",
        "them", "us", "his", "her", "its", "their", "our", "your", "my",
        "this", "that", "these", "those", "which", "what", "who", "whom",
        "if", "when", "where", "why", "how", "not", "no", "yes",
    }

    # Tokenize and normalize
    words = re.findall(r'\b[a-z]+\b', translation.lower())

    # Filter
    content_words = {
        word for word in words
        if len(word) >= min_length and word not in STOP_WORDS
    }

    return content_words


def compute_content_overlap(words1: Set[str], words2: Set[str]) -> float:
    """
    Compute Jaccard similarity of content words.

    Args:
        words1, words2: Sets of content words

    Returns:
        Jaccard coefficient (0-1)
    """
    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


# ============================================================================
# Deduplication Analysis
# ============================================================================

def compute_text_similarity(text1: str, text2: str) -> float:
    """
    Compute character-level similarity between two texts.

    Uses SequenceMatcher for approximate string matching.

    Args:
        text1, text2: Texts to compare

    Returns:
        Similarity ratio (0-1)
    """
    return SequenceMatcher(None, text1, text2).ratio()


def find_near_duplicates(
    corpus: pd.DataFrame,
    threshold: float = 0.85,
    sample_size: int = 500,
) -> List[tuple]:
    """
    Find near-duplicate pairs in corpus (sampled for performance).

    Args:
        corpus: DataFrame with transliteration column
        threshold: Similarity threshold for near-duplicate detection
        sample_size: Max number of texts to compare (for performance)

    Returns:
        List of (idx1, idx2, similarity) tuples
    """
    logger.info(f"Scanning for near-duplicates (threshold={threshold}, sampling {sample_size} texts)...")

    duplicates = []
    texts = corpus["transliteration"].fillna("").tolist()

    # Sample for performance (comparing all pairs is O(n²))
    n = min(sample_size, len(texts))
    indices = np.random.choice(len(texts), size=n, replace=False) if len(texts) > sample_size else list(range(len(texts)))

    sampled_texts = [(idx, texts[idx]) for idx in indices]

    comparisons = 0
    for i, (idx1, text1) in enumerate(sampled_texts):
        for idx2, text2 in sampled_texts[i + 1:]:
            comparisons += 1
            if comparisons % 10000 == 0:
                logger.info(f"  ...{comparisons} comparisons")

            sim = compute_text_similarity(text1, text2)
            if sim >= threshold:
                duplicates.append((idx1, idx2, sim))

    logger.info(f"Found {len(duplicates)} near-duplicate pairs (from {comparisons} comparisons)")
    return duplicates


# ============================================================================
# Retrieval Diagnostics
# ============================================================================

def diagnose_query(
    query_idx: int,
    query_text: str,
    query_translation: str,
    retrieved: List[Dict],
    corpus: pd.DataFrame,
    duplicate_set: Set[int],
) -> Dict:
    """
    Analyze retrieval quality for a single query.

    Args:
        query_idx: Index of query in corpus
        query_text: Akkadian query text
        query_translation: English translation
        retrieved: List of retrieved results from Retriever
        corpus: Full corpus DataFrame
        duplicate_set: Set of corpus indices that are near-duplicates

    Returns:
        Dict with diagnostic metrics
    """
    # Use Akkadian-side genre classifier (not English-side detect_genre)
    query_genre = classify_genre_akkadian(query_text)
    query_words = extract_content_words(query_translation)

    genre_matches = 0
    content_overlaps = []
    self_retrieval = False
    duplicate_retrievals = 0
    retrieved_indices = set()

    for result in retrieved:
        ret_idx = result["corpus_index"]
        ret_text = result["transliteration"]
        ret_transl = result["translation"]

        # Check for self-retrieval
        if ret_idx == query_idx:
            self_retrieval = True
            continue

        # Check for duplicate retrieval
        if ret_idx in retrieved_indices:
            duplicate_retrievals += 1
        retrieved_indices.add(ret_idx)

        # Check if retrieved item is a near-duplicate of query
        if ret_idx in duplicate_set:
            duplicate_retrievals += 1

        # Genre matching — use genre label from retriever result if available
        ret_genre = result.get("genre") or classify_genre_akkadian(ret_text)
        if ret_genre == query_genre and query_genre != "unknown":
            genre_matches += 1

        # Content overlap
        ret_words = extract_content_words(ret_transl)
        overlap = compute_content_overlap(query_words, ret_words)
        content_overlaps.append(overlap)

    # Compute metrics
    n_valid = len([r for r in retrieved if r["corpus_index"] != query_idx])

    return {
        "query_idx": query_idx,
        "query_genre": query_genre,
        "genre_precision": genre_matches / n_valid if n_valid > 0 else 0.0,
        "avg_content_overlap": np.mean(content_overlaps) if content_overlaps else 0.0,
        "self_retrieval": self_retrieval,
        "duplicate_retrievals": duplicate_retrievals,
        "n_retrieved": len(retrieved),
    }


def run_diagnostic(
    retriever: Retriever,
    val_df: pd.DataFrame,
    corpus: pd.DataFrame,
    k: int = 3,
    verbose: bool = False,
) -> Dict:
    """
    Run full retrieval diagnostic on validation set.

    Args:
        retriever: Loaded Retriever instance
        val_df: Validation DataFrame
        corpus: Full corpus DataFrame
        k: Number of examples to retrieve
        verbose: Print detailed per-query diagnostics

    Returns:
        Dict with aggregate metrics
    """
    logger.info(f"Running retrieval diagnostic on {len(val_df)} validation examples (k={k})...")

    # Find near-duplicates in corpus
    duplicate_pairs = find_near_duplicates(corpus, threshold=0.85)
    duplicate_set = set()
    for i, j, _ in duplicate_pairs:
        duplicate_set.add(i)
        duplicate_set.add(j)

    # Batch retrieve using BM25 genre-filtered retrieval
    queries = val_df["transliteration"].fillna("").tolist()
    all_results = retriever.retrieve_bm25_batch(queries, k=k)

    # Analyze each query
    diagnostics = []
    for idx, (query_text, query_transl, results) in enumerate(zip(
        val_df["transliteration"],
        val_df["translation"],
        all_results,
    )):
        query_idx = -1  # Unknown (val split doesn't track corpus indices)

        diag = diagnose_query(
            query_idx=query_idx,
            query_text=query_text,
            query_translation=query_transl,
            retrieved=results,
            corpus=corpus,
            duplicate_set=duplicate_set,
        )
        diagnostics.append(diag)

        if verbose:
            print(f"\n{'='*80}")
            print(f"Query {idx}: {query_text[:60]}...")
            print(f"Genre: {diag['query_genre']}")
            print(f"Genre precision: {diag['genre_precision']:.1%}")
            print(f"Avg content overlap: {diag['avg_content_overlap']:.1%}")
            print(f"Duplicate retrievals: {diag['duplicate_retrievals']}")
            print(f"\nTop-{k} retrieved:")
            for i, res in enumerate(results[:k], 1):
                score = res.get('score', res.get('distance', 0))
                print(f"  {i}. [{score:.3f}] {res['transliteration'][:50]}...")

    # Aggregate metrics
    genre_precisions = [d["genre_precision"] for d in diagnostics]
    content_overlaps = [d["avg_content_overlap"] for d in diagnostics]
    self_retrievals = sum(d["self_retrieval"] for d in diagnostics)
    duplicate_retrievals = sum(d["duplicate_retrievals"] for d in diagnostics)

    # Genre distribution
    genre_counts = Counter(d["query_genre"] for d in diagnostics)

    summary = {
        "n_queries": len(val_df),
        "k": k,
        "avg_genre_precision": np.mean(genre_precisions),
        "std_genre_precision": np.std(genre_precisions),
        "avg_content_overlap": np.mean(content_overlaps),
        "std_content_overlap": np.std(content_overlaps),
        "n_self_retrievals": self_retrievals,
        "n_duplicate_retrievals": duplicate_retrievals,
        "n_near_duplicates_in_corpus": len(duplicate_set),
        "genre_distribution": dict(genre_counts),
    }

    return summary, diagnostics


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose retrieval quality")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training.yaml",
        help="Path to training config",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of examples to retrieve",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-query diagnostics",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(log_file="logs/retrieval_diagnostic.log", level="INFO")

    logger.info("="*80)
    logger.info("RETRIEVAL QUALITY DIAGNOSTIC")
    logger.info("="*80)

    # Load config
    config = load_yaml(str(PROJECT_ROOT / args.config))

    # Load corpus
    corpus_path = PROJECT_ROOT / config["data"]["train_corpus"]
    logger.info(f"Loading corpus: {corpus_path}")
    corpus = pd.read_csv(corpus_path)
    logger.info(f"Corpus size: {len(corpus)}")

    # Split to get validation set (same logic as train.py)
    val_frac = config["data"]["validation_split"]
    test_frac = config["data"]["test_split"]
    seed = config.get("seed", 42)

    train_val, _ = train_test_split(corpus, test_size=test_frac, random_state=seed)
    val_ratio = val_frac / (1.0 - test_frac)
    _, val_df = train_test_split(train_val, test_size=val_ratio, random_state=seed)

    logger.info(f"Validation set size: {len(val_df)}")

    # Load retrieval system
    ret_cfg = config.get("retrieval", {})
    corpus_path_str = str(PROJECT_ROOT / ret_cfg["corpus_path"])
    index_path_str = str(PROJECT_ROOT / ret_cfg["index_path"])

    retriever = Retriever()
    retriever.load(corpus_path=corpus_path_str, index_path=index_path_str)

    # Run diagnostic
    summary, diagnostics = run_diagnostic(
        retriever=retriever,
        val_df=val_df,
        corpus=corpus,
        k=args.k,
        verbose=args.verbose,
    )

    # Print summary
    summary_text = []
    summary_text.append("\n" + "="*80)
    summary_text.append("RETRIEVAL DIAGNOSTIC SUMMARY")
    summary_text.append("="*80)
    summary_text.append(f"\nValidation set size: {summary['n_queries']}")
    summary_text.append(f"Top-k retrieved: {summary['k']}")
    summary_text.append(f"\n--- Quality Metrics ---")
    summary_text.append(f"Average genre precision: {summary['avg_genre_precision']:.1%} ± {summary['std_genre_precision']:.1%}")
    summary_text.append(f"Average content overlap: {summary['avg_content_overlap']:.1%} ± {summary['std_content_overlap']:.1%}")
    summary_text.append(f"\n--- Deduplication Analysis ---")
    summary_text.append(f"Near-duplicates in corpus: {summary['n_near_duplicates_in_corpus']} ({100*summary['n_near_duplicates_in_corpus']/len(corpus):.1f}%)")
    summary_text.append(f"Self-retrievals: {summary['n_self_retrievals']}")
    summary_text.append(f"Duplicate retrievals: {summary['n_duplicate_retrievals']}")
    summary_text.append(f"\n--- Genre Distribution ---")
    for genre, count in sorted(summary['genre_distribution'].items(), key=lambda x: -x[1]):
        summary_text.append(f"{genre:15s}: {count:4d} ({100*count/summary['n_queries']:5.1f}%)")

    # Performance targets
    summary_text.append(f"\n--- Performance Targets ---")
    genre_prec = summary['avg_genre_precision']
    if genre_prec >= 0.85:
        summary_text.append(f"✓ Genre precision {genre_prec:.1%} >= 85% target")
    elif genre_prec >= 0.60:
        summary_text.append(f"⚠ Genre precision {genre_prec:.1%} meets 60% baseline but below 85% target")
    else:
        summary_text.append(f"✗ Genre precision {genre_prec:.1%} below 60% baseline")

    content_ovl = summary['avg_content_overlap']
    if content_ovl >= 0.20:
        summary_text.append(f"✓ Content overlap {content_ovl:.1%} >= 20% target")
    else:
        summary_text.append(f"⚠ Content overlap {content_ovl:.1%} below 20% target")

    if summary['n_duplicate_retrievals'] > 0:
        summary_text.append(f"⚠ Found {summary['n_duplicate_retrievals']} duplicate retrievals - deduplication needed")
    else:
        summary_text.append(f"✓ No duplicate retrievals detected")

    summary_text.append("\n" + "="*80)

    # Print to stdout
    for line in summary_text:
        print(line)
    sys.stdout.flush()

    # Save to file
    output_file = PROJECT_ROOT / "logs" / "retrieval_diagnostic_summary.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write("\n".join(summary_text))
    logger.info(f"Summary saved to {output_file}")
    logger.info("Diagnostic complete")


if __name__ == "__main__":
    main()
