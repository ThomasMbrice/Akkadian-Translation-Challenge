"""
Deduplication for extracted publication data.

Removes pairs that duplicate existing training data or duplicate each other
within the extracted set.  Uses character trigram fingerprinting with a
trigram index for efficient candidate lookup — avoids an O(n*m) full scan
when the existing corpus is large.

Similarity metric: Jaccard similarity over character trigrams of the
normalised transliteration.  A threshold of 0.7 catches near-duplicates
despite OCR variation, while rejecting genuinely different texts.
"""

import re
import logging
from typing import Dict, List, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trigram helpers
# ---------------------------------------------------------------------------


def _normalize_for_comparison(text: str) -> str:
    """
    Normalise text for deduplication comparison.

    Lowercases, strips all whitespace and common punctuation so that minor
    OCR or formatting differences don't prevent duplicate detection.
    """
    text = text.lower()
    text = re.sub(r"\s+", "", text)
    text = text.replace("!", "").replace("?", "").replace(".", "")
    return text


def _get_trigrams(text: str) -> Set[str]:
    """Extract the set of character trigrams from normalised text."""
    normalised = _normalize_for_comparison(text)
    if len(normalised) < 3:
        return set()
    return {normalised[i : i + 3] for i in range(len(normalised) - 2)}


def _jaccard(set_a: Set[str], set_b: Set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Deduplicator
# ---------------------------------------------------------------------------


class Deduplicator:
    """
    Remove duplicate transliteration-translation pairs.

    Workflow:
        1. ``load_existing(df)``  — index the current training corpus.
        2. ``deduplicate(new_df)`` — filter a DataFrame of extracted pairs,
           removing anything too similar to existing data or to an earlier
           row in the same batch.
    """

    def __init__(self, similarity_threshold: float = 0.7):
        """
        Args:
            similarity_threshold: Jaccard similarity above which two
                transliterations are considered duplicates.  0.7 is robust
                to minor OCR variation while distinguishing genuinely
                different texts.
        """
        self.similarity_threshold = similarity_threshold

        # Indexed storage for existing training data
        self._existing_trigrams: List[Set[str]] = []
        # Inverted index: trigram → list of entry indices
        self._trigram_index: Dict[str, List[int]] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_existing(
        self,
        existing_df: pd.DataFrame,
        transliteration_col: str = "transliteration",
    ) -> None:
        """
        Load existing training data and build the trigram index.

        Args:
            existing_df: DataFrame containing the current training pairs.
            transliteration_col: Name of the transliteration column.
        """
        self._existing_trigrams = []
        self._trigram_index = {}

        for _, row in existing_df.iterrows():
            text = str(row[transliteration_col])
            trigrams = _get_trigrams(text)
            if not trigrams:
                continue

            idx = len(self._existing_trigrams)
            self._existing_trigrams.append(trigrams)

            for tri in trigrams:
                self._trigram_index.setdefault(tri, []).append(idx)

        logger.info(
            f"Loaded {len(self._existing_trigrams)} existing transliterations "
            f"into dedup index ({len(self._trigram_index)} unique trigrams)"
        )

    # ------------------------------------------------------------------
    # Duplicate checking
    # ------------------------------------------------------------------

    def is_duplicate(self, transliteration: str) -> bool:
        """
        Check whether *transliteration* is a near-duplicate of any
        previously loaded existing transliteration.

        Uses the inverted trigram index to restrict Jaccard computation
        to only those existing entries that share at least one trigram
        with the candidate — dramatically reducing unnecessary comparisons.

        Args:
            transliteration: The transliteration string to check.

        Returns:
            True if a duplicate exists above the similarity threshold.
        """
        new_trigrams = _get_trigrams(transliteration)
        if not new_trigrams:
            return True  # Too short to be useful — treat as duplicate

        # Collect candidate indices via the inverted index, counting how
        # many trigrams each candidate shares with the new text.
        candidate_shared: Dict[int, int] = {}
        for tri in new_trigrams:
            for idx in self._trigram_index.get(tri, []):
                candidate_shared[idx] = candidate_shared.get(idx, 0) + 1

        # Evaluate only candidates that could exceed the threshold.
        # Jaccard = intersection / union, union = |A| + |B| - intersection
        for idx, shared in candidate_shared.items():
            existing = self._existing_trigrams[idx]
            union = len(new_trigrams) + len(existing) - shared
            if union > 0 and shared / union >= self.similarity_threshold:
                return True

        return False

    # ------------------------------------------------------------------
    # Batch deduplication
    # ------------------------------------------------------------------

    def deduplicate(
        self,
        extracted_df: pd.DataFrame,
        transliteration_col: str = "transliteration",
    ) -> pd.DataFrame:
        """
        Remove duplicates from a DataFrame of extracted pairs.

        Two passes of removal:
        1. Against the existing training corpus (loaded via ``load_existing``).
        2. Internal deduplication within *extracted_df* itself (first
           occurrence wins).

        Args:
            extracted_df: DataFrame of newly extracted pairs.
            transliteration_col: Column name for transliterations.

        Returns:
            Deduplicated DataFrame with reset index.
        """
        if extracted_df.empty:
            return extracted_df

        initial_count = len(extracted_df)
        dup_existing = 0
        dup_internal = 0

        # Track trigram sets already accepted (for internal dedup)
        seen: List[Set[str]] = []
        keep_mask: List[bool] = []

        for _, row in extracted_df.iterrows():
            translit = str(row[transliteration_col])
            new_trigrams = _get_trigrams(translit)

            # --- Check against existing training data ---
            if self.is_duplicate(translit):
                keep_mask.append(False)
                dup_existing += 1
                continue

            # --- Check against already-accepted extracted pairs ---
            is_internal_dup = False
            for seen_trigrams in seen:
                if _jaccard(new_trigrams, seen_trigrams) >= self.similarity_threshold:
                    is_internal_dup = True
                    break

            if is_internal_dup:
                keep_mask.append(False)
                dup_internal += 1
            else:
                keep_mask.append(True)
                seen.append(new_trigrams)

        result = extracted_df[keep_mask].reset_index(drop=True)

        logger.info(
            f"Deduplication: {initial_count} → {len(result)} pairs "
            f"(removed {dup_existing} existing dups, {dup_internal} internal dups)"
        )
        return result
