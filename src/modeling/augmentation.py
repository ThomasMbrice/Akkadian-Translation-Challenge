"""
Data augmentation for Old Assyrian transliterations.

Implements synthetic gap insertion to help the model learn to handle
lacunae (damaged portions of tablets), which are extremely common in
ancient texts.

Gap types:
- <gap>: Single missing sign
- <big_gap>: Multiple missing signs or entire sections

Usage:
    augmenter = Augmenter(gap_prob=0.3, big_gap_ratio=0.2)
    augmented = augmenter.add_gaps("a-na A-šùr-i-mì-tí DUMU Ṣí-lí-{d}UTU qí-bi-ma")
    # Returns: "a-na <gap> DUMU Ṣí-lí-{d}UTU qí-bi-ma" (example)
"""

import logging
import random
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)


class Augmenter:
    """
    Data augmenter for Old Assyrian transliterations.

    Adds synthetic gaps to simulate damaged tablets.
    """

    def __init__(
        self,
        gap_prob: float = 0.2,
        big_gap_ratio: float = 0.3,
        preserve_proper_nouns: bool = True,
        preserve_sumerograms: bool = True,
        seed: int = None,
    ):
        """
        Args:
            gap_prob: Probability of adding gaps to a transliteration (0.0-1.0)
            big_gap_ratio: Ratio of gaps that should be <big_gap> vs <gap> (0.0-1.0)
            preserve_proper_nouns: Don't replace proper nouns with gaps
            preserve_sumerograms: Don't replace Sumerograms with gaps
            seed: Random seed for reproducibility
        """
        self.gap_prob = gap_prob
        self.big_gap_ratio = big_gap_ratio
        self.preserve_proper_nouns = preserve_proper_nouns
        self.preserve_sumerograms = preserve_sumerograms

        if seed is not None:
            random.seed(seed)

    # ------------------------------------------------------------------
    # Gap insertion
    # ------------------------------------------------------------------

    def add_gaps(self, transliteration: str) -> str:
        """
        Add synthetic gaps to a transliteration.

        Strategy:
        1. Tokenize by words (space-separated)
        2. Randomly select words to replace with gaps
        3. Preserve proper nouns and Sumerograms if configured
        4. Replace with <gap> or <big_gap> based on ratio

        Args:
            transliteration: Original transliteration string

        Returns:
            Augmented transliteration with synthetic gaps
        """
        if not transliteration or not isinstance(transliteration, str):
            return transliteration

        # Random skip: only augment gap_prob fraction of inputs
        if random.random() > self.gap_prob:
            return transliteration

        # Tokenize
        words = transliteration.split()
        if len(words) < 3:
            # Too short to augment meaningfully
            return transliteration

        # Identify words to preserve
        preserve_indices = set()
        for i, word in enumerate(words):
            if self.preserve_proper_nouns and self._is_proper_noun(word):
                preserve_indices.add(i)
            if self.preserve_sumerograms and self._is_sumerogram(word):
                preserve_indices.add(i)

        # Select words to replace (avoid first/last word for readability)
        replaceable_indices = [
            i for i in range(1, len(words) - 1)
            if i not in preserve_indices
        ]

        if not replaceable_indices:
            # Nothing to replace
            return transliteration

        # Replace 1-3 words with gaps
        num_gaps = random.randint(1, min(3, len(replaceable_indices)))
        gap_indices = random.sample(replaceable_indices, num_gaps)

        # Build augmented text
        result_words = []
        i = 0
        while i < len(words):
            if i in gap_indices:
                # Decide gap type
                if random.random() < self.big_gap_ratio:
                    # Big gap: replace 1-2 consecutive words
                    span = random.randint(1, min(2, len(words) - i))
                    result_words.append("<big_gap>")
                    i += span
                else:
                    # Single gap
                    result_words.append("<gap>")
                    i += 1
            else:
                result_words.append(words[i])
                i += 1

        return " ".join(result_words)

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def augment_batch(
        self,
        transliterations: List[str],
    ) -> List[str]:
        """
        Augment a batch of transliterations.

        Args:
            transliterations: List of transliteration strings

        Returns:
            List of augmented transliterations
        """
        return [self.add_gaps(t) for t in transliterations]

    def augment_corpus(
        self,
        transliterations: List[str],
        augmentation_factor: int = 1,
    ) -> Tuple[List[str], List[int]]:
        """
        Augment a corpus by creating multiple augmented versions.

        Args:
            transliterations: Original transliterations
            augmentation_factor: Number of augmented versions per original

        Returns:
            Tuple of (augmented_transliterations, original_indices)
            - augmented_transliterations: Combined original + augmented versions
            - original_indices: Maps each augmented entry back to original index
        """
        augmented = []
        original_indices = []

        for i, orig in enumerate(transliterations):
            # Keep original
            augmented.append(orig)
            original_indices.append(i)

            # Add augmented versions
            for _ in range(augmentation_factor):
                aug = self.add_gaps(orig)
                augmented.append(aug)
                original_indices.append(i)

        logger.info(
            f"Augmented corpus: {len(transliterations)} → {len(augmented)} "
            f"({augmentation_factor}x augmentation)"
        )
        return augmented, original_indices

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

    def _is_proper_noun(self, word: str) -> bool:
        """
        Check if word is a proper noun (capitalized).

        Old Assyrian convention: proper nouns are capitalized.
        """
        if not word:
            return False
        # Remove common prefixes/suffixes
        clean = word.strip(".,!?;:'\"()[]")
        return clean and clean[0].isupper()

    def _is_sumerogram(self, word: str) -> bool:
        """
        Check if word is a Sumerogram (all caps).

        Sumerograms: ALL CAPS words, optionally with dots (e.g., KÙ.BABBAR)
        """
        if not word or len(word) < 2:
            return False
        # Remove determinatives and brackets
        clean = re.sub(r'[{}\[\]]', '', word)
        clean = clean.strip(".,!?;:'\"()")
        # Check if all alphabetic characters are uppercase
        return clean.isupper() and any(c.isalpha() for c in clean)
