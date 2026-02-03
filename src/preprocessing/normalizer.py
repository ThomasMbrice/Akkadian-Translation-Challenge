"""
Text normalization for Old Assyrian transliterations.

Implements competition preprocessing rules:
- Remove certainty/uncertainty markers, glosses, damage markers
- Normalize gaps to <gap> and <big_gap>
- Preserve determinatives, Sumerograms, proper nouns
"""

import re
import logging
from typing import Dict

from src.utils.constants import (
    REMOVE_CHARS,
    REMOVE_PATTERNS,
    SINGLE_GAP_PATTERNS,
    BIG_GAP_PATTERNS,
    SINGLE_GAP_TOKEN,
    BIG_GAP_TOKEN,
    build_remove_chars_pattern,
    build_gap_normalization_patterns,
)

logger = logging.getLogger(__name__)


class Normalizer:
    """Normalize Old Assyrian transliterations per competition spec."""

    def __init__(self):
        """Initialize normalizer with compiled patterns."""
        self.remove_chars_pattern = re.compile(build_remove_chars_pattern())
        gap_patterns = build_gap_normalization_patterns()
        self.single_gap_pattern = re.compile(gap_patterns["single"])
        self.big_gap_pattern = re.compile(gap_patterns["big"])
        self.remove_patterns = [re.compile(p) for p in REMOVE_PATTERNS]

    def normalize(self, text: str) -> str:
        """
        Normalize transliteration text.

        Args:
            text: Raw transliteration string

        Returns:
            Normalized transliteration

        Example:
            >>> normalizer = Normalizer()
            >>> normalizer.normalize("a-na A-šùr! [x x] qí-bi-ma")
            'a-na A-šùr [x x] qí-bi-ma'
        """
        if not text or not isinstance(text, str):
            return ""

        # 1. Remove erasure patterns (including content)
        for pattern in self.remove_patterns:
            text = pattern.sub("", text)

        # 2. Normalize gaps (do this before removing brackets)
        text = self._normalize_gaps(text)

        # 3. Remove prohibited characters
        text = self.remove_chars_pattern.sub("", text)

        # 4. Clean up whitespace
        text = self._clean_whitespace(text)

        return text

    def _normalize_gaps(self, text: str) -> str:
        """
        Normalize gap patterns to <gap> and <big_gap>.

        Args:
            text: Text with gap patterns

        Returns:
            Text with normalized gaps
        """
        # Replace big gaps first (more specific)
        text = self.big_gap_pattern.sub(BIG_GAP_TOKEN, text)

        # Then replace single gaps
        text = self.single_gap_pattern.sub(SINGLE_GAP_TOKEN, text)

        return text

    def _clean_whitespace(self, text: str) -> str:
        """
        Clean up redundant whitespace.

        Args:
            text: Text with potential whitespace issues

        Returns:
            Text with normalized whitespace
        """
        # Replace multiple spaces with single space
        text = re.sub(r"\s+", " ", text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def normalize_batch(self, texts: list[str]) -> list[str]:
        """
        Normalize a batch of texts.

        Args:
            texts: List of transliteration strings

        Returns:
            List of normalized transliterations
        """
        return [self.normalize(text) for text in texts]


def normalize_transliteration(text: str) -> str:
    """
    Convenience function to normalize a single transliteration.

    Args:
        text: Raw transliteration

    Returns:
        Normalized transliteration
    """
    normalizer = Normalizer()
    return normalizer.normalize(text)
