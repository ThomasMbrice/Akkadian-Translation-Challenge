"""
Feature extraction for Old Assyrian transliterations.

Extracts:
- Proper nouns (capitalized words)
- Sumerograms (all-caps logograms)
- Determinatives (content in {braces})
"""

import re
import logging
from typing import Dict, List, Set
from collections import Counter

from src.utils.constants import (
    PROPER_NOUN_PATTERN,
    SUMEROGRAM_PATTERN,
    DETERMINATIVE_PATTERN,
    CAPITAL_LETTERS,
)

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract linguistic features from Old Assyrian text."""

    def __init__(self, min_sumerogram_length: int = 2):
        """
        Initialize feature extractor.

        Args:
            min_sumerogram_length: Minimum length for Sumerogram extraction
        """
        self.min_sumerogram_length = min_sumerogram_length
        self.proper_noun_pattern = re.compile(PROPER_NOUN_PATTERN)
        self.sumerogram_pattern = re.compile(SUMEROGRAM_PATTERN)
        self.determinative_pattern = re.compile(DETERMINATIVE_PATTERN)

    def extract_proper_nouns(self, text: str) -> List[str]:
        """
        Extract proper nouns (capitalized words).

        Args:
            text: Transliteration text

        Returns:
            List of proper nouns found

        Example:
            >>> extractor = FeatureExtractor()
            >>> extractor.extract_proper_nouns("a-na A-šùr-i-mì-tí")
            ['A-šùr-i-mì-tí']
        """
        if not text:
            return []

        # Find all capitalized words
        matches = self.proper_noun_pattern.findall(text)

        # Filter out Sumerograms (all caps)
        proper_nouns = []
        for match in matches:
            # Check if it's not all uppercase (which would be a Sumerogram)
            if not match.isupper():
                proper_nouns.append(match)

        return proper_nouns

    def extract_sumerograms(self, text: str) -> List[str]:
        """
        Extract Sumerograms (all-caps logograms).

        Args:
            text: Transliteration text

        Returns:
            List of Sumerograms found

        Example:
            >>> extractor = FeatureExtractor()
            >>> extractor.extract_sumerograms("DUMU Ṣí-lí-{d}UTU")
            ['DUMU', 'UTU']
        """
        if not text:
            return []

        # First, remove determinatives to avoid extracting content inside them
        text_no_det = self.determinative_pattern.sub("", text)

        # Find all-caps words
        matches = self.sumerogram_pattern.findall(text_no_det)

        # Filter by minimum length
        sumerograms = [m for m in matches if len(m) >= self.min_sumerogram_length]

        return sumerograms

    def extract_determinatives(self, text: str) -> List[str]:
        """
        Extract determinatives (content in {braces}).

        Args:
            text: Transliteration text

        Returns:
            List of determinatives found (without braces)

        Example:
            >>> extractor = FeatureExtractor()
            >>> extractor.extract_determinatives("Ṣí-lí-{d}UTU")
            ['d']
        """
        if not text:
            return []

        # Find all {content}
        matches = self.determinative_pattern.findall(text)

        # Remove braces
        determinatives = [m.strip("{}") for m in matches]

        return determinatives

    def extract_all(self, text: str) -> Dict[str, List[str]]:
        """
        Extract all features from text.

        Args:
            text: Transliteration text

        Returns:
            Dictionary with keys: 'proper_nouns', 'sumerograms', 'determinatives'
        """
        return {
            "proper_nouns": self.extract_proper_nouns(text),
            "sumerograms": self.extract_sumerograms(text),
            "determinatives": self.extract_determinatives(text),
        }

    def build_lexicons(
        self, texts: List[str]
    ) -> Dict[str, Dict[str, int]]:
        """
        Build lexicons from a corpus of texts.

        Args:
            texts: List of transliteration texts

        Returns:
            Dictionary with counters for each feature type
        """
        proper_noun_counter = Counter()
        sumerogram_counter = Counter()
        determinative_counter = Counter()

        for text in texts:
            features = self.extract_all(text)
            proper_noun_counter.update(features["proper_nouns"])
            sumerogram_counter.update(features["sumerograms"])
            determinative_counter.update(features["determinatives"])

        logger.info(f"Built lexicons from {len(texts)} texts")
        logger.info(f"  Proper nouns: {len(proper_noun_counter)} unique")
        logger.info(f"  Sumerograms: {len(sumerogram_counter)} unique")
        logger.info(f"  Determinatives: {len(determinative_counter)} unique")

        return {
            "proper_nouns": dict(proper_noun_counter),
            "sumerograms": dict(sumerogram_counter),
            "determinatives": dict(determinative_counter),
        }


def extract_features(text: str) -> Dict[str, List[str]]:
    """
    Convenience function to extract all features from a single text.

    Args:
        text: Transliteration text

    Returns:
        Dictionary with extracted features
    """
    extractor = FeatureExtractor()
    return extractor.extract_all(text)
