"""
Sentence alignment for Old Assyrian translations.

Handles conversion from document-level to sentence-level alignment
using the provided alignment aid file.
"""

import logging
from typing import List, Dict, Tuple, Optional
import pandas as pd

from src.utils.constants import MAX_LENGTH_RATIO

logger = logging.getLogger(__name__)


class SentenceAligner:
    """Align document-level translations to sentence-level."""

    def __init__(
        self,
        alignment_aid_df: Optional[pd.DataFrame] = None,
        max_length_ratio: float = MAX_LENGTH_RATIO,
    ):
        """
        Initialize sentence aligner.

        Args:
            alignment_aid_df: DataFrame with sentence alignment information
                Expected columns: text_uuid or oare_id, plus sentence info
            max_length_ratio: Maximum acceptable length ratio for validation
        """
        self.alignment_aid = alignment_aid_df
        self.max_length_ratio = max_length_ratio

        # Determine ID column name if alignment aid is provided
        self.id_column = None
        if alignment_aid_df is not None:
            if "oare_id" in alignment_aid_df.columns:
                self.id_column = "oare_id"
            elif "text_uuid" in alignment_aid_df.columns:
                self.id_column = "text_uuid"
            else:
                logger.warning(
                    "Alignment aid has no 'oare_id' or 'text_uuid' column. "
                    "Disabling alignment aid."
                )
                self.alignment_aid = None

    def align_document(
        self,
        oare_id: str,
        transliteration: str,
        translation: str,
    ) -> List[Dict[str, str]]:
        """
        Split document-level translation into sentence-level pairs.

        Args:
            oare_id: OARE identifier for the document
            transliteration: Full document transliteration
            translation: Full document translation

        Returns:
            List of sentence-level dictionaries with keys:
                - 'oare_id', 'transliteration', 'translation'
        """
        # If no alignment aid, return document as single sentence
        if self.alignment_aid is None:
            return [{
                "oare_id": oare_id,
                "transliteration": transliteration,
                "translation": translation,
            }]

        # Get alignment info for this document
        try:
            doc_alignment = self.alignment_aid[
                self.alignment_aid[self.id_column] == oare_id
            ]
        except (KeyError, TypeError):
            # Column not found or other error
            logger.debug(f"Cannot match {oare_id} in alignment aid")
            return [{
                "oare_id": oare_id,
                "transliteration": transliteration,
                "translation": translation,
            }]

        if doc_alignment.empty:
            # No alignment info, treat as single sentence
            logger.debug(f"No alignment info for {oare_id}, using full document")
            return [{
                "oare_id": oare_id,
                "transliteration": transliteration,
                "translation": translation,
            }]

        # Split based on alignment aid
        sentences = self._split_with_alignment_aid(
            oare_id,
            transliteration,
            translation,
            doc_alignment,
        )

        return sentences

    def _split_with_alignment_aid(
        self,
        oare_id: str,
        transliteration: str,
        translation: str,
        alignment_info: pd.DataFrame,
    ) -> List[Dict[str, str]]:
        """
        Split document using alignment aid information.

        This is a simplified implementation. In practice, you'd need to:
        1. Match first words to locate sentence boundaries
        2. Split transliteration based on line numbers
        3. Heuristically align translation sentences

        For now, we'll do simple sentence splitting.
        """
        # Simple fallback: split by newlines and periods
        akk_sentences = self._split_transliteration(transliteration)
        eng_sentences = self._split_translation(translation)

        # Try to align them (simple heuristic: equal split)
        if len(akk_sentences) != len(eng_sentences):
            logger.warning(
                f"Sentence count mismatch for {oare_id}: "
                f"{len(akk_sentences)} Akkadian vs {len(eng_sentences)} English"
            )
            # Fall back to document-level
            return [{
                "oare_id": oare_id,
                "transliteration": transliteration,
                "translation": translation,
            }]

        # Create aligned pairs
        sentences = []
        for akk, eng in zip(akk_sentences, eng_sentences):
            if self._validate_pair(akk, eng):
                sentences.append({
                    "oare_id": oare_id,
                    "transliteration": akk,
                    "translation": eng,
                })

        return sentences if sentences else [{
            "oare_id": oare_id,
            "transliteration": transliteration,
            "translation": translation,
        }]

    def _split_transliteration(self, text: str) -> List[str]:
        """
        Split transliteration into sentences.

        Uses newlines and line breaks as sentence boundaries.
        """
        # Split by newlines
        lines = text.split("\n")

        # Clean and filter
        sentences = []
        for line in lines:
            line = line.strip()
            if line:
                sentences.append(line)

        return sentences

    def _split_translation(self, text: str) -> List[str]:
        """
        Split English translation into sentences.

        Uses periods and newlines as sentence boundaries.
        """
        # First split by newlines
        lines = text.split("\n")

        sentences = []
        for line in lines:
            # Then split by periods (but preserve the period)
            parts = line.split(".")
            for part in parts:
                part = part.strip()
                if part:
                    sentences.append(part)

        return sentences

    def _validate_pair(self, akkadian: str, english: str) -> bool:
        """
        Validate sentence pair quality.

        Args:
            akkadian: Akkadian sentence
            english: English sentence

        Returns:
            True if pair passes validation
        """
        # Check minimum lengths
        if len(akkadian) < 3 or len(english) < 3:
            return False

        # Check length ratio (Akkadian is typically longer due to hyphens)
        ratio = len(akkadian) / max(len(english), 1)
        if ratio > self.max_length_ratio:
            logger.debug(
                f"Length ratio too high: {ratio:.2f} "
                f"(Akk: {len(akkadian)}, Eng: {len(english)})"
            )
            return False

        return True

    def align_corpus(
        self,
        df: pd.DataFrame,
        oare_id_col: str = "oare_id",
        transliteration_col: str = "transliteration",
        translation_col: str = "translation",
    ) -> pd.DataFrame:
        """
        Align an entire corpus from document-level to sentence-level.

        Args:
            df: DataFrame with document-level data
            oare_id_col: Column name for OARE ID
            transliteration_col: Column name for transliteration
            translation_col: Column name for translation

        Returns:
            DataFrame with sentence-level alignments
        """
        all_sentences = []

        for _, row in df.iterrows():
            sentences = self.align_document(
                oare_id=row[oare_id_col],
                transliteration=row[transliteration_col],
                translation=row[translation_col],
            )
            all_sentences.extend(sentences)

        logger.info(
            f"Aligned {len(df)} documents to {len(all_sentences)} sentences"
        )

        return pd.DataFrame(all_sentences)
