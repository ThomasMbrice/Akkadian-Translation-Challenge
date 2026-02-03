"""
Evaluation metrics for Akkadian NMT.

Implements:
- BLEU score
- chrF++ score
- Proper noun accuracy
"""

import logging
from typing import List, Dict, Any, Optional
import re

import sacrebleu
from collections import Counter

from src.preprocessing.extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate evaluation metrics for translation quality."""

    def __init__(self):
        """Initialize metrics calculator."""
        self.feature_extractor = FeatureExtractor()

    def calculate_bleu(
        self,
        hypotheses: List[str],
        references: List[List[str]],
    ) -> Dict[str, float]:
        """
        Calculate BLEU score.

        Args:
            hypotheses: List of predicted translations
            references: List of reference translations (each can have multiple refs)

        Returns:
            Dictionary with 'bleu' score and additional info
        """
        # Handle single reference per hypothesis
        if references and not isinstance(references[0], list):
            references = [[ref] for ref in references]

        # Transpose references for sacrebleu format
        refs_transposed = [
            [references[i][j] for i in range(len(references))]
            for j in range(len(references[0]))
        ]

        bleu = sacrebleu.corpus_bleu(hypotheses, refs_transposed)

        return {
            "bleu": bleu.score,
            "bleu_1": bleu.precisions[0],
            "bleu_2": bleu.precisions[1],
            "bleu_3": bleu.precisions[2],
            "bleu_4": bleu.precisions[3],
            "bp": bleu.bp,  # Brevity penalty
        }

    def calculate_chrf(
        self,
        hypotheses: List[str],
        references: List[List[str]],
    ) -> Dict[str, float]:
        """
        Calculate chrF++ score.

        chrF++ is more suitable for morphologically rich languages.

        Args:
            hypotheses: List of predicted translations
            references: List of reference translations

        Returns:
            Dictionary with 'chrf' score
        """
        # Handle single reference per hypothesis
        if references and not isinstance(references[0], list):
            references = [[ref] for ref in references]

        # Transpose references
        refs_transposed = [
            [references[i][j] for i in range(len(references))]
            for j in range(len(references[0]))
        ]

        chrf = sacrebleu.corpus_chrf(hypotheses, refs_transposed)

        return {
            "chrf": chrf.score,
        }

    def calculate_proper_noun_accuracy(
        self,
        hypotheses: List[str],
        references: List[str],
        source_texts: List[str],
    ) -> Dict[str, float]:
        """
        Calculate proper noun accuracy.

        Measures how well proper nouns from source are preserved in translations.

        Args:
            hypotheses: List of predicted translations
            references: List of reference translations
            source_texts: List of source Akkadian texts

        Returns:
            Dictionary with proper noun metrics
        """
        total_nouns = 0
        correct_in_hyp = 0
        correct_in_ref = 0

        for source, hyp, ref in zip(source_texts, hypotheses, references):
            # Extract proper nouns from source
            nouns = self.feature_extractor.extract_proper_nouns(source)

            if not nouns:
                continue

            total_nouns += len(nouns)

            # Check how many appear in hypothesis and reference
            for noun in nouns:
                # Fuzzy matching: normalize and check
                noun_normalized = self._normalize_for_matching(noun)

                if noun_normalized in self._normalize_for_matching(hyp):
                    correct_in_hyp += 1

                if noun_normalized in self._normalize_for_matching(ref):
                    correct_in_ref += 1

        accuracy_hyp = correct_in_hyp / total_nouns if total_nouns > 0 else 0.0
        accuracy_ref = correct_in_ref / total_nouns if total_nouns > 0 else 0.0

        return {
            "proper_noun_accuracy": accuracy_hyp,
            "proper_noun_recall_ref": accuracy_ref,
            "total_proper_nouns": total_nouns,
            "correct_in_hypothesis": correct_in_hyp,
            "correct_in_reference": correct_in_ref,
        }

    def _normalize_for_matching(self, text: str) -> str:
        """
        Normalize text for matching (case-insensitive, no diacritics approximation).

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        # Convert to lowercase
        text = text.lower()

        # Simple diacritic removal (approximate)
        replacements = {
            "ā": "a", "ē": "e", "ī": "i", "ū": "u",
            "š": "sh", "ṣ": "s", "ṭ": "t", "ḫ": "h",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        # Remove hyphens
        text = text.replace("-", "")

        return text

    def calculate_all_metrics(
        self,
        hypotheses: List[str],
        references: List[str],
        source_texts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate all evaluation metrics.

        Args:
            hypotheses: Predicted translations
            references: Reference translations
            source_texts: Optional source texts for proper noun accuracy

        Returns:
            Dictionary with all metrics
        """
        # Ensure references are in list format for sacrebleu
        refs_list = [[ref] for ref in references]

        metrics = {}

        # BLEU
        bleu_metrics = self.calculate_bleu(hypotheses, refs_list)
        metrics.update(bleu_metrics)

        # chrF++
        chrf_metrics = self.calculate_chrf(hypotheses, refs_list)
        metrics.update(chrf_metrics)

        # Proper noun accuracy (if source texts provided)
        if source_texts:
            pn_metrics = self.calculate_proper_noun_accuracy(
                hypotheses, references, source_texts
            )
            metrics.update(pn_metrics)

        return metrics

    def print_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Print metrics in a readable format.

        Args:
            metrics: Dictionary of metrics
        """
        logger.info("=" * 60)
        logger.info("EVALUATION METRICS")
        logger.info("=" * 60)

        if "bleu" in metrics:
            logger.info(f"BLEU:   {metrics['bleu']:.2f}")

        if "chrf" in metrics:
            logger.info(f"chrF++: {metrics['chrf']:.2f}")

        if "proper_noun_accuracy" in metrics:
            logger.info(
                f"Proper Noun Accuracy: {metrics['proper_noun_accuracy']:.2%} "
                f"({metrics['correct_in_hypothesis']}/{metrics['total_proper_nouns']})"
            )

        logger.info("=" * 60)


def calculate_metrics(
    hypotheses: List[str],
    references: List[str],
    source_texts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to calculate all metrics.

    Args:
        hypotheses: Predicted translations
        references: Reference translations
        source_texts: Optional source texts

    Returns:
        Dictionary with all metrics
    """
    calculator = MetricsCalculator()
    return calculator.calculate_all_metrics(hypotheses, references, source_texts)
