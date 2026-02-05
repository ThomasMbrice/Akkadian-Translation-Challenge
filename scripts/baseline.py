#!/usr/bin/env python3
"""
Zero-shot baseline evaluation using ByT5.

Tests pre-trained ByT5 model (without fine-tuning) on Old Assyrian translation
to establish baseline BLEU and chrF++ scores.

Usage:
    python scripts/baseline.py --model google/byt5-small --output outputs/baseline_results.json
"""

import argparse
import logging
import os
from pathlib import Path
import sys
import time
from typing import List
import json

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# Increase HuggingFace Hub download timeout to 300s (default is 10s)
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.io import load_csv, save_csv, setup_logging, save_json, ensure_dir
from src.preprocessing.normalizer import Normalizer
from src.evaluation.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


def _load_with_retry(load_fn, max_retries: int = 3, backoff: float = 5.0):
    """
    Retry a HuggingFace download call with exponential backoff.

    Args:
        load_fn: Callable that performs the download (e.g. AutoTokenizer.from_pretrained)
        max_retries: Number of retry attempts
        backoff: Base delay in seconds between retries (doubles each attempt)

    Returns:
        Result of load_fn()
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return load_fn()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = backoff * (2 ** attempt)
                logger.warning(
                    f"Download attempt {attempt + 1}/{max_retries} failed: {e}. "
                    f"Retrying in {wait:.0f}s..."
                )
                time.sleep(wait)
            else:
                logger.error(
                    f"Download failed after {max_retries} attempts."
                )
    raise last_error


class ByT5Translator:
    """Zero-shot translator using ByT5."""

    def __init__(
        self,
        model_name: str = "google/byt5-small",
        device: str = None,
        max_length: int = 512,
        cache_dir: str = None,
    ):
        """
        Initialize ByT5 translator.

        Args:
            model_name: Hugging Face model name or local path
            device: Device to use (cuda/cpu), auto-detect if None
            max_length: Maximum sequence length
            cache_dir: Local directory to cache downloaded models (None = HF default)
        """
        self.model_name = model_name
        self.max_length = max_length

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading model: {model_name}")
        logger.info(f"Device: {self.device}")
        if cache_dir:
            logger.info(f"Cache dir: {cache_dir}")

        # Build common kwargs for from_pretrained
        pretrained_kwargs = {}
        if cache_dir:
            pretrained_kwargs["cache_dir"] = cache_dir

        # Load tokenizer and model with retry logic
        self.tokenizer = _load_with_retry(
            lambda: AutoTokenizer.from_pretrained(model_name, **pretrained_kwargs)
        )
        self.model = _load_with_retry(
            lambda: AutoModelForSeq2SeqLM.from_pretrained(model_name, **pretrained_kwargs)
        )
        self.model.to(self.device)
        self.model.eval()

        logger.info("Model loaded successfully")

    def translate(
        self,
        texts: List[str],
        batch_size: int = 8,
        num_beams: int = 4,
    ) -> List[str]:
        """
        Translate Akkadian texts to English.

        Args:
            texts: List of Akkadian transliterations
            batch_size: Batch size for inference
            num_beams: Number of beams for beam search

        Returns:
            List of English translations
        """
        translations = []

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
                batch = texts[i : i + batch_size]

                # Add task prefix for T5-style models
                batch_with_prefix = [
                    f"translate Akkadian to English: {text}" for text in batch
                ]

                # Tokenize
                inputs = self.tokenizer(
                    batch_with_prefix,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Generate
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                )

                # Decode
                batch_translations = self.tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )
                translations.extend(batch_translations)

        return translations


def run_baseline(
    model_name: str,
    test_file: str,
    output_dir: str,
    batch_size: int = 8,
    num_beams: int = 4,
    sample_size: int = None,
    cache_dir: str = None,
) -> None:
    """
    Run zero-shot baseline evaluation.

    Args:
        model_name: Hugging Face model name
        test_file: Path to test/validation CSV
        output_dir: Output directory for results
        batch_size: Batch size for inference
        num_beams: Number of beams for beam search
        sample_size: Limit evaluation to N samples (None for all)
        cache_dir: Local directory to cache downloaded models
    """
    # Setup
    setup_logging(level="INFO")
    ensure_dir(output_dir)

    logger.info("=" * 60)
    logger.info("ZERO-SHOT BASELINE EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name}")

    # Load data
    logger.info(f"Loading data from {test_file}...")
    df = load_csv(test_file)

    # Check if we have translations (for evaluation)
    has_translations = "translation" in df.columns

    if not has_translations:
        logger.warning("No translations found in data - will generate predictions only")

    # Sample if requested
    if sample_size and sample_size < len(df):
        logger.info(f"Sampling {sample_size} examples for evaluation")
        df = df.sample(n=sample_size, random_state=42)

    # Normalize transliterations
    logger.info("Normalizing transliterations...")
    normalizer = Normalizer()
    df["transliteration_normalized"] = normalizer.normalize_batch(
        df["transliteration"].fillna("").tolist()
    )

    # Initialize translator
    translator = ByT5Translator(
        model_name=model_name,
        max_length=512,
        cache_dir=cache_dir,
    )

    # Translate
    logger.info(f"Translating {len(df)} examples...")
    predictions = translator.translate(
        df["transliteration_normalized"].tolist(),
        batch_size=batch_size,
        num_beams=num_beams,
    )

    # Save predictions
    df["prediction"] = predictions
    predictions_file = Path(output_dir) / "predictions.csv"
    save_csv(df, predictions_file)
    logger.info(f"Saved predictions to {predictions_file}")

    # Evaluate if we have references
    if has_translations:
        logger.info("Calculating metrics...")
        calculator = MetricsCalculator()

        references = df["translation"].tolist()
        sources = df["transliteration_normalized"].tolist()

        metrics = calculator.calculate_all_metrics(
            hypotheses=predictions,
            references=references,
            source_texts=sources,
        )

        # Print metrics
        calculator.print_metrics(metrics)

        # Save metrics
        metrics_file = Path(output_dir) / "baseline_results.json"
        save_json(metrics, metrics_file)
        logger.info(f"Saved metrics to {metrics_file}")

        # Log summary
        logger.info("=" * 60)
        logger.info("BASELINE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Model: {model_name}")
        logger.info(f"Examples: {len(df)}")
        logger.info(f"BLEU: {metrics.get('bleu', 0):.2f}")
        logger.info(f"chrF++: {metrics.get('chrf', 0):.2f}")
        logger.info(f">>> COMPETITION SCORE (Geometric Mean): {metrics.get('geometric_mean', 0):.2f} <<<")
        if "proper_noun_accuracy" in metrics:
            logger.info(
                f"Proper Noun Acc: {metrics['proper_noun_accuracy']:.2%}"
            )
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("BASELINE COMPLETE (no evaluation)")
        logger.info("=" * 60)
        logger.info(f"Generated {len(predictions)} predictions")
        logger.info(f"Saved to {predictions_file}")
        logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run zero-shot baseline evaluation")
    parser.add_argument(
        "--model",
        type=str,
        default="google/byt5-small",
        help="Hugging Face model name (default: google/byt5-small)",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="data/processed/train_sentences.csv",
        help="Test/validation file (use processed data with translations)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/baseline",
        help="Output directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="Number of beams for beam search",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample size (default: use all data)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Local directory to cache downloaded models (avoids re-downloading)",
    )

    args = parser.parse_args()

    try:
        run_baseline(
            model_name=args.model,
            test_file=args.test_file,
            output_dir=args.output,
            batch_size=args.batch_size,
            num_beams=args.num_beams,
            sample_size=args.sample,
            cache_dir=args.cache_dir,
        )
    except Exception as e:
        logger.error(f"Baseline evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
