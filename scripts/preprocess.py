#!/usr/bin/env python3
"""
Preprocessing pipeline for Old Assyrian transliterations.

Loads raw competition data, normalizes transliterations,
extracts features, and performs sentence-level alignment.

Usage:
    python scripts/preprocess.py --config configs/preprocessing.yaml
"""

import argparse
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.io import load_csv, save_csv, load_yaml, setup_logging, save_json
from src.preprocessing.normalizer import Normalizer
from src.preprocessing.extractor import FeatureExtractor
from src.preprocessing.aligner import SentenceAligner

logger = logging.getLogger(__name__)


def preprocess_pipeline(config_path: str) -> None:
    """
    Run the complete preprocessing pipeline.

    Args:
        config_path: Path to preprocessing configuration YAML
    """
    # Load configuration
    config = load_yaml(config_path)

    # Setup logging
    setup_logging(
        log_file=config.get("logging", {}).get("file"),
        level=config.get("logging", {}).get("level", "INFO"),
    )

    logger.info("=" * 60)
    logger.info("PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    # Initialize components
    normalizer = Normalizer()
    extractor = FeatureExtractor(
        min_sumerogram_length=config.get("extraction", {})
        .get("sumerograms", {})
        .get("min_length", 2)
    )

    # Load training data
    logger.info("Loading training data...")
    train_df = load_csv(
        config["input"]["train"],
        required_columns=["oare_id", "transliteration", "translation"],
    )

    # Load alignment aid if available
    alignment_aid_df = None
    if config.get("alignment", {}).get("sentence_splitting", {}).get("use_alignment_aid"):
        try:
            alignment_aid_df = load_csv(config["input"]["sentence_alignment_aid"])
            logger.info(f"Loaded alignment aid: {len(alignment_aid_df)} rows")
        except FileNotFoundError:
            logger.warning("Alignment aid file not found, skipping")

    aligner = SentenceAligner(
        alignment_aid_df=alignment_aid_df,
        max_length_ratio=config.get("alignment", {}).get("max_length_ratio", 3.0),
    )

    # Step 1: Normalize transliterations
    logger.info("Step 1: Normalizing transliterations...")
    train_df["transliteration_normalized"] = normalizer.normalize_batch(
        train_df["transliteration"].fillna("").tolist()
    )

    # Step 2: Extract features
    logger.info("Step 2: Extracting features...")
    all_features = []
    for text in train_df["transliteration_normalized"]:
        features = extractor.extract_all(text)
        all_features.append(features)

    # Step 3: Build lexicons
    logger.info("Step 3: Building lexicons...")
    lexicons = extractor.build_lexicons(
        train_df["transliteration_normalized"].tolist()
    )

    # Save lexicons
    save_json(
        lexicons["proper_nouns"],
        config["output"]["proper_nouns"],
    )
    save_json(
        lexicons["sumerograms"],
        config["output"]["sumerograms"],
    )

    logger.info(f"Saved lexicons to {Path(config['output']['proper_nouns']).parent}")

    # Step 4: Sentence alignment
    if config.get("alignment", {}).get("sentence_splitting", {}).get("enabled", True):
        logger.info("Step 4: Sentence-level alignment...")

        # Use normalized transliterations
        train_df_for_alignment = train_df.copy()
        train_df_for_alignment["transliteration"] = train_df_for_alignment[
            "transliteration_normalized"
        ]

        sentences_df = aligner.align_corpus(
            train_df_for_alignment,
            oare_id_col="oare_id",
            transliteration_col="transliteration",
            translation_col="translation",
        )

        # Save sentence-level data
        save_csv(sentences_df, config["output"]["train_sentences"])
        logger.info(
            f"Saved {len(sentences_df)} sentence pairs to "
            f"{config['output']['train_sentences']}"
        )
    else:
        # Save document-level data with normalized transliterations
        output_df = train_df[[
            "oare_id",
            "transliteration_normalized",
            "translation"
        ]].copy()
        output_df.rename(
            columns={"transliteration_normalized": "transliteration"},
            inplace=True
        )
        save_csv(output_df, config["output"]["train_sentences"])
        logger.info(
            f"Saved {len(output_df)} document-level pairs "
            f"(sentence alignment disabled)"
        )

    # Step 5: Quality statistics
    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Input documents: {len(train_df)}")

    if config.get("alignment", {}).get("sentence_splitting", {}).get("enabled", True):
        logger.info(f"Output sentences: {len(sentences_df)}")
        logger.info(f"Avg sentences per doc: {len(sentences_df) / len(train_df):.2f}")
    else:
        logger.info(f"Output documents: {len(output_df)}")

    logger.info(f"Unique proper nouns: {len(lexicons['proper_nouns'])}")
    logger.info(f"Unique Sumerograms: {len(lexicons['sumerograms'])}")
    logger.info(f"Unique determinatives: {len(lexicons['determinatives'])}")
    logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Preprocess Old Assyrian transliterations"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/preprocessing.yaml",
        help="Path to preprocessing configuration file",
    )

    args = parser.parse_args()

    try:
        preprocess_pipeline(args.config)
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
