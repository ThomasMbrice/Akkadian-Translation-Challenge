#!/usr/bin/env python3
"""
Create competition submission file from predictions.

Formats predictions according to competition requirements and
validates submission format.

Usage:
    python scripts/submit.py --predictions predictions.csv --output submission.csv
"""

import argparse
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.io import load_csv, save_csv, setup_logging

logger = logging.getLogger(__name__)


def create_submission(
    test_file: str,
    predictions_file: str,
    output_file: str,
) -> None:
    """
    Create competition submission file.

    Args:
        test_file: Original test.csv with IDs
        predictions_file: CSV with predictions column
        output_file: Output submission file path
    """
    logger.info("=" * 60)
    logger.info("CREATING SUBMISSION FILE")
    logger.info("=" * 60)

    # Load test file (has the IDs)
    logger.info(f"Loading test file: {test_file}")
    test_df = load_csv(test_file)

    if "id" not in test_df.columns:
        raise ValueError("Test file must have 'id' column")

    logger.info(f"Test file has {len(test_df)} examples")

    # Load predictions
    logger.info(f"Loading predictions: {predictions_file}")
    pred_df = load_csv(predictions_file)

    if "prediction" not in pred_df.columns:
        raise ValueError("Predictions file must have 'prediction' column")

    logger.info(f"Predictions file has {len(pred_df)} examples")

    # Validate counts match
    if len(test_df) != len(pred_df):
        raise ValueError(
            f"Count mismatch: {len(test_df)} test examples vs "
            f"{len(pred_df)} predictions"
        )

    # Create submission DataFrame
    # Competition expects: id, translation
    submission_df = test_df[["id"]].copy()
    submission_df["translation"] = pred_df["prediction"].values

    # Validate no missing translations
    missing_count = submission_df["translation"].isna().sum()
    if missing_count > 0:
        logger.warning(f"Found {missing_count} missing translations (will use empty strings)")
        submission_df["translation"].fillna("", inplace=True)

    # Save submission
    logger.info(f"Saving submission: {output_file}")
    save_csv(submission_df, output_file)

    # Summary
    logger.info("=" * 60)
    logger.info("SUBMISSION CREATED")
    logger.info("=" * 60)
    logger.info(f"Rows: {len(submission_df)}")
    logger.info(f"Columns: {list(submission_df.columns)}")
    logger.info(f"Output: {output_file}")
    logger.info("=" * 60)

    # Show sample
    logger.info("\nSample rows:")
    print(submission_df.head(3).to_string(index=False))
    logger.info("")


def validate_submission(submission_file: str) -> bool:
    """
    Validate submission file format.

    Args:
        submission_file: Path to submission CSV

    Returns:
        True if valid, False otherwise
    """
    logger.info("Validating submission format...")

    try:
        df = load_csv(submission_file)

        # Check required columns
        required_cols = {"id", "translation"}
        if not required_cols.issubset(df.columns):
            logger.error(f"Missing required columns. Expected: {required_cols}, Got: {set(df.columns)}")
            return False

        # Check for missing values
        missing_ids = df["id"].isna().sum()
        missing_trans = df["translation"].isna().sum()

        if missing_ids > 0:
            logger.error(f"Found {missing_ids} missing IDs")
            return False

        if missing_trans > 0:
            logger.warning(f"Found {missing_trans} missing translations")

        # Check ID format (should be integers)
        if not df["id"].dtype in ["int64", "int32"]:
            logger.warning(f"ID column type is {df['id'].dtype}, expected int")

        logger.info("✓ Submission format is valid")
        logger.info(f"  Rows: {len(df)}")
        logger.info(f"  Columns: {list(df.columns)}")

        return True

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create competition submission file"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="data/raw/deep-past-initiative-machine-translation/test.csv",
        help="Test file with IDs",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="CSV file with predictions",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission.csv",
        help="Output submission file",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate submission format after creation",
    )

    args = parser.parse_args()

    setup_logging(level="INFO")

    try:
        # Create submission
        create_submission(
            test_file=args.test_file,
            predictions_file=args.predictions,
            output_file=args.output,
        )

        # Validate if requested
        if args.validate:
            is_valid = validate_submission(args.output)
            if not is_valid:
                logger.error("Submission validation failed!")
                sys.exit(1)

        logger.info("✓ Submission file ready for upload!")

    except Exception as e:
        logger.error(f"Submission creation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
