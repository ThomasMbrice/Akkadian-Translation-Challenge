#!/usr/bin/env python3
"""
Extract transliteration-translation pairs from scholarly publications.

Phase 1 critical-path script.  Mines ~950 academic papers (OCR'd text in
publications.csv) to extract parallel Akkadian–English sentence pairs,
multiplying the training corpus 2-3x over the competition-provided train.csv.

Pipeline:
    1. Load publications.csv, filter to has_akkadian pages, deduplicate rows.
    2. Per-page: clean OCR artifacts → extract pairs via transition detection.
    3. Deduplicate extracted pairs against existing training data and internally.
    4. Normalise transliterations using the same rules as preprocessing.
    5. Save extracted pairs and a combined corpus (existing + new).

Usage:
    # Quick smoke test on 10 PDFs (with OA filter):
    python scripts/extract_publications.py --sample 10

    # Full extraction (default: uses data/oa_pdf_filter.txt):
    python scripts/extract_publications.py

    # Full extraction with custom output path:
    python scripts/extract_publications.py --output data/processed/my_extracted.csv

    # Extract from ALL PDFs (disable filter):
    python scripts/extract_publications.py --pdf-filter=none

    # Use custom PDF filter:
    python scripts/extract_publications.py --pdf-filter data/my_pdfs.txt
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.io import load_csv, save_csv, setup_logging, ensure_dir
from src.data_extraction.ocr_corrector import OCRCorrector
from src.data_extraction.publication_parser import PublicationParser
from src.data_extraction.deduplicator import Deduplicator
from src.preprocessing.normalizer import Normalizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------

PUBLICATIONS_CSV = (
    "data/raw/deep-past-initiative-machine-translation/publications.csv"
)
TRAIN_SENTENCES_CSV = "data/processed/train_sentences.csv"
DEFAULT_OUTPUT = "data/processed/extracted_pairs.csv"
COMBINED_OUTPUT = "data/processed/combined_corpus.csv"
DEFAULT_PDF_FILTER = "data/oa_pdf_filter.txt"


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------


def extract_from_publications(
    sample_size: int = None,
    output_path: str = DEFAULT_OUTPUT,
    pdf_filter_path: str = None,
) -> pd.DataFrame:
    """
    Run the full extraction pipeline.

    Args:
        sample_size: If set, randomly sample this many PDFs for a quick test
            run.  None processes all PDFs.
        output_path: Destination CSV for the extracted pairs.
        pdf_filter_path: Path to text file containing PDF names to process
            (one per line). If None, processes all PDFs. Lines starting with
            '#' are treated as comments.

    Returns:
        DataFrame of deduplicated, normalised extracted pairs.
    """
    setup_logging(level="INFO")

    logger.info("=" * 60)
    logger.info("PHASE 1: PUBLICATION EXTRACTION")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load & filter
    # ------------------------------------------------------------------
    logger.info(f"Loading publications from {PUBLICATIONS_CSV}...")
    pubs_df = pd.read_csv(PUBLICATIONS_CSV)
    total_pages = len(pubs_df)
    total_pdfs = pubs_df["pdf_name"].nunique()
    logger.info(f"Loaded {total_pages} pages from {total_pdfs} PDFs")

    # Keep only pages flagged as containing Akkadian
    pubs_df = pubs_df[pubs_df["has_akkadian"] == True].copy()
    logger.info(f"Filtered to {len(pubs_df)} pages with has_akkadian=True")

    # Drop duplicate rows (same PDF + page appearing multiple times)
    before_dedup = len(pubs_df)
    pubs_df = pubs_df.drop_duplicates(subset=["pdf_name", "page"])
    pubs_df = pubs_df.reset_index(drop=True)
    if len(pubs_df) < before_dedup:
        logger.info(
            f"Removed {before_dedup - len(pubs_df)} duplicate page rows "
            f"({len(pubs_df)} unique pages remain)"
        )

    # Optional: filter to specific PDFs from filter file
    if pdf_filter_path:
        logger.info(f"Loading PDF filter from {pdf_filter_path}...")
        allowed_pdfs = set()
        with open(pdf_filter_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    allowed_pdfs.add(line)

        before_filter = len(pubs_df)
        before_filter_pdfs = pubs_df["pdf_name"].nunique()
        pubs_df = pubs_df[pubs_df["pdf_name"].isin(allowed_pdfs)]
        pubs_df = pubs_df.reset_index(drop=True)

        logger.info(
            f"Filtered to {len(allowed_pdfs)} allowed PDFs: "
            f"{before_filter} → {len(pubs_df)} pages "
            f"({before_filter_pdfs} → {pubs_df['pdf_name'].nunique()} PDFs)"
        )

    # Optional: sample a subset of PDFs for testing
    unique_pdfs = pubs_df["pdf_name"].unique()
    if sample_size is not None and sample_size < len(unique_pdfs):
        sampled_pdfs = (
            pd.Series(unique_pdfs)
            .sample(n=sample_size, random_state=42)
            .tolist()
        )
        pubs_df = pubs_df[pubs_df["pdf_name"].isin(sampled_pdfs)]
        logger.info(
            f"Sampled {sample_size} PDFs ({len(pubs_df)} pages) for testing"
        )

    # ------------------------------------------------------------------
    # 2. Extract pairs
    # ------------------------------------------------------------------
    corrector = OCRCorrector()
    parser = PublicationParser()

    logger.info("Extracting transliteration-translation pairs...")
    all_pairs: List[Dict[str, str]] = []

    for pdf_name in tqdm(
        pubs_df["pdf_name"].unique(), desc="Processing PDFs"
    ):
        pdf_pages = (
            pubs_df[pubs_df["pdf_name"] == pdf_name]
            .sort_values("page")
            .copy()
        )

        # Clean OCR artifacts page by page
        pdf_pages["page_text"] = pdf_pages["page_text"].apply(
            lambda t: corrector.correct(t) if pd.notna(t) else ""
        )

        # Extract pairs from this publication
        pairs = parser.parse_publication(pdf_pages)
        all_pairs.extend(pairs)

    logger.info(
        f"Extracted {len(all_pairs)} raw pairs from "
        f"{pubs_df['pdf_name'].nunique()} PDFs"
    )

    if not all_pairs:
        logger.warning(
            "No pairs extracted. Check input data or parser thresholds."
        )
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # 3. Convert to DataFrame & deduplicate
    # ------------------------------------------------------------------
    extracted_df = pd.DataFrame(all_pairs)
    logger.info(f"Extracted columns: {list(extracted_df.columns)}")

    # Load existing training data for dedup
    logger.info("Loading existing training data for deduplication...")
    existing_df = load_csv(TRAIN_SENTENCES_CSV)

    deduplicator = Deduplicator(similarity_threshold=0.7)
    deduplicator.load_existing(existing_df)
    extracted_df = deduplicator.deduplicate(extracted_df)

    if extracted_df.empty:
        logger.warning("All extracted pairs were duplicates of existing data.")
        return extracted_df

    # ------------------------------------------------------------------
    # 4. Normalise transliterations
    # ------------------------------------------------------------------
    logger.info("Normalising transliterations...")
    normalizer = Normalizer()
    extracted_df["transliteration"] = normalizer.normalize_batch(
        extracted_df["transliteration"].tolist()
    )

    # ------------------------------------------------------------------
    # 5. Save extracted pairs
    # ------------------------------------------------------------------
    ensure_dir(str(Path(output_path).parent))
    save_csv(extracted_df, output_path)
    logger.info(f"Saved {len(extracted_df)} extracted pairs to {output_path}")

    # ------------------------------------------------------------------
    # 6. Create combined corpus (existing + extracted)
    # ------------------------------------------------------------------
    logger.info("Creating combined corpus...")
    common_cols = ["transliteration", "translation"]
    existing_subset = existing_df[common_cols].copy()
    extracted_subset = extracted_df[common_cols].copy()

    combined = pd.concat(
        [existing_subset, extracted_subset], ignore_index=True
    )
    save_csv(combined, COMBINED_OUTPUT)
    logger.info(
        f"Combined corpus: {len(combined)} pairs "
        f"({len(existing_subset)} existing + {len(extracted_subset)} new)"
    )

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("EXTRACTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"PDFs processed:              {pubs_df['pdf_name'].nunique()}")
    logger.info(f"Pages scanned:               {len(pubs_df)}")
    logger.info(f"Pairs extracted (after dedup): {len(extracted_df)}")
    logger.info(f"Combined corpus size:        {len(combined)}")
    logger.info(
        f"Multiplier:                  {len(combined) / max(len(existing_subset), 1):.2f}x"
    )
    logger.info("=" * 60)

    return extracted_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract transliteration-translation pairs from scholarly "
            "publications (Phase 1 critical path)"
        )
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Limit to N randomly-sampled PDFs for testing (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--pdf-filter",
        type=str,
        default=DEFAULT_PDF_FILTER,
        help=(
            f"Path to PDF filter file (default: {DEFAULT_PDF_FILTER}). "
            "Use --pdf-filter=none to disable filtering."
        ),
    )

    args = parser.parse_args()

    # Handle --pdf-filter=none
    pdf_filter = args.pdf_filter if args.pdf_filter != "none" else None

    try:
        extract_from_publications(
            sample_size=args.sample,
            output_path=args.output,
            pdf_filter_path=pdf_filter,
        )
    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
