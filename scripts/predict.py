#!/usr/bin/env python3
"""
Generate submission.csv for the Kaggle competition.

Loads the fine-tuned ByT5 checkpoint, runs inference on test.csv,
and writes a submission.csv with the required id,translation format.

Usage:
    # Default: uses configs/training.yaml paths, outputs submission.csv
    python scripts/predict.py

    # Override checkpoint or output path
    python scripts/predict.py --checkpoint models/byt5_finetuned/final
    python scripts/predict.py --output outputs/submission.csv
    python scripts/predict.py --no-rag
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import setup_logging, load_yaml
from src.modeling import ByT5Trainer, ContextAssembler
from src.retrieval import Retriever
from src.preprocessing.pretranslator import PreTranslator

logger = logging.getLogger(__name__)

BATCH_SIZE = 16


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Kaggle submission CSV")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to fine-tuned model (default: from config output_dir/final)")
    parser.add_argument("--test-csv", default=None,
                        help="Path to test.csv (default: from config)")
    parser.add_argument("--output", default="submission.csv",
                        help="Output path for submission CSV")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG context")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max generation length (default: 512)")
    parser.add_argument("--num-beams", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(level="INFO")

    config = load_yaml(str(PROJECT_ROOT / args.config))

    # Test data path
    if args.test_csv:
        test_path = Path(args.test_csv)
    else:
        test_path = PROJECT_ROOT / "data/raw/deep-past-initiative-machine-translation/test.csv"

    logger.info(f"Loading test data: {test_path}")
    test_df = pd.read_csv(test_path)
    logger.info(f"Test examples: {len(test_df)}")

    if "transliteration" not in test_df.columns:
        raise ValueError(f"test.csv missing 'transliteration' column. Got: {list(test_df.columns)}")
    if "id" not in test_df.columns:
        raise ValueError(f"test.csv missing 'id' column. Got: {list(test_df.columns)}")

    # Checkpoint path
    if args.checkpoint:
        checkpoint = args.checkpoint
    else:
        checkpoint = str(PROJECT_ROOT / config["output"]["output_dir"] / "final")
    logger.info(f"Checkpoint: {checkpoint}")

    # Retrieval setup
    retriever = None
    lexicon = None
    assembler = None

    if not args.no_rag:
        ret_cfg = config.get("retrieval", {})
        if ret_cfg.get("enabled", False):
            corpus_path = str(PROJECT_ROOT / ret_cfg["corpus_path"])
            index_path = str(PROJECT_ROOT / ret_cfg["index_path"])
            logger.info("Loading retrieval system…")
            retriever = Retriever()
            retriever.load(corpus_path=corpus_path, index_path=index_path)
            lexicon = retriever.lexicon
            pt = PreTranslator(lexicon=lexicon) if lexicon is not None else None
            assembler = ContextAssembler(
                retriever=retriever,
                lexicon=lexicon,
                pretranslator=pt,
                max_length=ret_cfg.get("max_context_length", 800),
                num_examples=ret_cfg.get("k_examples", 3),
                include_lexicon=True,
                include_examples=True,
            )
            logger.info("RAG enabled")
        else:
            logger.info("RAG disabled in config")
    else:
        logger.info("RAG disabled via --no-rag flag")

    # Load model
    model_cfg = config["model"]
    trainer = ByT5Trainer(
        model_name=checkpoint,
        output_dir=str(PROJECT_ROOT / config["output"]["output_dir"]),
        use_rag=assembler is not None,
        context_assembler=assembler,
        max_source_length=model_cfg["max_input_length"],
        max_target_length=args.max_length,
    )

    # Run inference in batches
    sources = test_df["transliteration"].fillna("").tolist()
    translations = []

    logger.info(f"Running inference ({len(sources)} examples, batch_size={BATCH_SIZE})…")
    for start in range(0, len(sources), BATCH_SIZE):
        batch = sources[start: start + BATCH_SIZE]
        translations.extend(
            trainer.translate(batch, num_beams=args.num_beams, max_length=args.max_length)
        )
        logger.info(f"  {min(start + BATCH_SIZE, len(sources))}/{len(sources)}")

    # Write submission
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    submission = pd.DataFrame({"id": test_df["id"], "translation": translations})
    submission.to_csv(output_path, index=False)
    logger.info(f"Saved {len(submission)} predictions to {output_path}")

    # Quick sanity check
    logger.info("\nSample predictions:")
    for _, row in submission.head(3).iterrows():
        logger.info(f"  [{row['id']}] {row['translation'][:120]}")


if __name__ == "__main__":
    main()
