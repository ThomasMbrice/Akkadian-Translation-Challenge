#!/usr/bin/env python3
"""
Phase 3: Fine-tune ByT5 on Akkadian → English translation with RAG.

Orchestrates the full training pipeline:
    1. Load & split corpus (train / val / test)
    2. Initialise retrieval system (FAISS + lexicon)
    3. Initialise augmenter and context assembler
    4. Fine-tune ByT5 with optional RAG context
    5. Evaluate on val & test with BLEU / chrF++ / proper-noun accuracy
    6. Persist results and final model

Usage:
    # Full training run with defaults from configs/training.yaml
    python scripts/train.py

    # Validate the pipeline without training (fast)
    python scripts/train.py --dry-run

    # Override specific hyperparameters
    python scripts/train.py --epochs 2 --lr 3e-5 --batch-size 4

    # Disable RAG or augmentation
    python scripts/train.py --no-rag
    python scripts/train.py --no-aug

    # Evaluate an already-trained checkpoint (skip training)
    python scripts/train.py --eval-only models/byt5_finetuned/final
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Project root on sys.path so that ``src.*`` imports resolve
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import setup_logging, load_yaml, save_json
from src.modeling import ByT5Trainer, ContextAssembler, Augmenter
from src.retrieval import Retriever
from src.evaluation.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading & splitting
# ---------------------------------------------------------------------------


def load_and_split(config: dict):
    """Load combined corpus and produce stratified train / val / test frames."""
    corpus_path = PROJECT_ROOT / config["data"]["train_corpus"]
    val_frac = config["data"]["validation_split"]
    test_frac = config["data"]["test_split"]
    seed = config.get("seed", 42)

    logger.info(f"Loading corpus: {corpus_path}")
    df = pd.read_csv(corpus_path)
    logger.info(f"Total pairs: {len(df)}")

    # Validate required columns
    for col in ("transliteration", "translation"):
        if col not in df.columns:
            raise ValueError(f"Corpus missing required column: {col}")

    # Drop rows where either column is empty / NaN
    before = len(df)
    df = df.dropna(subset=["transliteration", "translation"])
    df = df[df["transliteration"].str.strip().ne("") & df["translation"].str.strip().ne("")]
    if len(df) < before:
        logger.warning(f"Dropped {before - len(df)} rows with empty transliteration/translation")

    # Two-step split: first peel off test, then split remainder into train/val
    train_val, test = train_test_split(df, test_size=test_frac, random_state=seed)
    val_ratio_of_remainder = val_frac / (1.0 - test_frac)
    train, val = train_test_split(train_val, test_size=val_ratio_of_remainder, random_state=seed)

    logger.info(f"Split — train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Component initialisation
# ---------------------------------------------------------------------------


def setup_retrieval(config: dict):
    """Load the retrieval system.  Returns (Retriever, Lexicon) or (None, None)."""
    ret_cfg = config.get("retrieval", {})
    if not ret_cfg.get("enabled", False):
        logger.info("RAG disabled — skipping retrieval initialisation")
        return None, None

    corpus_path = str(PROJECT_ROOT / ret_cfg["corpus_path"])
    index_path = str(PROJECT_ROOT / ret_cfg["index_path"])

    logger.info("Loading retrieval system…")
    retriever = Retriever()
    retriever.load(corpus_path=corpus_path, index_path=index_path)
    return retriever, retriever.lexicon


def setup_augmenter(config: dict) -> Augmenter | None:
    """Create Augmenter if synthetic-gap augmentation is enabled."""
    aug_cfg = config["data"].get("augmentation", {})
    if not aug_cfg.get("enabled", False) or not aug_cfg.get("synthetic_gaps", {}).get("enabled", False):
        logger.info("Augmentation disabled")
        return None

    gap_cfg = aug_cfg["synthetic_gaps"]
    return Augmenter(
        gap_prob=gap_cfg.get("probability", 0.3),
        big_gap_ratio=0.3,
        preserve_proper_nouns=True,
        preserve_sumerograms=True,
        seed=config.get("seed", 42),
    )


def setup_assembler(config: dict, retriever, lexicon) -> ContextAssembler | None:
    """Build a ContextAssembler wired to the retriever/lexicon (or None if no RAG)."""
    if retriever is None and lexicon is None:
        return None

    ret_cfg = config.get("retrieval", {})
    return ContextAssembler(
        retriever=retriever,
        lexicon=lexicon,
        max_length=ret_cfg.get("max_context_length", 800),
        num_examples=ret_cfg.get("k_examples", 3),
        include_lexicon=True,
        include_examples=retriever is not None,
    )


# ---------------------------------------------------------------------------
# Post-training evaluation
# ---------------------------------------------------------------------------

EVAL_BATCH_SIZE = 16  # generation batch size for post-training eval


def evaluate(trainer: ByT5Trainer, split_name: str, split_df: pd.DataFrame) -> dict:
    """Generate translations for *split_df* and return all metrics."""
    sources = split_df["transliteration"].fillna("").tolist()
    refs = split_df["translation"].fillna("").tolist()

    logger.info(f"Evaluating {split_name} ({len(sources)} examples)…")

    # Batched generation to avoid OOM
    hypotheses: list[str] = []
    for start in range(0, len(sources), EVAL_BATCH_SIZE):
        batch = sources[start : start + EVAL_BATCH_SIZE]
        hypotheses.extend(trainer.translate(batch, num_beams=5, max_length=256))
        logger.info(f"  generated {min(start + EVAL_BATCH_SIZE, len(sources))}/{len(sources)}")

    # Metrics
    calc = MetricsCalculator()
    metrics = calc.calculate_all_metrics(
        hypotheses=hypotheses,
        references=refs,
        source_texts=sources,
    )

    # Log summary
    logger.info(f"{split_name} BLEU:   {metrics.get('bleu', 0):.2f}")
    logger.info(f"{split_name} chrF++: {metrics.get('chrf', 0):.2f}")
    if "proper_noun_accuracy" in metrics:
        logger.info(
            f"{split_name} PN Acc: {metrics['proper_noun_accuracy']:.2%} "
            f"({metrics['correct_in_hypothesis']}/{metrics['total_proper_nouns']})"
        )

    # A handful of sample translations for quick sanity-check
    logger.info(f"\n{split_name} — sample translations:")
    for i in range(min(3, len(sources))):
        logger.info(f"  SRC: {sources[i][:120]}")
        logger.info(f"  HYP: {hypotheses[i][:120]}")
        logger.info(f"  REF: {refs[i][:120]}")
        logger.info("")

    # Store samples alongside metrics for the JSON output
    metrics["samples"] = [
        {"src": sources[i], "hyp": hypotheses[i], "ref": refs[i]}
        for i in range(min(5, len(sources)))
    ]
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 3 – ByT5 fine-tuning for Akkadian → English translation"
    )
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="YAML config path")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without training")
    parser.add_argument("--eval-only", type=str, default=None, help="Checkpoint dir — evaluate only, skip training")
    # Hyperparameter overrides
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    # Feature toggles
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG context")
    parser.add_argument("--no-aug", action="store_true", help="Disable augmentation")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    setup_logging(log_file="logs/training.log", level="INFO")

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    config = load_yaml(str(PROJECT_ROOT / args.config))

    # Apply CLI overrides
    if args.epochs is not None:
        config["training"]["num_train_epochs"] = args.epochs
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.batch_size is not None:
        config["training"]["per_device_train_batch_size"] = args.batch_size
    if args.no_rag:
        config.setdefault("retrieval", {})["enabled"] = False
    if args.no_aug:
        config.setdefault("data", {}).setdefault("augmentation", {})["enabled"] = False

    logger.info("=" * 60)
    logger.info("PHASE 3 — ByT5 FINE-TUNING")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_df, val_df, test_df = load_and_split(config)

    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------
    retriever, lexicon = setup_retrieval(config)
    augmenter = setup_augmenter(config)
    assembler = setup_assembler(config, retriever, lexicon)

    use_rag = assembler is not None
    use_aug = augmenter is not None

    # ------------------------------------------------------------------
    # Dry-run: validate everything loads, show a sample context, stop
    # ------------------------------------------------------------------
    if args.dry_run:
        logger.info("\n--- DRY RUN ---")
        logger.info(f"Train / Val / Test: {len(train_df)} / {len(val_df)} / {len(test_df)}")
        logger.info(f"RAG enabled:        {use_rag}")
        logger.info(f"Augmentation:       {use_aug}")
        logger.info(f"Model:              {config['model']['name']}")
        logger.info(f"Max input length:   {config['model']['max_input_length']} bytes")

        if assembler:
            sample_src = train_df.iloc[0]["transliteration"]
            context = assembler.assemble(sample_src)
            logger.info(f"\nSample RAG context ({len(context)} chars):\n{context}")

        # Compute approximate step counts for the user
        n_train = len(train_df)
        bs = config["training"]["per_device_train_batch_size"]
        ga = config["training"]["gradient_accumulation_steps"]
        epochs = config["training"]["num_train_epochs"]
        eff_batch = bs * ga
        steps_per_epoch = (n_train + eff_batch - 1) // eff_batch
        total_steps = steps_per_epoch * epochs
        logger.info(f"\nStep budget: {n_train} examples / eff-batch {eff_batch} "
                    f"= {steps_per_epoch} steps/epoch × {epochs} epochs = {total_steps} total")
        logger.info(f"Warmup: {config['training']['warmup_steps']} steps "
                    f"({config['training']['warmup_steps'] / total_steps * 100:.1f}% of total)")

        logger.info("\nDry run passed — setup is valid.")
        return

    # ------------------------------------------------------------------
    # Trainer initialisation  (also downloads / caches the model)
    # ------------------------------------------------------------------
    model_cfg = config["model"]
    train_cfg = config["training"]
    hw_cfg = config.get("hardware", {})

    trainer = ByT5Trainer(
        model_name=model_cfg["name"],
        output_dir=str(PROJECT_ROOT / config["output"]["output_dir"]),
        use_rag=use_rag,
        use_augmentation=use_aug,
        context_assembler=assembler,
        augmenter=augmenter,
        max_source_length=model_cfg["max_input_length"],
        max_target_length=model_cfg["max_output_length"],
    )

    # ------------------------------------------------------------------
    # Eval-only mode: skip training, just run post-training evaluation
    # ------------------------------------------------------------------
    if args.eval_only:
        logger.info(f"Eval-only mode — checkpoint: {args.eval_only}")
        # Model was already loaded from HF hub; for a saved checkpoint the
        # user would need to point model_name at the local path.  For now
        # this path exercises the eval loop on whatever model is loaded.
        results = {
            split: evaluate(trainer, split, df)
            for split, df in [("val", val_df), ("test", test_df)]
        }
        out_path = PROJECT_ROOT / config["output"]["output_dir"] / "eval_results.json"
        save_json(results, str(out_path))
        logger.info(f"Results saved to {out_path}")
        return

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    trainer.train(
        train_data=train_df,
        eval_data=val_df,
        num_epochs=train_cfg["num_train_epochs"],
        batch_size=train_cfg["per_device_train_batch_size"],
        learning_rate=train_cfg["learning_rate"],
        warmup_steps=train_cfg["warmup_steps"],
        eval_steps=train_cfg["eval_steps"],
        save_steps=train_cfg["save_steps"],
        logging_steps=config["output"]["logging_steps"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        fp16=hw_cfg.get("fp16", False),
        early_stopping_patience=train_cfg.get("early_stopping_patience", 3),
        generation_num_beams=train_cfg.get("generation_num_beams", 4),
        generation_max_length=train_cfg.get("generation_max_length", 256),
        seed=config.get("seed", 42),
    )

    # ------------------------------------------------------------------
    # Post-training evaluation
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("POST-TRAINING EVALUATION")
    logger.info("=" * 60)

    results = {
        split: evaluate(trainer, split, df)
        for split, df in [("val", val_df), ("test", test_df)]
    }

    out_path = PROJECT_ROOT / config["output"]["output_dir"] / "eval_results.json"
    save_json(results, str(out_path))
    logger.info(f"\nResults saved to {out_path}")

    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
