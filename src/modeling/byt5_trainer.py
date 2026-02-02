"""
ByT5 trainer for Old Assyrian → English translation.

Uses Hugging Face Transformers Trainer API for fine-tuning ByT5 models
on parallel Akkadian-English pairs with optional RAG context.

Models supported:
- google/byt5-small (300M params, fast)
- google/byt5-base (580M params, balanced)
- google/byt5-large (1.2B params, slow, best quality)

Usage:
    from src.modeling import ByT5Trainer

    trainer = ByT5Trainer(
        model_name="google/byt5-small",
        output_dir="models/byt5_finetuned",
    )

    trainer.train(
        train_data=train_df,
        eval_data=eval_df,
        num_epochs=5,
    )
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

from src.modeling.context_assembler import ContextAssembler
from src.modeling.augmentation import Augmenter

logger = logging.getLogger(__name__)


class ByT5Trainer:
    """
    ByT5 fine-tuning trainer for Akkadian→English translation.

    Handles data preparation, model initialization, and training.
    """

    def __init__(
        self,
        model_name: str = "google/byt5-small",
        output_dir: str = "models/byt5_finetuned",
        use_rag: bool = False,
        use_augmentation: bool = False,
        context_assembler: Optional[ContextAssembler] = None,
        augmenter: Optional[Augmenter] = None,
        max_source_length: int = 512,
        max_target_length: int = 256,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_name: Hugging Face model ID (byt5-small/base/large)
            output_dir: Directory to save checkpoints and final model
            use_rag: Whether to use RAG context (requires context_assembler)
            use_augmentation: Whether to augment training data with gaps
            context_assembler: ContextAssembler instance (for RAG)
            augmenter: Augmenter instance (for data augmentation)
            max_source_length: Maximum input length in bytes
            max_target_length: Maximum output length in bytes
            device: Device to use ("cuda", "mps", "cpu", or None for auto)
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.use_rag = use_rag
        self.use_augmentation = use_augmentation
        self.context_assembler = context_assembler
        self.augmenter = augmenter
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        # Initialize tokenizer and model
        logger.info(f"Loading tokenizer and model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)

        logger.info(
            f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M"
        )

        # Validate RAG/augmentation setup
        if self.use_rag and self.context_assembler is None:
            raise ValueError("use_rag=True requires context_assembler")
        if self.use_augmentation and self.augmenter is None:
            logger.warning("use_augmentation=True but no augmenter provided")

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_data(
        self,
        train_df: pd.DataFrame,
        eval_df: Optional[pd.DataFrame] = None,
        transliteration_col: str = "transliteration",
        translation_col: str = "translation",
    ) -> Tuple[Dataset, Optional[Dataset]]:
        """
        Prepare training and evaluation datasets.

        Args:
            train_df: Training data with transliteration, translation columns
            eval_df: Evaluation data (optional)
            transliteration_col: Column name for source text
            translation_col: Column name for target text

        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        logger.info("Preparing datasets...")

        # Prepare training data
        train_sources = train_df[transliteration_col].fillna("").tolist()
        train_targets = train_df[translation_col].fillna("").tolist()

        # Apply augmentation if enabled
        if self.use_augmentation and self.augmenter is not None:
            logger.info("Applying data augmentation...")
            train_sources = self.augmenter.augment_batch(train_sources)

        # Apply RAG context if enabled
        if self.use_rag and self.context_assembler is not None:
            logger.info("Assembling RAG context...")
            train_sources = self.context_assembler.assemble_batch(
                train_sources,
                include_instruction=True,
            )

        train_dataset = self._create_dataset(train_sources, train_targets)
        logger.info(f"Training dataset: {len(train_dataset)} examples")

        # Prepare evaluation data
        eval_dataset = None
        if eval_df is not None:
            eval_sources = eval_df[transliteration_col].fillna("").tolist()
            eval_targets = eval_df[translation_col].fillna("").tolist()

            # Apply RAG to eval (no augmentation)
            if self.use_rag and self.context_assembler is not None:
                eval_sources = self.context_assembler.assemble_batch(
                    eval_sources,
                    include_instruction=True,
                )

            eval_dataset = self._create_dataset(eval_sources, eval_targets)
            logger.info(f"Evaluation dataset: {len(eval_dataset)} examples")

        return train_dataset, eval_dataset

    def _create_dataset(
        self,
        sources: List[str],
        targets: List[str],
    ) -> Dataset:
        """
        Create Hugging Face Dataset from source/target lists.

        Args:
            sources: List of source texts (transliterations or contexts)
            targets: List of target texts (translations)

        Returns:
            Dataset ready for training
        """
        # Tokenize
        model_inputs = self.tokenizer(
            sources,
            max_length=self.max_source_length,
            truncation=True,
            padding=False,  # Will pad in collator
        )

        # Tokenize targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                max_length=self.max_target_length,
                truncation=True,
                padding=False,
            )

        # Create dataset
        data_dict = {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels["input_ids"],
        }

        return Dataset.from_dict(data_dict)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_data: pd.DataFrame,
        eval_data: Optional[pd.DataFrame] = None,
        num_epochs: int = 5,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        warmup_steps: int = 100,
        eval_steps: int = 200,
        save_steps: int = 500,
        logging_steps: int = 50,
        gradient_accumulation_steps: int = 1,
        fp16: bool = False,
        transliteration_col: str = "transliteration",
        translation_col: str = "translation",
    ) -> None:
        """
        Train the model.

        Args:
            train_data: Training DataFrame
            eval_data: Evaluation DataFrame (optional)
            num_epochs: Number of training epochs
            batch_size: Per-device batch size
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            eval_steps: Evaluate every N steps
            save_steps: Save checkpoint every N steps
            logging_steps: Log every N steps
            gradient_accumulation_steps: Gradient accumulation steps
            fp16: Use mixed precision (requires CUDA)
            transliteration_col: Source column name
            translation_col: Target column name
        """
        # Prepare datasets
        train_dataset, eval_dataset = self.prepare_data(
            train_data,
            eval_data,
            transliteration_col=transliteration_col,
            translation_col=translation_col,
        )

        # Set up training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps if eval_dataset is not None else None,
            evaluation_strategy="steps" if eval_dataset is not None else "no",
            save_total_limit=3,
            load_best_model_at_end=eval_dataset is not None,
            metric_for_best_model="eval_loss" if eval_dataset is not None else None,
            greater_is_better=False,
            gradient_accumulation_steps=gradient_accumulation_steps,
            fp16=fp16 and self.device == "cuda",
            report_to=["tensorboard"],
            push_to_hub=False,
            predict_with_generate=True,
        )

        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            padding=True,
        )

        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        # Train
        logger.info("=" * 60)
        logger.info("STARTING TRAINING")
        logger.info("=" * 60)
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Training examples: {len(train_dataset)}")
        logger.info(f"Eval examples: {len(eval_dataset) if eval_dataset else 0}")
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Device: {self.device}")
        logger.info("=" * 60)

        trainer.train()

        # Save final model
        final_model_path = self.output_dir / "final"
        trainer.save_model(str(final_model_path))
        self.tokenizer.save_pretrained(str(final_model_path))

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Model saved to: {final_model_path}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def translate(
        self,
        transliterations: List[str],
        num_beams: int = 5,
        max_length: int = 256,
    ) -> List[str]:
        """
        Translate Akkadian transliterations to English.

        Args:
            transliterations: List of transliterations
            num_beams: Number of beams for beam search
            max_length: Maximum output length

        Returns:
            List of English translations
        """
        # Apply RAG context if enabled
        inputs = transliterations
        if self.use_rag and self.context_assembler is not None:
            inputs = self.context_assembler.assemble_batch(
                transliterations,
                include_instruction=True,
            )

        # Tokenize
        encoded = self.tokenizer(
            inputs,
            max_length=self.max_source_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Generate
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **encoded,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
            )

        # Decode
        translations = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
        )

        return translations
