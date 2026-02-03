"""
Context assembler for ByT5 translation with RAG.

Combines lexicon lookups and retrieved translation examples into a
formatted context string that ByT5 can use for translation.

Format:
    Lexicon:
    - DUMU = son
    - A-šùr-i-mì-tí = Aššur-imittī (person name)

    Similar translations:
    [Example 1: Akkadian → English]
    [Example 2: Akkadian → English]

    Translate: [input transliteration]

Usage:
    assembler = ContextAssembler(retriever, max_length=800)
    context = assembler.assemble(
        transliteration="a-na A-šùr-i-mì-tí DUMU Ṣí-lí-{d}UTU qí-bi-ma"
    )
"""

import logging
from typing import List, Dict, Optional

from src.retrieval import Retriever, Lexicon

logger = logging.getLogger(__name__)


class ContextAssembler:
    """
    Assembles RAG context for ByT5 translation.

    Combines lexicon and retrieved examples into a formatted prompt.
    """

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        lexicon: Optional[Lexicon] = None,
        max_length: int = 800,
        num_examples: int = 3,
        include_lexicon: bool = True,
        include_examples: bool = True,
    ):
        """
        Args:
            retriever: Retriever instance (optional, for examples)
            lexicon: Lexicon instance (optional, for lookups)
            max_length: Maximum context length in characters (ByT5 limit ~1024 bytes)
            num_examples: Number of retrieved examples to include
            include_lexicon: Whether to include lexicon lookups
            include_examples: Whether to include retrieved examples
        """
        self.retriever = retriever
        self.lexicon = lexicon
        self.max_length = max_length
        self.num_examples = num_examples
        self.include_lexicon = include_lexicon
        self.include_examples = include_examples

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------

    def assemble(
        self,
        transliteration: str,
        include_instruction: bool = True,
    ) -> str:
        """
        Assemble full context for translation.

        Args:
            transliteration: Akkadian transliteration to translate
            include_instruction: Whether to include "Translate:" instruction

        Returns:
            Formatted context string
        """
        parts = []

        # 1. Lexicon lookups
        if self.include_lexicon and self.lexicon is not None:
            lexicon_text = self._format_lexicon(transliteration)
            if lexicon_text:
                parts.append(lexicon_text)

        # 2. Retrieved examples
        if self.include_examples and self.retriever is not None:
            examples_text = self._format_examples(transliteration)
            if examples_text:
                parts.append(examples_text)

        # 3. Input instruction
        if include_instruction:
            parts.append(f"Translate: {transliteration}")

        # Join and truncate
        context = "\n\n".join(parts)
        if len(context) > self.max_length:
            context = self._truncate_context(context, parts)

        return context

    def assemble_batch(
        self,
        transliterations: List[str],
        include_instruction: bool = True,
    ) -> List[str]:
        """
        Assemble context for a batch of transliterations.

        Args:
            transliterations: List of transliterations
            include_instruction: Whether to include "Translate:" instruction

        Returns:
            List of formatted context strings
        """
        return [
            self.assemble(t, include_instruction=include_instruction)
            for t in transliterations
        ]

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_lexicon(self, transliteration: str) -> str:
        """
        Format lexicon lookups for a transliteration.

        Extracts proper nouns and Sumerograms, looks them up, and formats.

        Args:
            transliteration: Akkadian transliteration

        Returns:
            Formatted lexicon section (or empty string if nothing found)
        """
        if self.lexicon is None:
            return ""

        entries = []

        # Extract proper nouns
        proper_nouns = self.lexicon.extract_proper_nouns(transliteration)
        for pn in proper_nouns:
            entries.append(f"- {pn['form']} = {pn['norm']} ({pn['type']})")

        # Extract Sumerograms
        sumerograms = self.lexicon.extract_sumerograms(transliteration)
        for sg in sumerograms:
            entries.append(f"- {sg['sumerogram']} = {sg['definition']}")

        if not entries:
            return ""

        # Deduplicate
        entries = list(dict.fromkeys(entries))  # Preserve order

        return "Lexicon:\n" + "\n".join(entries)

    def _format_examples(self, transliteration: str) -> str:
        """
        Format retrieved translation examples.

        Args:
            transliteration: Akkadian transliteration (query)

        Returns:
            Formatted examples section (or empty string if retriever unavailable)
        """
        if self.retriever is None:
            return ""

        # Retrieve examples
        try:
            results = self.retriever.retrieve(
                transliteration,
                k=self.num_examples,
                use_lexicon=True,
            )
        except Exception as e:
            logger.warning(f"Retrieval failed: {e}")
            return ""

        if not results:
            return ""

        # Format examples
        examples = []
        for i, result in enumerate(results, 1):
            # Truncate if too long
            trans_lit = result["transliteration"][:80]
            translation = result["translation"][:100]
            examples.append(f"[{i}] {trans_lit} → {translation}")

        return "Similar translations:\n" + "\n".join(examples)

    def _truncate_context(self, context: str, parts: List[str]) -> str:
        """
        Truncate context to fit within max_length.

        Strategy:
        1. Always keep the "Translate:" instruction (last part)
        2. Truncate lexicon and examples proportionally

        Args:
            context: Full context string
            parts: List of context parts (lexicon, examples, instruction)

        Returns:
            Truncated context
        """
        if len(parts) == 1:
            # Only instruction, truncate it
            return context[:self.max_length]

        # Always keep last part (instruction)
        instruction = parts[-1]
        budget = self.max_length - len(instruction) - 4  # Reserve for "\n\n"

        if len(parts) == 2:
            # Only lexicon or examples + instruction
            truncated = parts[0][:budget]
            return f"{truncated}\n\n{instruction}"

        # Both lexicon and examples
        # Give each half the budget
        half_budget = budget // 2
        lexicon_part = parts[0][:half_budget]
        examples_part = parts[1][:half_budget]

        return f"{lexicon_part}\n\n{examples_part}\n\n{instruction}"

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def estimate_length(self, transliteration: str) -> int:
        """
        Estimate context length without actually assembling.

        Args:
            transliteration: Transliteration to estimate for

        Returns:
            Estimated context length in characters
        """
        # Rough estimates
        lexicon_len = 100 if self.include_lexicon else 0
        examples_len = 300 if self.include_examples else 0
        instruction_len = len(transliteration) + 12  # "Translate: " + text

        return lexicon_len + examples_len + instruction_len
