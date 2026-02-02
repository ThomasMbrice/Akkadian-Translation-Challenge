"""
Lexicon builder for proper nouns and Sumerograms.

Loads lexical data from OA_Lexicon_eBL.csv and eBL_Dictionary.csv to provide:
1. Proper noun dictionary (person names, geographic names)
2. Sumerogram → English mappings (e.g., DUMU → "son", KÙ.BABBAR → "silver")
3. Fuzzy matching for inflected forms

Usage:
    lexicon = Lexicon()
    lexicon.load()

    # Look up proper noun
    result = lexicon.lookup_proper_noun("A-šùr-i-mì-tí")
    # Returns: {"form": "A-šùr-i-mì-tí", "norm": "Aššur-imittī", "type": "PN"}

    # Look up Sumerogram
    result = lexicon.lookup_sumerogram("DUMU")
    # Returns: "son"
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher

import pandas as pd

logger = logging.getLogger(__name__)


class Lexicon:
    """
    Lexicon for Old Assyrian proper nouns and Sumerograms.

    Provides exact and fuzzy lookup for transliterated forms.
    """

    def __init__(
        self,
        oa_lexicon_path: str = "data/raw/deep-past-initiative-machine-translation/OA_Lexicon_eBL.csv",
        ebl_dict_path: str = "data/raw/deep-past-initiative-machine-translation/eBL_Dictionary.csv",
        fuzzy_threshold: float = 0.8,
    ):
        """
        Args:
            oa_lexicon_path: Path to OA_Lexicon_eBL.csv
            ebl_dict_path: Path to eBL_Dictionary.csv
            fuzzy_threshold: Minimum similarity score for fuzzy matches (0.0-1.0)
        """
        self.oa_lexicon_path = oa_lexicon_path
        self.ebl_dict_path = ebl_dict_path
        self.fuzzy_threshold = fuzzy_threshold

        # Storage
        self.proper_nouns: Dict[str, Dict] = {}  # form → {norm, type, lexeme}
        self.sumerograms: Dict[str, str] = {}     # SUMEROGRAM → English
        self.dictionary: Dict[str, str] = {}      # word → definition

        # For fuzzy matching
        self._proper_noun_forms: List[str] = []
        self._sumerogram_forms: List[str] = []

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load lexical data from CSV files."""
        logger.info("Loading lexicon data...")

        # Load OA Lexicon
        oa_df = pd.read_csv(self.oa_lexicon_path)
        logger.info(f"Loaded {len(oa_df)} entries from OA_Lexicon_eBL.csv")

        # Split by type
        for _, row in oa_df.iterrows():
            form = str(row["form"]).strip()
            norm = str(row["norm"]).strip() if pd.notna(row["norm"]) else form
            lexeme = str(row["lexeme"]).strip() if pd.notna(row["lexeme"]) else norm
            entry_type = str(row["type"]).strip()

            if entry_type in ["PN", "GN"]:  # Person name, Geographic name
                self.proper_nouns[form] = {
                    "norm": norm,
                    "lexeme": lexeme,
                    "type": entry_type,
                }

        self._proper_noun_forms = list(self.proper_nouns.keys())
        logger.info(f"Loaded {len(self.proper_nouns)} proper nouns (PN + GN)")

        # Load eBL Dictionary
        ebl_df = pd.read_csv(self.ebl_dict_path)
        logger.info(f"Loaded {len(ebl_df)} entries from eBL_Dictionary.csv")

        for _, row in ebl_df.iterrows():
            word = str(row["word"]).strip()
            definition = str(row["definition"]).strip() if pd.notna(row["definition"]) else ""

            # Check if all-caps (likely Sumerogram)
            if word.isupper() and len(word) >= 2:
                # Clean definition (remove quotes, extra whitespace)
                clean_def = definition.strip('"').strip()
                if clean_def:
                    self.sumerograms[word] = clean_def

            # Store all definitions
            if definition:
                self.dictionary[word] = definition.strip('"').strip()

        self._sumerogram_forms = list(self.sumerograms.keys())
        logger.info(f"Loaded {len(self.sumerograms)} Sumerograms")
        logger.info(f"Total dictionary entries: {len(self.dictionary)}")

    # ------------------------------------------------------------------
    # Lookup: Exact
    # ------------------------------------------------------------------

    def lookup_proper_noun(self, form: str, fuzzy: bool = False) -> Optional[Dict]:
        """
        Look up a proper noun (person or geographic name).

        Args:
            form: Transliterated form (e.g., "A-šùr-i-mì-tí")
            fuzzy: If True, attempt fuzzy matching if exact match fails

        Returns:
            Dict with keys: norm, lexeme, type, similarity (if fuzzy)
            None if no match found
        """
        # Exact match
        if form in self.proper_nouns:
            result = self.proper_nouns[form].copy()
            result["form"] = form
            result["similarity"] = 1.0
            return result

        # Fuzzy match
        if fuzzy:
            return self._fuzzy_match(form, self._proper_noun_forms, "proper_noun")

        return None

    def lookup_sumerogram(self, sumerogram: str) -> Optional[str]:
        """
        Look up a Sumerogram (e.g., DUMU → "son").

        Args:
            sumerogram: All-caps Sumerogram form

        Returns:
            English definition, or None if not found
        """
        # Exact match
        if sumerogram in self.sumerograms:
            return self.sumerograms[sumerogram]

        # Try with dots removed (e.g., "KÙ.BABBAR" → "KÙBABBAR")
        normalized = sumerogram.replace(".", "")
        if normalized in self.sumerograms:
            return self.sumerograms[normalized]

        return None

    def lookup_dictionary(self, word: str) -> Optional[str]:
        """
        Look up a word in the general dictionary.

        Args:
            word: Word form

        Returns:
            Definition string, or None if not found
        """
        return self.dictionary.get(word)

    # ------------------------------------------------------------------
    # Lookup: Fuzzy
    # ------------------------------------------------------------------

    def _fuzzy_match(
        self,
        query: str,
        candidates: List[str],
        match_type: str,
    ) -> Optional[Dict]:
        """
        Find best fuzzy match using sequence similarity.

        Args:
            query: Search string
            candidates: List of candidate strings
            match_type: "proper_noun" or "sumerogram"

        Returns:
            Match dict with similarity score, or None if below threshold
        """
        if not candidates:
            return None

        best_match = None
        best_score = 0.0

        for candidate in candidates:
            score = SequenceMatcher(None, query.lower(), candidate.lower()).ratio()
            if score > best_score:
                best_score = score
                best_match = candidate

        if best_score >= self.fuzzy_threshold:
            if match_type == "proper_noun":
                result = self.proper_nouns[best_match].copy()
                result["form"] = best_match
                result["similarity"] = best_score
                return result
            elif match_type == "sumerogram":
                return {
                    "sumerogram": best_match,
                    "definition": self.sumerograms[best_match],
                    "similarity": best_score,
                }

        return None

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def extract_proper_nouns(self, text: str) -> List[Dict]:
        """
        Extract all proper nouns from transliterated text.

        Uses capitalization heuristic: words that start with capital letter
        and contain hyphens (Old Assyrian convention).

        Args:
            text: Transliterated Akkadian text

        Returns:
            List of proper noun matches with metadata
        """
        # Pattern: Capitalized word with optional hyphens and diacritics
        pattern = r'\b[A-Z][a-zšṭṣḫāēīūṢŠḤĪŪĀĒṦ\-]+\b'
        candidates = re.findall(pattern, text)

        results = []
        for candidate in candidates:
            match = self.lookup_proper_noun(candidate, fuzzy=True)
            if match:
                results.append(match)

        return results

    def extract_sumerograms(self, text: str) -> List[Dict]:
        """
        Extract all Sumerograms from transliterated text.

        Uses all-caps heuristic: words in ALL CAPS with optional dots.

        Args:
            text: Transliterated Akkadian text

        Returns:
            List of Sumerogram matches with definitions
        """
        # Pattern: All-caps words (2+ chars) with optional dots
        pattern = r'\b[A-Z]{2,}(?:\.[A-Z]+)*\b'
        candidates = re.findall(pattern, text)

        results = []
        for candidate in candidates:
            definition = self.lookup_sumerogram(candidate)
            if definition:
                results.append({
                    "sumerogram": candidate,
                    "definition": definition,
                })

        return results

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, int]:
        """Return lexicon statistics."""
        return {
            "proper_nouns": len(self.proper_nouns),
            "sumerograms": len(self.sumerograms),
            "dictionary_entries": len(self.dictionary),
        }
