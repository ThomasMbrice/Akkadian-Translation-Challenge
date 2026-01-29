"""
Competition formatting rules and constants for Old Assyrian preprocessing.

Based on the Deep Past Challenge specification.
"""

import re
from typing import List, Dict

# Characters to remove (competition spec)
REMOVE_CHARS = [
    "!",  # Certainty marker
    "?",  # Uncertainty marker
    "/",  # Gloss separator
    ":",  # Colon
    ".",  # Period
    "˹",  # Partial damage start
    "˺",  # Partial damage end
]

# Patterns to remove (including content)
REMOVE_PATTERNS = [
    r"<<.*?>>",  # Erasure markers
]

# Patterns to preserve (keep content inside)
PRESERVE_PATTERNS = {
    "damaged": r"\[.*?\]",  # Damaged text
    "correction": r"<.*?>",  # Scribal corrections
    "determinative": r"\{.*?\}",  # Determinatives
}

# Gap normalization patterns
SINGLE_GAP_PATTERNS = [
    r"\[x\]",
    r"\[\?\.\]",
]

BIG_GAP_PATTERNS = [
    r"\[x x\]",
    r"\[x x x\]",
    r"\[x x x x\]",
    r"\[x x x x x\]",
    r"\[\.\.\.\]",
    r"\[n lines broken\]",
    r"\[broken\]",
    r"\[lost\]",
    r"\[missing\]",
]

# Gap replacements
SINGLE_GAP_TOKEN = "<gap>"
BIG_GAP_TOKEN = "<big_gap>"

# Feature extraction patterns
PROPER_NOUN_PATTERN = r"\b[A-ZĀĒĪŪṢṬŠḪ][a-zA-Zāēīūṣṭšḫ-]*\b"
SUMEROGRAM_PATTERN = r"\b[A-Z]{2,}(?:\.[A-Z]+)*\b"
DETERMINATIVE_PATTERN = r"\{[^}]+\}"

# Proper noun indicators (capital letters including diacritics)
CAPITAL_LETTERS = set("AĀEĒIĪUŪṢṬŠḪBCDFGHJKLMNPQRVWXYZ")

# Minimum lengths for quality filtering
MIN_AKKADIAN_LENGTH = 3
MAX_AKKADIAN_LENGTH = 1000
MIN_ENGLISH_LENGTH = 3
MAX_ENGLISH_LENGTH = 2000

# Alignment validation
MAX_LENGTH_RATIO = 3.0  # Maximum Akkadian/English length ratio


def build_remove_chars_pattern() -> str:
    """Build regex pattern for characters to remove."""
    escaped = [re.escape(char) for char in REMOVE_CHARS]
    return f"[{''.join(escaped)}]"


def build_gap_normalization_patterns() -> Dict[str, str]:
    """Build compiled patterns for gap normalization."""
    return {
        "single": "|".join(SINGLE_GAP_PATTERNS),
        "big": "|".join(BIG_GAP_PATTERNS),
    }
