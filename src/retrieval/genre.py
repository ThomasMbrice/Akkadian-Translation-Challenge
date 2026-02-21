"""
Rule-based genre classifier for Old Assyrian Akkadian transliteration.

Classifies texts into: letter, legal, commercial, administrative, unknown.
Uses Akkadian-side heuristics (not English translation).
"""

import re
from typing import Optional

# Akkadian-side commodity Sumerograms (commercial documents)
_COMMODITY_SG = {"ANNA", "TÚG", "KÙ.BABBAR", "KÙBABBAR", "GÚ", "GÍN", "MA.NA", "MANA"}


def classify_genre(transliteration: str) -> str:
    """
    Classify Akkadian transliteration into a document genre.

    Priority order: letter > legal > commercial > administrative > unknown.

    Args:
        transliteration: Akkadian transliteration text

    Returns:
        One of: "letter", "legal", "commercial", "administrative", "unknown"
    """
    # Letter: epistolary formula on the Akkadian side
    if re.search(r'um-ma.{0,100}qí-bi[₄]?-ma', transliteration) or \
       re.search(r'a-na.{0,100}qí-bi[₄]?-ma', transliteration):
        return "letter"

    # Legal: witness marker IGI, oath/tablet formulas
    if "IGI" in transliteration:
        return "legal"
    if re.search(r'\b(ṭup-pa-am|ṭup-pí|ma-ma-an|qá-qá-ad|iš-tí)\b', transliteration):
        return "legal"

    # Commercial: 2+ commodity Sumerograms without letter framing
    upper = transliteration.upper()
    commodity_hits = sum(1 for sg in _COMMODITY_SG if sg in upper)
    if commodity_hits >= 2:
        return "commercial"

    # Administrative: 3+ numeric tokens (quantities/counts in lists)
    if len(re.findall(r'\b\d+\b', transliteration)) >= 3:
        return "administrative"

    return "unknown"


def detect_letter_formula(transliteration: str) -> Optional[dict]:
    """
    Extract sender/recipient from Old Assyrian letter formula.

    Handles two common orderings:
      Pattern A: um-ma {sender} ... a-na {recipient} qí-bi-ma
      Pattern B: a-na {recipient} qí-bi-ma um-ma {sender}-ma

    Args:
        transliteration: Akkadian transliteration (ideally first ~30 tokens)

    Returns:
        {"sender": str, "recipient": str} or None if not detected
    """
    # Pattern A: um-ma X ... a-na Y qí-bi-ma
    m = re.search(
        r'um-ma\s+(.{3,50}?)\s+a-na\s+(.{3,50}?)\s+qí-bi[₄]?-ma',
        transliteration,
    )
    if m:
        return {"sender": m.group(1).strip(), "recipient": m.group(2).strip()}

    # Pattern B: a-na Y qí-bi-ma um-ma X-ma
    m = re.search(
        r'a-na\s+(.{3,50}?)\s+qí-bi[₄]?-ma\s+um-ma\s+(.{3,50}?)-ma',
        transliteration,
    )
    if m:
        return {"sender": m.group(2).strip(), "recipient": m.group(1).strip()}

    return None
