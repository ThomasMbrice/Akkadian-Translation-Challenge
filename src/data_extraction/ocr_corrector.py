"""
OCR artifact correction for publication page text.

Handles common OCR errors in academic cuneiform publications:
- Running headers and footers repeated on every page
- Page numbers at text boundaries
- Whitespace normalization

Note: We intentionally do NOT attempt to fix line-break hyphenation,
because Akkadian itself uses hyphens as syllable separators (e.g.,
"a-na", "i-na"). Joining broken lines would corrupt transliterations.
"""

import re
import logging
from typing import List

logger = logging.getLogger(__name__)


class OCRCorrector:
    """Clean OCR artifacts from publication page text."""

    # Page number at start of line: standalone integer, optionally followed by
    # a single letter and a journal title (e.g., "322  F  Journal of Near Eastern Studies")
    _PAGE_NUM_RE = re.compile(r"^\d+\s*(?:\w\s+[A-Z][A-Za-z\s\.\,\-]+)?\s*$")

    # All-caps line that is likely a running header (author name or title).
    # Must be short, all uppercase, and contain no Akkadian diacritics.
    _AKKADIAN_DIACRITICS = set("šṭṣḫāēīūṣṭŠḤĪŪĀĒṦḫšṢ")

    def __init__(self):
        pass

    def correct(self, text: str) -> str:
        """
        Apply all OCR corrections to page text.

        Args:
            text: Raw OCR page text from a single publication page

        Returns:
            Cleaned text with headers/footers removed and whitespace normalized
        """
        if not text or not isinstance(text, str):
            return ""

        text = self._remove_page_headers(text)
        text = self._normalize_whitespace(text)
        return text.strip()

    def _remove_page_headers(self, text: str) -> str:
        """
        Remove page numbers and running headers from the start of text.

        Academic PDF OCR frequently prepends each page with:
        - A bare page number ("322")
        - A page number + journal abbreviation ("322  F  Journal of...")
        - An all-caps author/title line ("MOGEUS TROLLE LARSEN")

        We strip these leading lines until we hit content that looks like
        real body text (contains lowercase, punctuation, or Akkadian markers).
        """
        lines = text.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Empty line — skip
            if not line:
                i += 1
                continue

            # Bare page number or "NNN  X  Journal Title" style
            if self._PAGE_NUM_RE.match(line):
                i += 1
                continue

            # Short all-caps line with no Akkadian diacritics → running header
            if (
                line == line.upper()
                and len(line) < 80
                and not any(c in self._AKKADIAN_DIACRITICS for c in line)
                and any(c.isalpha() for c in line)
            ):
                i += 1
                continue

            # Reached real content
            break

        return "\n".join(lines[i:])

    def _normalize_whitespace(self, text: str) -> str:
        """Collapse runs of spaces/tabs; preserve newlines."""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" +\n", "\n", text)
        return text

    def correct_batch(self, texts: List[str]) -> List[str]:
        """Apply corrections to a batch of page texts."""
        return [self.correct(text) for text in texts]
