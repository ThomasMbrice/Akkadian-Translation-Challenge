"""
Publication parser for extracting transliteration-translation pairs.

The dominant pattern in Old Assyrian academic publications (tablet editions)
is a block of raw transliteration followed by its scholarly English translation.
The two are separated by either:
    - A colon:  "...A-šur-sig5: Ennānum owes Zikkur-ilī..."
    - A period: "...igi I-na-aḫ. Karšini and Alili owe..."

Pages in appendices/editions are also segmented by "Text N." headers, each
containing one tablet edition with its own transliteration:translation pair.

Extraction strategy:
1. Split page into segments by "Text N." headers (if present).
2. Within each segment, scan candidate split points (after `:` or `.`).
3. Score the text before and after each candidate using Akkadian and English
   heuristics.
4. Accept the split point with the highest combined score above thresholds.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Character / pattern sets
# ---------------------------------------------------------------------------

# Akkadian diacritical characters (transliteration alphabet)
AKKADIAN_DIACRITICS = set("šṭṣḫāēīūṢŠḤĪŪĀĒṦ")

# Sumerogram logograms: all-caps words of length >= 2, optionally with dots
SUMEROGRAM_RE = re.compile(r"\b[A-Z]{2,}(?:\.[A-Z]+)*\b")

# Gap markers from damaged tablets
GAP_RE = re.compile(r"\[[\w\s\.x\?\+]*\]")

# "Text N." or "Text Na." headers in tablet editions
TEXT_HEADER_RE = re.compile(r"\bText\s+\d+[a-z]?[\.\:]", re.IGNORECASE)

# Common Akkadian function words / formula fragments (lowercase)
AKKADIAN_KEYWORDS = {
    "dumu", "i-na", "a-na", "itu.kam", "li-mu-um", "iš-tù", "ḫa-mu",
    "kù.babbar", "ma-na", "gín", "um-ma", "igi", "ṣí-ib-tám", "ú-ṣa-áb",
    "i-ṣé-er", "i-šu", "ḫa-mu-uš-tim", "i-ša-qal", "il5-qé",
}

# Common English function words (high-confidence English indicators)
ENGLISH_FUNCTION_WORDS = {
    "the", "and", "of", "to", "in", "is", "that", "he", "she", "his",
    "her", "was", "will", "from", "with", "has", "have", "had", "not",
    "but", "or", "as", "at", "by", "be", "it", "its", "this", "for",
    "an", "are", "were", "been", "being", "which", "who", "whom",
    "they", "them", "their", "we", "our", "you", "your",
}

# Domain-specific English words common in Akkadian tablet translations
ENGLISH_DOMAIN_WORDS = {
    "owes", "silver", "mina", "minas", "shekel", "shekels", "son", "debt",
    "paid", "pay", "interest", "month", "week", "weeks", "witness",
    "seal", "tablet", "took", "gave", "received", "said", "wrote",
    "brother", "father", "copper", "tin", "textile", "textiles",
    "talent", "talents", "loaves", "bread", "donkey", "palace",
    "swore", "oath", "city", "claim", "claims", "owed",
}

# German function words — used to reject non-English translations
GERMAN_FUNCTION_WORDS = {
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer",
    "einem", "einen", "einer", "und", "oder", "aber", "wenn", "weil",
    "damit", "dass", "von", "mit", "aus", "nach", "für", "ist", "sind",
    "war", "werden", "hat", "haben", "nicht", "auch", "noch", "sich",
    "wie", "so", "nur", "schon", "immer", "jetzt", "hier", "dort",
    "sehr", "noch", "ob", "bei", "wie", "wird", "kann", "sein",
}

# Turkish function words — used to reject non-English translations
TURKISH_FUNCTION_WORDS = {
    "bir", "bu", "için", "olan", "ile", "veya", "ama", "ancak",
    "daha", "çok", "sonra", "önce", "hem", "ya", "da", "olan",
    "milyonu", "olarak", "olduğu", "edecek", "edildiğinden",
    "yılında", "oğlu", "gümüş", "satır", "metni", "teslim",
}

# French function words — used to reject non-English translations
FRENCH_FUNCTION_WORDS = {
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "ou",
    "mais", "donc", "car", "ni", "que", "qui", "quoi", "dont",
    "où", "est", "sont", "dans", "pour", "par", "sur", "avec",
    "sans", "sous", "entre", "vers", "depuis", "selon", "comme",
    "très", "bien", "aussi", "même", "plus", "moins", "tout",
    "tous", "toutes", "cette", "ces", "leur", "leurs", "ses",
    "notre", "nos", "votre", "vos", "leur", "ses", "même",
    "qui", "que", "quoi", "dont", "où", "il", "elle", "ils",
    "elles", "nous", "vous", "lui", "eux", "ce", "cela",
}


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------


def akkadian_score(text: str) -> float:
    """
    Score how likely a text segment is Akkadian transliteration.

    Combines:
    - Diacritic character density
    - Sumerogram presence
    - Gap marker presence
    - Known Akkadian keyword matches

    Returns:
        Float score; higher = more likely Akkadian. Roughly 0.3+ indicates
        a credible transliteration block.
    """
    if not text or len(text.strip()) < 5:
        return 0.0

    char_count = max(len(text), 1)

    # Diacritic density (most reliable single signal)
    diacritic_count = sum(1 for c in text if c in AKKADIAN_DIACRITICS)
    diacritic_density = diacritic_count / char_count

    # Sumerogram count (capped contribution)
    sumerogram_count = len(SUMEROGRAM_RE.findall(text))
    sumerogram_score = min(sumerogram_count / 3.0, 1.0)

    # Gap markers
    gap_count = len(GAP_RE.findall(text))
    gap_score = min(gap_count / 2.0, 1.0)

    # Akkadian keyword matches
    text_lower = text.lower()
    keyword_matches = sum(1 for kw in AKKADIAN_KEYWORDS if kw in text_lower)
    keyword_score = min(keyword_matches / 3.0, 1.0)

    # Weighted combination
    score = (
        diacritic_density * 5.0
        + sumerogram_score * 2.0
        + gap_score * 1.0
        + keyword_score * 2.0
    )
    return score


def english_score(text: str) -> float:
    """
    Score how likely a text segment is English translation.

    Uses function-word density and domain-word density. Does NOT penalise
    diacritics, because scholarly English translations routinely contain
    transliterated proper nouns (e.g., "Šumma-libbi-Aššur owes...").

    Returns:
        Float score; higher = more likely English. Roughly 0.15+ indicates
        credible English prose.
    """
    if not text or len(text.strip()) < 10:
        return 0.0

    words = text.split()
    if not words:
        return 0.0

    # Normalise: lowercase, strip punctuation
    clean_words = [w.lower().strip(".,!?;:'\"()[]") for w in words]
    total = max(len(clean_words), 1)

    function_matches = sum(1 for w in clean_words if w in ENGLISH_FUNCTION_WORDS)
    domain_matches = sum(1 for w in clean_words if w in ENGLISH_DOMAIN_WORDS)

    score = (function_matches / total) * 3.0 + (domain_matches / total) * 2.0
    return min(score, 2.0)


def non_english_score(text: str) -> float:
    """
    Score how likely a text segment is in a non-English language (German /
    Turkish / French).  Used to reject translations that are not in English —
    many AKT / APU volumes publish German or Turkish translations, and some
    NABU notes are in French.

    Returns:
        Float score; >= 1.0 indicates likely non-English translation.
    """
    if not text or len(text.strip()) < 10:
        return 0.0

    words = text.split()
    if not words:
        return 0.0

    clean_words = [w.lower().strip(".,!?;:'\"()[]") for w in words]
    total = max(len(clean_words), 1)

    german_matches = sum(1 for w in clean_words if w in GERMAN_FUNCTION_WORDS)
    turkish_matches = sum(1 for w in clean_words if w in TURKISH_FUNCTION_WORDS)
    french_matches = sum(1 for w in clean_words if w in FRENCH_FUNCTION_WORDS)

    # Function-word density scaled to produce score >= 1.0 for genuine
    # non-English text.  French shares some words with English ("le", "la"
    # are not English but "les", "des" are distinctly French) so we use a
    # slightly lower multiplier and rely on multiple matches.
    score = max(
        (german_matches / total) * 15.0,
        (turkish_matches / total) * 20.0,
        (french_matches / total) * 12.0,
    )
    return score


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class PublicationParser:
    """Extract transliteration-translation pairs from OCR'd publication pages."""

    # Minimum scores to accept a candidate split
    MIN_AKKADIAN_SCORE = 0.3
    MIN_ENGLISH_SCORE = 0.15

    # Maximum English score allowed on the *before* side of a split.
    # Commentary / analysis pages have English prose with embedded Akkadian
    # proper nouns; both sides score as their respective languages but the
    # before-text is genuinely English, not transliteration.  Rejecting
    # candidates where the before-window reads as English eliminates these.
    MAX_BEFORE_ENGLISH = 0.3

    # Maximum non-English score allowed on the *after* side.
    # Many AKT / APU volumes publish German or Turkish translations; we only
    # want English.
    MAX_AFTER_NON_ENGLISH = 1.0

    # Minimum character lengths for accepted transliteration / translation
    MIN_TRANSLITERATION_LEN = 20
    MIN_TRANSLATION_LEN = 15

    # Scoring window: used to determine *whether* a split is valid.
    # Kept narrow so that English text on the wrong side of a split
    # doesn't inflate the Akkadian score.
    _SCORE_BEFORE = 200  # chars
    _SCORE_AFTER = 300   # chars

    # Extraction window: once the best split is chosen, how much text
    # to grab for the actual transliteration / translation output.
    _EXTRACT_BEFORE = 800   # chars
    _EXTRACT_AFTER = 2000   # chars

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract_pairs(self, page_text: str) -> List[Dict[str, str]]:
        """
        Extract all transliteration-translation pairs from a single page.

        Args:
            page_text: Cleaned OCR text from one page

        Returns:
            List of dicts, each with 'transliteration' and 'translation' keys.
        """
        if not page_text or not isinstance(page_text, str):
            return []

        segments = self._split_into_segments(page_text)
        pairs = []
        for segment in segments:
            pair = self._extract_from_segment(segment)
            if pair:
                pairs.append(pair)
        return pairs

    def parse_publication(self, pub_df: pd.DataFrame) -> List[Dict[str, str]]:
        """
        Parse all Akkadian-flagged pages of a single publication.

        Args:
            pub_df: DataFrame for one pdf_name, already filtered to
                    has_akkadian == True and sorted by page number.

        Returns:
            List of extracted pairs, each augmented with 'source_pdf' and
            'source_page' metadata.
        """
        if pub_df.empty:
            return []

        pdf_name = str(pub_df["pdf_name"].iloc[0])
        all_pairs: List[Dict[str, str]] = []

        for _, row in pub_df.iterrows():
            page_pairs = self.extract_pairs(str(row["page_text"]))
            for pair in page_pairs:
                pair["source_pdf"] = pdf_name
                pair["source_page"] = int(row["page"])
            all_pairs.extend(page_pairs)

        return all_pairs

    # ------------------------------------------------------------------
    # Segment splitting
    # ------------------------------------------------------------------

    def _split_into_segments(self, text: str) -> List[str]:
        """
        Split page text into segments by "Text N." headers.

        If no headers are found the entire page is returned as a single
        segment.  Any text preceding the first header is included as its
        own segment (it may contain a standalone pair).
        """
        split_positions = [m.start() for m in TEXT_HEADER_RE.finditer(text)]

        if not split_positions:
            return [text]

        segments: List[str] = []

        # Prefix before the first header
        if split_positions[0] > 0:
            prefix = text[: split_positions[0]].strip()
            if prefix:
                segments.append(prefix)

        # Each header-delimited segment
        for i, start in enumerate(split_positions):
            end = split_positions[i + 1] if i + 1 < len(split_positions) else len(text)
            segments.append(text[start:end])

        return segments

    # ------------------------------------------------------------------
    # Pair extraction from a single segment
    # ------------------------------------------------------------------

    # Post-extraction quality thresholds applied to the *full* extracted text
    # (as opposed to the narrow scoring window used during split detection).
    # The scoring window can miss contamination that spans outside its range;
    # these checks catch it.
    _MAX_FULL_TRANSLIT_ENGLISH = 0.2   # transliteration must not read as English
    _MIN_FULL_TRANSL_ENGLISH = 0.4     # translation must be substantially English
    _MAX_FULL_TRANSL_NON_ENGLISH = 0.5 # translation must not be German/Turkish/French
    _MIN_FULL_TEXT_LEN = 100           # both sides must have >= 100 chars of content

    def _extract_from_segment(self, segment: str) -> Optional[Dict[str, str]]:
        """
        Find the best transliteration → translation split in a segment.

        Scans every `:` or `.` followed by whitespace as a candidate split
        point.  Scores the text before (Akkadian?) and after (English?).
        Returns the pair with the highest combined score, or None if no
        candidate meets the thresholds.

        After cleaning, a second quality pass validates the *full* extracted
        texts to reject commentary pages and non-English translations that
        slipped through the narrow scoring window.
        """
        result = self._find_best_split(segment)
        if not result:
            return None

        raw_translit, raw_transl = result

        transliteration = self._clean_transliteration(raw_translit)
        translation = self._clean_translation(raw_transl)

        if not transliteration or not translation:
            return None

        # --- Post-extraction quality validation ---
        # Both sides must have meaningful length
        if (
            len(transliteration) < self._MIN_FULL_TEXT_LEN
            or len(translation) < self._MIN_FULL_TEXT_LEN
        ):
            return None

        # Transliteration must actually contain Akkadian
        if akkadian_score(transliteration) < 1.0:
            return None

        # Commentary pages: transliteration side reads as English prose
        if english_score(transliteration) > self._MAX_FULL_TRANSLIT_ENGLISH:
            return None

        # Transliteration must not be French / German / Turkish
        if non_english_score(transliteration) > self._MAX_FULL_TRANSL_NON_ENGLISH:
            return None

        # Translation must be substantially English overall
        if english_score(translation) < self._MIN_FULL_TRANSL_ENGLISH:
            return None

        # Translation must START with English (not another Akkadian block
        # followed by English commentary later in the window)
        if english_score(translation[:300]) < 0.15:
            return None

        # Translation must not be German / Turkish / French
        if non_english_score(translation) > self._MAX_FULL_TRANSL_NON_ENGLISH:
            return None

        return {
            "transliteration": transliteration,
            "translation": translation,
        }

    def _find_best_split(self, text: str) -> Optional[Tuple[str, str]]:
        """
        Scan candidate split points and return the (before, after) pair
        with the highest combined Akkadian × English score.

        Candidate split points are positions immediately after a `:` or `.`
        that is followed by a space or newline.  We skip colons that are
        part of citation ranges (e.g., "BIN 4 61:34") by requiring at least
        one whitespace character after the colon.

        In tablet editions the structure is always transliteration first,
        then translation.  The correct boundary is therefore the *first*
        (leftmost) split point where the text before scores as Akkadian
        and the text after scores as English — any later split that also
        passes the thresholds is necessarily inside the English translation.

        Scoring uses a narrow window around the candidate so that text
        on the far side of a later split does not inflate scores.  Once
        the split is found, a wider extraction window captures the full
        transliteration and translation.
        """
        # Regex: colon or period followed by at least one space/newline
        candidates = [
            m.end() for m in re.finditer(r"[:\.][ \t\n]+", text)
        ]

        if not candidates:
            return None

        for pos in candidates:
            # --- Narrow window for scoring accuracy ---
            score_before = text[max(0, pos - self._SCORE_BEFORE) : pos].strip()
            score_after = text[pos : pos + self._SCORE_AFTER].strip()

            if (
                len(score_before) < self.MIN_TRANSLITERATION_LEN
                or len(score_after) < self.MIN_TRANSLATION_LEN
            ):
                continue

            akk = akkadian_score(score_before)
            eng = english_score(score_after)

            if akk < self.MIN_AKKADIAN_SCORE or eng < self.MIN_ENGLISH_SCORE:
                continue

            # --- Quality gate: reject commentary pages ---
            # Commentary / analysis pages have English prose with embedded
            # Akkadian proper nouns.  The before-window scores as Akkadian
            # (diacritics in names) but is fundamentally English.  A high
            # english_score on the before-side disqualifies the candidate.
            eng_before = english_score(score_before)
            if eng_before > self.MAX_BEFORE_ENGLISH:
                continue

            # --- Quality gate: reject non-English translations ---
            # Many AKT / APU volumes publish German or Turkish translations.
            # Detect and skip these so the output corpus is English-only.
            non_eng = non_english_score(score_after)
            if non_eng > self.MAX_AFTER_NON_ENGLISH:
                continue

            # First valid split wins — this IS the transliteration boundary
            # Wide window for extraction completeness
            before = text[max(0, pos - self._EXTRACT_BEFORE) : pos].strip()
            after = text[pos : pos + self._EXTRACT_AFTER].strip()
            return (before, after)

        return None

    # ------------------------------------------------------------------
    # Cleaning helpers
    # ------------------------------------------------------------------

    def _clean_transliteration(self, text: str) -> str:
        """
        Strip non-transliteration cruft from the extracted block:
        - "Text N. Citation" headers
        - Citation references at the start (e.g., "BIN 4 61:34–69 (...)")
        - Labels like "(Tablet)", "(Envelope)", "(Composite Text)"
        - Normalise whitespace
        """
        # Remove "Text N." headers
        text = TEXT_HEADER_RE.sub("", text)

        # Remove citation / reference lines at the start if they are NOT
        # Akkadian.  Academic editions typically begin each text block with
        # a citation like "BIN 4 61:34–69 (see further, Appendix 4)."
        # A line is kept only if it contains Akkadian diacritics — this
        # prevents all-caps catalog abbreviations (e.g., "AKT") from being
        # mistaken for Sumerograms and blocking citation removal.
        lines = text.split("\n")
        while lines and lines[0].strip():
            first = lines[0].strip()
            has_diacritics = any(c in AKKADIAN_DIACRITICS for c in first)
            if not has_diacritics:
                lines.pop(0)
            else:
                break
        text = "\n".join(lines)

        # Remove labels
        text = re.sub(r"\((Tablet|Envelope|Composite Text)\)\s*", "", text)

        # Normalise whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text if len(text) >= self.MIN_TRANSLITERATION_LEN else ""

    def _clean_translation(self, text: str) -> str:
        """
        Truncate and clean the extracted translation block:
        - Stop at the next "Text N." header (next tablet edition)
        - Stop at any line that scores as a new transliteration block
        - Strip trailing catalog references (e.g., "REL 84 III.")
        - Normalise whitespace
        """
        # Truncate at next "Text N." header
        match = TEXT_HEADER_RE.search(text)
        if match:
            text = text[: match.start()]

        # Truncate at any line that looks like a new transliteration block.
        # A genuine transliteration line has high Akkadian score AND low
        # English score.  Lines like "From the week of Aššur-imittī, REL 105
        # II" contain embedded proper nouns (diacritics) and catalog refs
        # (all-caps) but are clearly English prose — english_score catches
        # this.
        lines = text.split("\n")
        kept: List[str] = []
        for line in lines:
            stripped = line.strip()
            if (
                len(stripped) > 25
                and any(c in AKKADIAN_DIACRITICS for c in stripped)
                and akkadian_score(stripped) > 1.0
                and english_score(stripped) < 0.3
            ):
                break
            kept.append(line)
        text = "\n".join(kept)

        # Strip trailing catalog references like "REL 84 III." or "REL 82 XI."
        text = re.sub(r"\s*REL\s+\d+\s+[IVX]+\.?\s*$", "", text)

        # Normalise whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text if len(text) >= self.MIN_TRANSLATION_LEN else ""
