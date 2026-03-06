"""
Pre-translation pipeline for Old Assyrian Akkadian.

Applies deterministic rule-based translations before the neural model,
reducing the burden on the model for elements that have fixed mappings:
  - Determinative stripping ({d}, {ki}, {URU}, etc.)
  - Letter formula translation (um-ma X a-na Y qí-bi-ma → English template)
  - Proper noun resolution (lexicon exact + fuzzy match, patronymics)
  - Sumerogram translation (ANNA → tin, DUMU → son, etc.)
  - Number parsing (4 GÚ 20 ma-na ANNA → 4 talents 20 minas of tin)

Templates match the most common gold translation convention:
  Pattern A (um-ma X … a-na Y qí-bi-ma)  → "From X to Y:"
  Pattern B (a-na Y qí-bi-ma um-ma X-ma) → "To Y from X:"

Usage:
    from src.preprocessing.pretranslator import PreTranslator
    from src.retrieval.lexicon import Lexicon

    lexicon = Lexicon()
    lexicon.load()
    pt = PreTranslator(lexicon=lexicon)

    result = pt.pre_translate("a-na ku-li-a qí-bi₄-ma um-ma a-šùr-i-mì-tí-ma 4 GÚ 20 ma-na ANNA")
    # result["scaffold"]   → "To Kuliya from Aššur-imittī: 4 talents 20 minas of tin"
    # result["remaining"]  → ""
"""

import re
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Sexagesimal fraction detection
#
# Old Assyrian tablets record fractional weights by removing the decimal point
# from the floating-point expansion:
#   42.3333... minas  →  4233333 ma-na
#    0.6666... minas  →  066666  ma-na
#    1.8333... minas  →  1833333 ma-na   (float noise: 18333300000000001)
#    0.5       minas  →  05      ma-na
#   23.5       GÚ    →  235     GÚ
#
# Fraction map (leading digit of the decimal expansion → Unicode fraction):
#   ⅙ = 0.1666...  prefix 1 + repeating 6s
#   ⅓ = 0.3333...  repeating 3s
#   ⅔ = 0.6666...  repeating 6s
#   ⅚ = 0.8333...  prefix 8 + repeating 3s
#   ½ = 0.5        single 5 (only when number has ≥ 2 digits)
# ---------------------------------------------------------------------------

# Matches recurring decimal fraction suffixes including Python float noise
# (trailing zeros + optional off-by-one digit from float repr).
_FRAC_RE = re.compile(
    r'^(\d*?)'                         # minimal integer prefix
    r'(83{4,}[30]*[01]?'              # ⅚: 83333...
    r'|6{5,}[30]*[01]?'               # ⅔: 66666...
    r'|3{5,}[30]*[01]?'               # ⅓: 33333...
    r'|16{4,}[30]*[01]?'              # ⅙: 16666...
    r')$'
)
_FRAC_CHAR = {'8': '⅚', '6': '⅔', '3': '⅓', '1': '⅙'}


def _parse_number(num_str):
    """
    Convert a decimal-encoded OA number to readable mixed-number notation.

    Returns the original string unchanged when no fraction pattern is detected.

    Examples:
        "4233333" → "42⅓"
        "066666"  → "⅔"
        "05"      → "½"
        "235"     → "23½"
        "5"       → "5"   (single digit: not a fraction)
        "42"      → "42"
    """
    m = _FRAC_RE.match(num_str)
    if m:
        int_part = m.group(1)
        frac_ch = _FRAC_CHAR[m.group(2)[0]]
        n = int(int_part) if int_part else 0
        return frac_ch if n == 0 else "{}{}".format(n, frac_ch)

    # Half-fraction: ≥2-digit number ending in 5
    if len(num_str) >= 2 and num_str.endswith('5'):
        stripped = num_str[:-1].lstrip('0') or '0'
        n = int(stripped)
        return '½' if n == 0 else "{}½".format(n)

    return num_str

# ---------------------------------------------------------------------------
# Determinative handling
# ---------------------------------------------------------------------------

_DET_RE = re.compile(r'\{([^}]+)\}')

_DET_TYPE = {
    "d": "deity",
    "f": "female",
    "m": "male",
    "ki": "place",
    "URU": "city",
}

# ---------------------------------------------------------------------------
# Weight units: can be followed by a commodity Sumerogram ("N units of X")
# ---------------------------------------------------------------------------

_WEIGHT_UNIT_SINGULAR = {
    "GÚ": "talent",
    "GÍN": "shekel",
    "GIN": "shekel",
    "ma-na": "mina",
    "MA.NA": "mina",
    "MANA": "mina",
    "me-at": "hundred",   # "hundred" doesn't pluralise to "hundreds" here
}

# Commodities that follow a weight unit with an "of" connector
_COMMODITIES = {
    "KÙBABBAR": "silver",
    "KÙ.BABBAR": "silver",
    "ANNA": "tin",
    "KÙ.GI": "gold",
    "URUDU": "copper",
}

# Regex: weight number + unit + optional commodity
# Longer commodity spellings first to avoid partial matches.
_COMM_ALT = "KÙ\\.BABBAR|KÙBABBAR|KÙ\\.GI|URUDU|ANNA"
_WEIGHT_ALT = "MA\\.NA|MANA|GÍN|GIN|GÚ|ma-na|me-at"
_NUM_WEIGHT_RE = re.compile(
    # (?<![\w-]) prevents matching digits inside hyphenated compounds (e.g. x-6 GÍN
    # should not match at the 6).
    # \b after the unit prevents matching GÍN inside GÍNTA.
    # (?!-) after unit and commodity prevents splitting hyphenated forms like
    # ma-na-im (genitive suffix) or URUDU-a-kà (commodity as compound component).
    rf'(?<![\w-])(\d+)\s+({_WEIGHT_ALT})\b(?!-)(?:\s+({_COMM_ALT})(?!-))?',
    re.UNICODE,
)

# ---------------------------------------------------------------------------
# Count units: standalone items counted by number (textiles, etc.)
# ---------------------------------------------------------------------------

_COUNT_UNIT_SINGULAR = {
    "TÚG": "textile",
    "TUG": "textile",
}

_NUM_COUNT_RE = re.compile(
    r'(?<![\w-])(\d+)\s+(TÚG|TUG)',
    re.UNICODE,
)

# ---------------------------------------------------------------------------
# Sumerogram overrides (standalone, not preceded by a digit)
# ---------------------------------------------------------------------------

_SUMEROGRAM_EN = {
    "KÙBABBAR": "silver",
    "KÙ.BABBAR": "silver",
    "ANNA": "tin",
    "TÚG": "textiles",
    "GÚ": "talents",
    "GÍN": "shekels",
    "GIN": "shekels",
    "DUMU": "son",
    "DUMU.MUNUS": "daughter",
    "KÙ.GI": "gold",
    "IGI": "before",
    "MANA": "minas",
    "MA.NA": "minas",
    "URU": "city",
    "URUDU": "copper",
}

# Sumerogram token pattern: 2+ uppercase chars, optional subscript digits,
# optional dot-separated segments.
_UC = "A-ZĀĒĪŪṢṬŠḪÁÀÉÈÍÌÚÙÂÊÎÛÄÏÜ"
_SG_RE = re.compile(rf'\b[{_UC}]{{2,}}[₀-₉]*(?:\.[{_UC}]+[₀-₉]*)*\b')

# ---------------------------------------------------------------------------
# Letter formula patterns (same as genre.py, kept here for m.end() access)
# ---------------------------------------------------------------------------

_FORMULA_A = re.compile(
    r'um-ma\s+(.{3,50}?)\s+a-na\s+(.{3,50}?)\s+qí-bi[₄]?-ma'
)
_FORMULA_B = re.compile(
    r'a-na\s+(.{3,50}?)\s+qí-bi[₄]?-ma\s+um-ma\s+(.{3,50}?)-ma'
)

# Old Assyrian conjunction used in multi-name expressions
_OA_AND = re.compile(r'\s+ù\s+|\s+u\s+')


# ---------------------------------------------------------------------------
# PreTranslator class
# ---------------------------------------------------------------------------

class PreTranslator:
    """
    Pre-translates deterministic elements from Akkadian transliteration.

    Returns a scaffold (pre-translated English) and the remaining Akkadian
    that the neural model still needs to handle.
    """

    def __init__(self, lexicon=None):
        """
        Args:
            lexicon: Lexicon instance (optional). Enables PN fuzzy matching
                     and extended Sumerogram lookup beyond the hardcoded list.
        """
        self.lexicon = lexicon

    def pre_translate(self, transliteration: str) -> Dict:
        """
        Pre-translate deterministic elements from Akkadian transliteration.

        Returns dict with:
            scaffold: pre-translated English components (ordered, space-joined)
            remaining: untranslated Akkadian for the neural model
            formula: extracted letter formula metadata or None
        """
        text = transliteration

        # 1. Strip determinatives; collect semantic hints for name resolution
        text, det_hints = _strip_determinatives(text)

        # 2. Extract letter formula and produce English template.
        # Only search the opening portion of the text — letter formulas always
        # appear at the start.  Searching the full text can match quoted
        # sub-letters in the body ("letter within a letter" phenomenon).
        formula_en = None
        formula_info = None
        head = text[:300]

        # Try Pattern B first: it anchors on "a-na" which must appear near the
        # start of the text.  This prevents a "um-ma X a-na Y qí-bi-ma"
        # sub-clause in the letter body from being misread as Pattern A.
        m = _FORMULA_B.search(head)
        if m:
            recip_en = self._resolve_name(m.group(1).strip(), det_hints)
            sender_en = self._resolve_name(m.group(2).strip(), det_hints)
            # Most common gold convention for Pattern B
            formula_en = f"To {recip_en} from {sender_en}:"
            formula_info = {
                "pattern": "B",
                "sender": m.group(2).strip(),
                "recipient": m.group(1).strip(),
            }
            text = text[m.end():].strip()
        else:
            m = _FORMULA_A.search(head)
            if m:
                sender_raw = m.group(1).strip()
                # Pattern A captures sender before "a-na"; in OA the sender clause
                # ends with the enclitic -ma ("um-ma PN-ma a-na Y qí-bi-ma").
                # Strip it so name resolution works against the clean base form.
                if sender_raw.endswith("-ma"):
                    sender_raw = sender_raw[:-3]
                sender_en = self._resolve_name(sender_raw, det_hints)
                recip_en = self._resolve_name(m.group(2).strip(), det_hints)
                # Most common gold convention for Pattern A
                formula_en = f"From {sender_en} to {recip_en}:"
                formula_info = {
                    "pattern": "A",
                    "sender": sender_raw,
                    "recipient": m.group(2).strip(),
                }
                text = text[m.end():].strip()

        # 3. Process body text into ordered translated / remaining segments
        segments = self._segment_body(text)

        # 4. Assemble scaffold and remaining
        scaffold_parts = []
        if formula_en:
            scaffold_parts.append(formula_en)

        remaining_parts = []
        for seg_type, content in segments:
            if seg_type == "translated":
                scaffold_parts.append(content)
            else:
                remaining_parts.append(content)

        return {
            "scaffold": " ".join(scaffold_parts),
            "remaining": " ".join(remaining_parts),
            "formula": formula_info,
        }

    # ------------------------------------------------------------------
    # Name resolution
    # ------------------------------------------------------------------

    def _resolve_name(self, raw: str, det_hints: Dict) -> str:
        """
        Resolve a name to its English normalized form.

        Handles:
          - Patronymics: "X DUMU Y" → "X son of Y"
          - Multi-name expressions with OA conjunction ù: "X ù Y" → "X and Y"
          - Single name: lexicon lookup then mechanical normalization
        """
        # Multi-name: split on OA conjunction ù / u
        if re.search(r'\s+ù\s+', raw):
            parts = re.split(r'\s+ù\s+', raw)
            resolved = [self._resolve_single_or_patronymic(p.strip(), det_hints)
                        for p in parts]
            return " and ".join(resolved)

        return self._resolve_single_or_patronymic(raw, det_hints)

    def _resolve_single_or_patronymic(self, raw: str, det_hints: Dict) -> str:
        """Handle 'X DUMU Y' patronymics, then delegate to single-name lookup."""
        pat = re.match(r'^(.+?)\s+DUMU\s+(.+)$', raw)
        if pat:
            person = self._resolve_single_name(pat.group(1).strip(), det_hints)
            father = self._resolve_single_name(pat.group(2).strip(), det_hints)
            return f"{person} son of {father}"
        return self._resolve_single_name(raw, det_hints)

    def _resolve_single_name(self, raw: str, det_hints: Dict) -> str:
        """
        Resolve a single name token via lexicon or mechanical normalization.
        """
        if self.lexicon is not None:
            result = self.lexicon.lookup_proper_noun(raw, fuzzy=True)
            if result:
                return result["norm"]
        return _mechanical_normalize(raw)

    # ------------------------------------------------------------------
    # Body text segmentation
    # ------------------------------------------------------------------

    def _segment_body(self, text: str) -> List[Tuple[str, str]]:
        """
        Process body text into ordered (type, content) segments.

        Returns list of ("translated", english) or ("remaining", akkadian) tuples.
        """
        replacements = []  # (start, end, english_text)

        # Pass 1: weight number + unit (+ optional commodity)
        for m in _NUM_WEIGHT_RE.finditer(text):
            english = _format_weight(m.group(1), m.group(2), m.group(3))
            replacements.append((m.start(), m.end(), english))

        # Pass 2: count number + unit (TÚG etc.)
        covered = set()
        for start, end, _ in replacements:
            covered.update(range(start, end))

        for m in _NUM_COUNT_RE.finditer(text):
            if any(i in covered for i in range(m.start(), m.end())):
                continue
            english = _format_count(m.group(1), m.group(2))
            replacements.append((m.start(), m.end(), english))
            covered.update(range(m.start(), m.end()))

        # Pass 3: standalone Sumerograms (not already covered)
        for m in _SG_RE.finditer(text):
            if any(i in covered for i in range(m.start(), m.end())):
                continue
            # Skip Sumerograms embedded in hyphenated compounds, immediately
            # following a closing paren (determinative-prefix notation like (d)ENLÍL),
            # or immediately following an opening paren (type-indicator notation like
            # (TÚG)ku-ta-ni where (TÚG) classifies the following Akkadian word).
            # Extracting them would split the compound and produce orphaned fragments
            # like "-a-šur" (from PUZUR₄-a-šur), "i-tur₄-" (from i-tur₄-DINGIR),
            # or "(" / ")ku-ta-ni" (from (TÚG)ku-ta-ni).
            pre = text[m.start() - 1] if m.start() > 0 else ''
            post = text[m.end()] if m.end() < len(text) else ''
            if pre in ('-', ')', '(') or post in ('-', ')'):
                continue
            sg = m.group(0)
            en = _SUMEROGRAM_EN.get(sg)
            if en is None and self.lexicon is not None:
                en = self.lexicon.lookup_sumerogram(sg)
            if en:
                replacements.append((m.start(), m.end(), en))
                covered.update(range(m.start(), m.end()))

        replacements.sort(key=lambda x: x[0])

        # Build ordered segments
        segments = []
        pos = 0
        for start, end, english in replacements:
            if start > pos:
                chunk = text[pos:start].strip()
                if chunk:
                    segments.append(("remaining", chunk))
            segments.append(("translated", english))
            pos = end

        if pos < len(text):
            tail = text[pos:].strip()
            if tail:
                segments.append(("remaining", tail))

        return segments


# ---------------------------------------------------------------------------
# Number formatting helpers
# ---------------------------------------------------------------------------

def _pluralise(singular: str, num_str: str) -> str:
    """
    Return singular or plural form based on the number string.

    "hundred" never pluralises (2 me-at = "2 hundred" not "2 hundreds").
    """
    if singular == "hundred":
        return "hundred"
    try:
        n = int(num_str)
    except ValueError:
        return singular + "s"
    return singular if n == 1 else singular + "s"


def _format_weight(num_str: str, unit_key: str, commodity_key: str) -> str:
    """Format a weight number + unit (+ optional commodity)."""
    singular = _WEIGHT_UNIT_SINGULAR.get(unit_key, unit_key.lower())
    unit_en = _pluralise(singular, num_str)
    result = f"{num_str} {unit_en}"
    if commodity_key:
        comm_en = _COMMODITIES.get(commodity_key, commodity_key.lower())
        result += f" of {comm_en}"
    return result


def _format_count(num_str: str, unit_key: str) -> str:
    """Format a count number + unit (TÚG, etc.)."""
    singular = _COUNT_UNIT_SINGULAR.get(unit_key, unit_key.lower())
    unit_en = _pluralise(singular, num_str)
    return f"{num_str} {unit_en}"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _strip_determinatives(text: str) -> Tuple[str, Dict]:
    """
    Remove determinative markers ({d}, {ki}, {m}, {f}, {URU}) from text.

    Returns (cleaned_text, det_hints) where det_hints maps each word
    immediately following a determinative to its semantic category.
    """
    hints = {}
    for m in _DET_RE.finditer(text):
        cat = _DET_TYPE.get(m.group(1), "")
        if cat:
            after = text[m.end():].lstrip()
            nw = re.match(r'\S+', after)
            if nw:
                hints[nw.group(0)] = cat

    cleaned = _DET_RE.sub("", text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned, hints


def _mechanical_normalize(name: str) -> str:
    """
    Fallback name normalization: remove hyphens, title-case each word.

    Handles multi-syllable OA names like "a-lá-ḫi-im" → "Aláḫiim".
    Not accurate without the lexicon, but avoids run-together all-lowercase.
    """
    return name.replace("-", "").capitalize()
