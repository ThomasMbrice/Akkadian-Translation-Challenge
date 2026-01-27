# Data Formats and Preprocessing Rules

## Competition Data Files

### train.csv
~1,500 transliterations with English translations (document-level alignment)

**Fields:**
- `oare_id`: Unique identifier in OARE database
- `transliteration`: Old Assyrian transliteration
- `translation`: English translation

### test.csv
~4,000 sentences from ~400 documents (sentence-level alignment)

**Fields:**
- `id`: Unique sentence identifier
- `text_id`: Document identifier (groups sentences)
- `line_start`, `line_end`: Sentence boundaries on tablet (str type: '1', '1'', '1''')
- `transliteration`: Old Assyrian transliteration (target for translation)

**Important:** Test data has **sentence-level** alignment vs train's **document-level** alignment

### published_texts.csv
~8,000 transliterations without translations

**Key fields:**
- `oare_id`: OARE identifier
- `transliteration`: Clean transliteration
- `transliteration_orig`: Original from OARE
- `cdli_id`, `aliases`, `label`: Cross-reference identifiers
- `online_transcript`, `online_catalog`, `online_information`: URLs for additional resources
- `AICC_translation`: URL to (poor quality) machine translations
- `note`, `interlinear_commentary`: Specialist notes

**Use for:** Additional training data via alignment with publications

### publications.csv
~880 scholarly publications (OCR text)

**Fields:**
- `pdf_name`: Publication identifier (links to bibliography.csv)
- `page`: Page number
- `page_text`: OCR-extracted text
- `has_akkadian`: Boolean flag for Akkadian presence

**Critical for:** Extracting aligned translations (often in non-English languages)

### OA_Lexicon_eBL.csv
Old Assyrian word forms → dictionary lemmas

**Fields:**
- `type`: word | PN (person name) | GN (geographic name)
- `form`: Literal word in transliteration
- `norm`: Normalized form (no hyphens, vowel length indicators)
- `lexeme`: Dictionary lemma
- `eBL`: URL to electronic Babylonian Library entry
- `I_IV`, `A_D`: Homonym designations (CDA, CAD)

### eBL_Dictionary.csv
Complete Akkadian dictionary from eBL

**Use for:** Lexicon lookup component of retrieval pipeline

### Sentences_Oare_FirstWord_LinNum.csv
Sentence alignment aid for train.csv

**Use for:** Converting document-level → sentence-level alignments

---

## Preprocessing Rules (Competition Specification)

### Characters to REMOVE
- `!` - Certainty marker
- `?` - Uncertainty marker
- `/` - Gloss separator
- `:` - Colon (various uses)
- `.` - Period
- `˹` `˺` - Partial damage markers
- `<<...>>` - Erasure markers (remove markers, remove content)

### Characters/Patterns to PRESERVE
- `{determinatives}` - Semantic classifiers (e.g., `{d}` = divine, `{f}` = female)
- `SUMEROGRAMS` - All-caps logograms (e.g., `DUMU` = son, `KÙ.BABBAR` = silver)
- Capitalized words - Proper nouns (person/place names)
- `[...]` - Damaged text (keep content inside brackets)
- `<...>` - Scribal corrections (keep content inside)

### Gap Normalization
- `[x]` - Single missing sign → `<gap>`
- `[x x]`, `[...]`, `[n lines broken]` - Multiple missing signs → `<big_gap>`

### Line Numbers (test.csv)
Format: `'1'`, `'1''`, `'1'''` (string type)
- Primes indicate obverse/reverse/edge conventions
- Order sentences within document using these

---

## Data Extraction Workflow

### Phase 1: Publication → Translation Pairs
1. **Locate:** Match document IDs (oare_id, aliases) in publications.csv
2. **Extract:** Use LLM to extract parallel Akkadian-Translation pairs from OCR text
3. **Translate:** Convert non-English translations → English (LLM)
4. **Align:** Split into sentence-level pairs
5. **Deduplicate:** Same tablet may appear in multiple publications

### Phase 2: Quality Filtering
- **Keep:** Literal translations (word-for-word)
- **Remove:** Paraphrases, summaries, scholarly commentary
- **Validate:** Proper nouns, Sumerograms match lexicons

### Phase 3: Augmentation
- **Synthetic gaps:** Randomly mask spans with `<gap>` / `<big_gap>`
- **Back-translation:** English → Akkadian → English (if model available)

---

## Target Corpus Statistics

| Source | Raw Count | After Processing | Notes |
|--------|-----------|------------------|-------|
| train.csv | 1,500 docs | ~10k sentences | Sentence-split via alignment aid |
| publications.csv | 900 PDFs | 10-30k pairs | Depends on extraction quality |
| published_texts.csv | 8,000 texts | 0 (no translations) | May align with publications |

**Target:** 20-50k parallel sentence pairs (2-3x provided training data)

---

## Example Processing

### Raw Input
```
a-na A-šùr-i-mì-tí / DUMU Ṣí-lí-{d}UTU [x x] qí-bi-ma!
```

### After Preprocessing
```
a-na A-šùr-i-mì-tí DUMU Ṣí-lí-{d}UTU <big_gap> qí-bi-ma
```

### Extracted Context
```
Proper nouns: A-šùr-i-mì-tí (Aššur-imitti), Ṣí-lí-{d}UTU (Silli-Shamash)
Sumerograms: DUMU → "son"
Determinatives: {d} → divine name marker
```

---

## File Organization

```
data/
├── raw/                      # Competition data (as downloaded)
│   ├── train.csv
│   ├── test.csv
│   ├── published_texts.csv
│   ├── publications.csv
│   ├── bibliography.csv
│   ├── OA_Lexicon_eBL.csv
│   ├── eBL_Dictionary.csv
│   └── Sentences_Oare_FirstWord_LinNum.csv
│
├── processed/                # Cleaned, aligned data
│   ├── train_sentences.csv  # Sentence-split train.csv
│   ├── extracted_pairs.csv  # From publications
│   └── combined_corpus.csv  # Final training corpus
│
├── augmented/                # Synthetic data
│   ├── with_gaps.csv        # Gap-augmented pairs
│   └── back_translated.csv  # Back-translation pairs
│
└── indices/                  # Retrieval artifacts
    ├── faiss_index.bin      # FAISS similarity index
    ├── proper_nouns.json    # Extracted proper noun dictionary
    └── sumerograms.json     # Sumerogram → English mapping
```

---

## Data Quality Checklist

Before training:
- [ ] All prohibited characters removed
- [ ] Gaps normalized to `<gap>` / `<big_gap>`
- [ ] Proper nouns preserved (capitalization intact)
- [ ] Sumerograms preserved (all-caps intact)
- [ ] Determinatives preserved (`{...}` intact)
- [ ] Sentence-level alignment verified
- [ ] Duplicate tablets removed
- [ ] Non-English translations converted to English
- [ ] Paraphrases filtered out

---

*See ARCHITECTURE.md for how this data flows through the preprocessing → retrieval → generation pipeline.*
