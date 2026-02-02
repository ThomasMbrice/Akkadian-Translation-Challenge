# Akkadian NMT Project: Architecture Summary

## Problem
Build neural machine translation from transliterated Old Assyrian cuneiform → English. Constraints: ~8,000 texts provided, ~50% with translations, 900 unprocessed scholarly publications available for extraction. Old Assyrian is a low-resource, morphologically complex Semitic language (single words encode full English clauses). No native speakers exist for evaluation.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT PIPELINE                           │
├─────────────────────────────────────────────────────────────┤
│  Raw Transliteration                                        │
│  "a-na A-šùr-i-mì-tí DUMU Ṣí-lí-{d}UTU [x] qí-bi-ma"       │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  PREPROCESSOR                                        │   │
│  │  - Strip: ! ? / : . ˹˺ << >>                        │   │
│  │  - Normalize: [...] → content, gaps → <gap>/<big_gap>│   │
│  │  - Preserve: {determinatives}, SUMEROGRAMS, Names   │   │
│  │  - Extract: proper nouns (Capitalized), logograms   │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│            ┌──────────────┴──────────────┐                 │
│            ▼                              ▼                 │
│  ┌──────────────────┐          ┌──────────────────────┐    │
│  │ TRANSLATION      │          │ LEXICON LOOKUP       │    │
│  │ MEMORY (RAG)     │          │                      │    │
│  │                  │          │ - Proper noun dict   │    │
│  │ - Embed English  │          │ - Sumerogram dict    │    │
│  │   side of corpus │          │   (KÙ.BABBAR=silver) │    │
│  │ - FAISS index    │          │ - Determinative map  │    │
│  │ - Retrieve k=5   │          │ - Fuzzy matching     │    │
│  │   similar pairs  │          │                      │    │
│  └────────┬─────────┘          └──────────┬───────────┘    │
│           │                               │                 │
│           └───────────────┬───────────────┘                 │
│                           ▼                                 │
├─────────────────────────────────────────────────────────────┤
│                    GENERATION                               │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐   │
│  │  CONTEXT ASSEMBLY                                    │   │
│  │                                                      │   │
│  │  Lexicon:                                           │   │
│  │  - A-šùr-i-mì-tí = Aššur-imitti (personal name)    │   │
│  │  - DUMU = son                                       │   │
│  │  - {d}UTU = divine determinative + Shamash         │   │
│  │                                                      │   │
│  │  Similar translations:                              │   │
│  │  [Retrieved example 1: Akkadian → English]          │   │
│  │  [Retrieved example 2: Akkadian → English]          │   │
│  │  [Retrieved example 3: Akkadian → English]          │   │
│  │                                                      │   │
│  │  Translate: [cleaned input]                         │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  FINE-TUNED MODEL                                    │   │
│  │                                                      │   │
│  │  Base: ByT5 (byte-level, no tokenization issues)    │   │
│  │  Training: ~20-50k parallel pairs                   │   │
│  │  Augmentation: back-translation, synthetic gaps     │   │
│  │  Multi-task: translation + lemmatization + POS      │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│                    English Output                           │
└─────────────────────────────────────────────────────────────┘
```

## Key Technical Decisions

| Decision                         | Choice                    | Rationale |
|----------------------------------|---------------------------|-----------|
| Tokenization                     | Byte-level (ByT5)         | Akkadian has hyphens, diacritics, mixed scripts—BPE mangles it |
| Retrieval embedding              | Embed English side        | No pretrained Akkadian embedders exist |
| RAG integration                  | Context stuffing          | Simpler than Fusion-in-Decoder, sufficient for formulaic texts |
| Data augmentation                | Back-translation + synthetic gaps  | Tablets have lacunae; model must handle missing content |
| Evaluation                       | BLEU + chrF++ + manual    | chrF++ better for morphologically rich languages |

## Data Pipeline

```
Sources:
├── Competition data: 8,000 texts, ~50% with translations
├── 900 scholarly publications (PDF/scans) → OCR → LLM correction → alignment extraction
└── Lexicons: proper nouns, Sumerograms, determinatives (provided)

Processing:
├── Preprocessing: normalize scribal notations per competition spec
├── Alignment: extract parallel sentences from publications
├── Deduplication: same tablet appears in multiple publications
└── Quality filter: remove paraphrases, keep literal translations

Target corpus size: 20-50k parallel pairs (2-3x provided data)
```

## Formatting Rules (from competition)

**Remove:** `!` `?` `/` `:` `.` `˹˺` `<<...>>` (keep content inside `[...]` and `<...>`)

**Preserve:** `{determinatives}`, `SUMEROGRAMS`, Capitalized proper nouns

**Normalize:** Gaps → `<gap>` (single sign) or `<big_gap>` (multiple signs)

## Roadmap

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| 0. Baseline | Week 1 | Zero-shot baseline score, submission pipeline |
| 1. Data | Weeks 2-3 | Clean corpus, publication extraction, 2-3x data |
| 2. Retrieval | Weeks 3-4 | Translation memory index, lexicon lookup |
| 3. Fine-tuning | Weeks 4-6 | Trained ByT5, RAG integration |
| 4. Iteration | Weeks 6-8 | Error analysis, ensembling, final submission |

## Critical Path

```
Publication extraction → Corpus size → Model performance
```

The 900 publications are the data multiplier. This extraction work has highest leverage.

## Evaluation Targets

| Metric | Baseline (zero-shot) | Target |
|--------|---------------------|--------|
| BLEU | ~5-10 | 30+ |
| chrF++ | ~20-30 | 50+ |
| Proper noun accuracy | ~40% | 85%+ |

## Reference Work

- **Akkademia** (2023): BLEU 36-37 using CNN on ~56k pairs from ORACC
- **BabyLemmatizer**: 94-96% accuracy on Akkadian morphological analysis
- **SLAB-NLP/Akk**: 89% hit@5 on gap-filling with BERT MLM

---

*Project context: Deep Past Challenge competition for Old Assyrian tablet translation. Goal is to unlock 10,000+ untranslated tablets and create a blueprint for low-resource ancient language NMT.*
