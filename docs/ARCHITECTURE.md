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
│  │  Training: 1,589 pairs + augmentation               │   │
│  │  Augmentation: back-translation, synthetic gaps     │   │
│  │  Multi-task: translation + lemmatization + POS      │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  POST-PROCESSOR (Mini-LLM)                          │   │
│  │                                                      │   │
│  │  Model: Phi-3-mini or Llama-3.2-1B (1-4B params)   │   │
│  │  Task: Clean and refine ByT5 output                │   │
│  │  - Fix formatting errors                            │   │
│  │  - Ensure proper noun capitalization               │   │
│  │  - Preserve Sumerograms faithfully                 │   │
│  │  - Improve fluency without hallucination           │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│                    English Output (Refined)                 │
└─────────────────────────────────────────────────────────────┘
```

## Key Technical Decisions

| Decision                         | Choice                    | Rationale |
|----------------------------------|---------------------------|-----------|
| Tokenization                     | Byte-level (ByT5)         | Akkadian has hyphens, diacritics, mixed scripts—BPE mangles it |
| Retrieval embedding              | Embed English side        | No pretrained Akkadian embedders exist |
| RAG integration                  | Context stuffing          | Simpler than Fusion-in-Decoder, sufficient for formulaic texts |
| Data augmentation                | Back-translation + synthetic gaps  | Tablets have lacunae; model must handle missing content |
| Post-processing                  | Mini-LLM (Phi-3/Llama)    | Refine outputs, fix formatting, improve proper noun handling |
| Evaluation                       | **Geometric mean (sqrt(BLEU × chrF))** + manual | **Competition metric**; chrF++ better for morphologically rich languages |

## Data Pipeline

```
Sources:
├── Competition data: 8,000 texts, ~50% with translations
├── 900 scholarly publications (PDF/scans) → OCR → LLM correction (28 pairs extracted)
└── Lexicons: proper nouns, Sumerograms, determinatives (provided)

Processing:
├── Preprocessing: normalize scribal notations per competition spec
├── Alignment: extract parallel sentences from publications
├── Deduplication: same tablet appears in multiple publications
└── Quality filter: remove paraphrases, keep literal translations

Corpus: 1,589 parallel pairs (1,561 competition + 28 extracted)
Augmentation: synthetic gaps at training time (30% probability)
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
| 4. Post-processing | Week 6 | Mini-LLM output refinement, +3-5 point improvement |
| 5. Iteration | Weeks 7-8 | Error analysis, ensembling, final submission |

## Critical Path

```
RAG context quality → Model performance
```

Publication extraction yielded only 28 pairs (0.55% yield); the leverage is now
retrieval fidelity — lexicon glosses and translation memory — plus augmentation.

## Evaluation Targets

| Metric | Baseline (zero-shot) | Target |
|--------|---------------------|--------|
| BLEU | ~5-10 (actual: 0.00) | 30+ |
| chrF++ | ~20-30 (actual: 4.88) | 50+ |
| **Geometric Mean (SCORE)** | **~12-18 (actual: 0.00)** | **~39+** |
| Proper noun accuracy | ~40% (actual: 2.94%) | 85%+ |

> **Competition Scoring:** The official metric is the geometric mean: sqrt(BLEU × chrF)

## Reference Work

- **Akkademia** (2023): BLEU 36-37 using CNN on ~56k pairs from ORACC
- **BabyLemmatizer**: 94-96% accuracy on Akkadian morphological analysis
- **SLAB-NLP/Akk**: 89% hit@5 on gap-filling with BERT MLM

---

*Project context: Deep Past Challenge competition for Old Assyrian tablet translation. Goal is to unlock 10,000+ untranslated tablets and create a blueprint for low-resource ancient language NMT.*
