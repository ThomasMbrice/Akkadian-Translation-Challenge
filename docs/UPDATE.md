# RAG Redesign: Akkadian NMT Pipeline

**Date:** 2026-02-20
**Status:** Implemented ✓ — diagnostic targets met, ready to retrain
**Goal:** Fix retrieval quality to improve competition score from 5.41 → 15+ (intermediate target)

## Results (2026-02-20)

| Metric | Before | After | Target | Status |
|---|---|---|---|---|
| Genre precision | 21.2% | **91.8%** | 85%+ | ✓ |
| Content overlap | 6.5% | **46.4%** | 20%+ | ✓ |
| Duplicate retrievals | 0 | 2 | 0 | ⚠ minor |
| Unknown genre bucket | 17.6% | 8.2% | — | improved |

---

## Problem

Current retrieval is failing. Diagnostic on the validation set (n=159, k=3):

| Metric | Current | Target |
|---|---|---|
| Genre precision | 21.2% | 85%+ |
| Content overlap | 6.5% | 20%+ |
| PN accuracy | 2.63% | 50%+ |

The system embeds the English side of the corpus and does semantic similarity search. This is fundamentally broken because at inference time we don't have the English — that's what we're generating. Retrieved examples are from the wrong genre ~80% of the time and share almost no vocabulary with the input. The model is learning *despite* the RAG context, not because of it.

---

## Changes

### 1. Genre Classifier (Rule-Based)

Classify every text in the corpus and every input at inference time into one of five categories using Akkadian-side heuristics:

- **letter**: contains `um-ma ... qí-bi-ma` or `a-na ... qí-bi₄-ma`
- **legal**: contains witness markers (`IGI`), oath formulas, debt/obligation terms
- **commercial**: heavy commodity Sumerograms (`ANNA`, `TÚG`, `KÙBABBAR`, `GÚ`, `GÍN`) without letter framing
- **administrative**: inventory/list structure, no narrative
- **unknown**: fallback when no pattern matches

Priority: reduce the 17.6% "unknown" bucket by inspecting those 28 texts and writing additional rules if a pattern emerges.

### 2. BM25 Index on the Akkadian Side

Replace the primary retrieval signal. Index every Akkadian source text in the corpus using BM25 (e.g., `rank_bm25`). At query time, score the transliterated input directly against the Akkadian corpus.

This fixes content overlap because we're matching on the actual words the model needs to translate — Sumerograms, proper nouns, verbal forms — not on English translations we haven't produced yet.

### 3. Genre-Filtered Retrieval

At query time:
1. Classify the input
2. Filter the BM25 candidate pool to the same genre (hard filter, not soft weight)
3. Rank within that filtered pool by BM25 score
4. If genre is "unknown", fall back to unfiltered BM25

### 4. Relevance Threshold

After genre-filtered BM25 ranking, only include examples above a minimum BM25 score. If the best match is weak, return fewer examples rather than padding with noise.

Calibration: sweep threshold values on the validation set. Plot retrieval score vs. downstream translation quality and pick the elbow.

- k=3 maximum, but 1 good example > 3 mediocre ones
- If no example passes threshold, skip retrieved examples entirely and rely on lexicon glosses alone

### 5. Letter Formula Detection

For texts classified as letters, run a regex-based detector on the first ~30 tokens to identify:

- **Pattern A:** `um-ma X ... a-na Y qí-bi-ma` → sender=X, recipient=Y
- **Pattern B:** `a-na Y qí-bi₄-ma um-ma X-ma` → recipient=Y, sender=X

Handle variants: `qí-bi` vs `qí-bi₄`, patronymics (`X DUMU Y`), multiple names joined by `ù`.

Output: a structural annotation prepended to the context.

---

## Context Assembly (New Format)

For each input, build the prompt context in this order:

```
[1] Structural annotation (if letter detected):
    "Letter from {sender} to {recipient}"

[2] Lexicon glosses (terse, one per line):
    KÙBABBAR = silver
    ANNA = tin
    A-šùr-i-mì-tí = Aššur-imittī (PN)
    DUMU = son

[3] Retrieved examples (1-3, ranked by relevance):
    Example 1: [full Akkadian → English pair]
    Example 2: [full pair, if above threshold]
    Example 3: [full pair, if above threshold]

[4] Translate: [cleaned input]
```

Key change: examples are ordered by relevance, capped by threshold, and come from the same genre. The structural annotation is new. Lexicon glosses stay the same.

---

## Validation Plan

**Before retraining:**

Rerun the retrieval diagnostic with the new pipeline on the same 159 validation examples. Check:

- Genre precision ≥ 85%
- Content overlap ≥ 20%
- No self-retrievals or duplicate retrievals

If these don't improve on paper, debug before retraining.

**After retraining:**

Same evaluation as before — geometric mean of BLEU × chrF++, plus PN accuracy. Run the same 50-epoch config (run 1704838) to isolate the retrieval variable.

---

## What's NOT Changing

- Model architecture (ByT5-small)
- Training hyperparameters
- Data augmentation strategy
- Post-processing pipeline
- Lexicon lookup logic
- Preprocessing / normalization

One variable at a time.

---

## Implementation Order

1. ✓ Build genre classifier, tag the corpus (`src/retrieval/genre.py`)
2. ✓ Build BM25 index on Akkadian side (`src/retrieval/bm25_index.py`)
3. ✓ Wire up genre-filtered retrieval with threshold (`Retriever.retrieve_bm25()`)
4. ✓ Build letter formula detector (`genre.detect_letter_formula()`)
5. ✓ Update context assembly (`src/modeling/context_assembler.py`)
6. ✓ Rerun retrieval diagnostic — targets met
7. → Retrain and evaluate