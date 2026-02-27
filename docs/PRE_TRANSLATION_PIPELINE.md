# Pre-Translation Pipeline: Rule-Based Deterministic Components

**Date:** 2026-02-27
**Status:** Proposed
**Depends on:** RAG Redesign (completed, genre precision 91.8%, content overlap 46.4%)
**Model:** ByT5-base (580M params)
**Goal:** Remove deterministic translation burden from the neural model. Target score: 15+

---

## Problem

Across all runs (small, base, large), PN accuracy never exceeds 3.16%. The model cannot learn to copy names, translate Sumerograms, parse numbers, or reproduce letter formulas from 1.5k examples regardless of model size. These are all deterministic mappings that don't require neural inference.

Current best: 6.46 (ByT5-small + upgraded RAG). The model is doing all the work — formulas, names, numbers, Sumerograms, and genuine semantic content. It should only be doing the last one.

---

## Design

Split translation into two phases:

1. **Pre-translate** everything deterministic → produce a partial English scaffold with placeholders for ambiguous content
2. **Model translates** only the remaining ambiguous Akkadian, guided by the scaffold

The model's input changes from "translate this entire text" to "complete this partial translation."

---

## Components

### 1. Letter Formula Translator

Already have the regex detector from the RAG redesign. Extend it to produce English output directly.

**Pattern A:** `um-ma X ... a-na Y qí-bi-ma`
→ `"Thus (says) {X_english}; say to {Y_english}:"`

**Pattern B:** `a-na Y qí-bi₄-ma um-ma X-ma`
→ `"Say to {Y_english}, thus {X_english}:"`

Name slots get resolved through the proper noun lexicon (component 2). If a name isn't in the lexicon, pass through the transliteration as-is — the model or post-processing can attempt it.

After extracting the formula, strip it from the Akkadian input so the model only sees the body of the letter.

### 2. Proper Noun Resolver

Use the existing proper noun dictionary to map transliterated names to normalized English forms.

**Input:** Capitalized sequences in the Akkadian text (already extracted by preprocessor)
**Process:**
- Exact match against lexicon: `A-šùr-i-mì-tí` → `Aššur-imittī`
- Fuzzy match for scribal variants (edit distance ≤ 2): `A-šur-i-mì-tí` → `Aššur-imittī`
- Patronymic handling: `X DUMU Y` → `X son of Y` (resolve both names independently)
- Unknown names: transliterate mechanically (remove hyphens, normalize diacritics) rather than letting the model hallucinate

**Output:** A name map injected into the scaffold. In the Akkadian input, replace each name with a tagged placeholder like `<PN1>` so the model doesn't need to generate names at all.

### 3. Sumerogram Translator

Sumerograms are logograms with fixed deterministic meanings. These should never reach the neural model.

**Core mappings (from provided lexicon):**
- `KÙBABBAR` → silver
- `ANNA` → tin
- `TÚG` → textile(s)
- `GÚ` → talent (weight unit)
- `GÍN` → shekel
- `ma-na` → mina
- `DUMU` → son
- `KÙ.GI` → gold
- `IGI` → witness / before

Replace in the Akkadian input with English equivalents or tagged placeholders. Context determines singular/plural — default to singular, let the model adjust in context.

### 4. Number Parser

Old Assyrian numbers follow predictable patterns that the model consistently gets wrong.

**Rules:**
- `me-at` = hundred (multiplicative with preceding number)
- Digit + `GÍN` = N shekels
- Digit + `ma-na` = N minas
- Digit + `GÚ` = N talents
- Compound: `4 GÚ 20 ma-na` = 4 talents 20 minas

Parse left-to-right, accumulating number + unit pairs. Output the English number string directly.

### 5. Determinative Resolver

Determinatives are semantic classifiers in curly braces. Fixed mappings:

- `{d}` → divine (marks a deity name)
- `{f}` → female (marks a woman's name)
- `{m}` → male (marks a man's name, often omitted)
- `{ki}` → place (marks a geographic name)
- `{URU}` → city

Strip the determinative from output but use it to inform name resolution — `{d}UTU` = the god Shamash, `{ki}` signals a place name lookup instead of a personal name lookup.

---

## Scaffold Format

The pre-translation pipeline produces a partial English translation with markers for the model to complete.

**Example input:**
```
a-na ku-li-a qí-bi₄-ma um-ma a-šùr-i-mì-tí-ma 4 GÚ 20 ma-na ANNA ku-nu-ki 121 TÚG
mì-ma a-ni-im a-ra-de₈-a-kum
```

**After pre-translation:**
```
Scaffold: "Say to Kuliya, thus Aššur-imittī: 4 talents 20 minas of tin [sealed] 121 textiles"
Remaining Akkadian: "mì-ma a-ni-im a-ra-de₈-a-kum"
```

**Model task:** Translate the remaining Akkadian and integrate it with the scaffold to produce a complete, fluent English translation.

---

## Context Assembly (Updated)

```
[1] Scaffold (pre-translated components):
    "Say to Kuliya, thus Aššur-imittī: 4 talents 20 minas of tin [sealed] 121 textiles"

[2] Remaining lexicon glosses (only for untranslated terms):
    ku-nu-ki = under seal
    a-ra-de₈-a-kum = I am bringing to you

[3] Retrieved examples (1-3, genre-filtered, BM25-ranked):
    [Akkadian → English pair]

[4] Complete the translation: [remaining Akkadian]
```

Key change from previous: the model now sees a partial answer and needs to fill in the gaps rather than translating from scratch. RAG examples still provide structural guidance for the ambiguous portion.

---

## Training Implications

The training data format changes. Every training pair needs to be processed through the same pre-translation pipeline so the model learns to work with scaffolds, not raw Akkadian.

**Training input:** scaffold + remaining Akkadian + RAG context
**Training target:** full English translation

This means reprocessing the entire corpus through the pipeline before training. The model learns: "given this partial translation and remaining source text, produce the complete translation."

---

## Evaluation Plan

**Unit tests before training:**
- Run the formula detector on all letters in the corpus. Manually check 20 for correctness.
- Run the PN resolver on all names in the test set. Compute accuracy against gold translations. Target: 70%+ (up from 3%).
- Run the number parser on all numeric expressions in the test set. Spot-check 10.

**Integration test:**
- Generate scaffolds for the full validation set. Manually inspect 10 for quality.
- Check that no information is lost — the scaffold + remaining Akkadian should contain everything needed to produce the gold translation.

**Model evaluation:**
- Same setup: ByT5-base, 50 epochs, same hyperparameters
- Compare geometric mean score with and without pre-translation
- Track PN accuracy separately — this should jump dramatically since names are pre-resolved

---

## What's NOT Changing

- RAG retrieval logic (genre-filtered BM25, already validated)
- Model architecture (ByT5-base)
- Augmentation strategy
- Evaluation metrics
- Post-processing pipeline

---

## Implementation Order

1. Build proper noun resolver (exact + fuzzy matching against lexicon)
2. Build Sumerogram translator (dictionary lookup)
3. Build number parser
4. Build determinative resolver
5. Extend letter formula detector to produce English output
6. Build scaffold assembler (combines all components)
7. Reprocess training corpus through the pipeline
8. Update context assembly format
9. Unit test each component on validation set
10. Retrain and evaluate
