# Akkadian Machine Translation: Low-Resource NMT for Old Assyrian Cuneiform

A neural machine translation system for translating transliterated Old Assyrian cuneiform tablets into English, built for the [Kaggle Deep Past Challenge](https://www.kaggle.com/competitions/deep-past-challenge). Old Assyrian is a 4,000-year-old Semitic language with no native speakers, extreme morphological complexity, and minimal parallel training data — making it one of the hardest low-resource NMT problems in existence.

## The Problem

Old Assyrian presents challenges that standard NMT approaches can't handle out of the box:

- **Extreme low-resource setting.** ~1,589 parallel pairs total. For comparison, modern NMT systems train on millions. The prior state-of-the-art (Akkademia, 2023) used ~56,000 pairs from ORACC and achieved BLEU 36–37.
- **Mixed writing systems.** Transliterated texts contain Akkadian syllables (`a-na`, `qí-bi-ma`), Sumerograms (`KÙBABBAR` = silver), determinatives (`{d}` = divine), proper nouns, and numeric expressions — all in the same sentence.
- **Morphological density.** A single Akkadian word can encode an entire English clause. Standard BPE tokenization destroys the internal structure of these words.
- **Formulaic but variable.** Old Assyrian merchant letters follow rigid opening patterns, but the body contains free-form commercial, legal, and personal content.
- **No evaluation oracle.** With no native speakers, evaluation relies entirely on automated metrics (BLEU, chrF++) against scholarly translations that themselves vary in convention.

## Architecture

```
Raw Transliteration
        │
        ▼
┌─────────────────┐
│  Preprocessor    │  Strip scribal notation, normalize gaps,
│                  │  preserve determinatives/Sumerograms/names
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────────┐
│ BM25   │ │ Lexicon      │
│ Index  │ │ Lookup       │
│        │ │              │
│ Genre- │ │ 13,438 PNs   │
│ filter │ │ 228 Sumerog. │
│ k=3    │ │ 15,344 dict  │
└───┬────┘ └──────┬───────┘
    │              │
    └──────┬───────┘
           ▼
┌─────────────────────┐
│  Context Assembly    │  Lexicon glosses + genre-filtered
│                      │  BM25 examples + full input
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  ByT5-small          │  Byte-level seq2seq (299M params)
│  Fine-tuned          │  No tokenization — handles diacritics,
│                      │  hyphens, mixed scripts natively
└──────────┬──────────┘
           ▼
     English Output
```

## Key Design Decisions

**ByT5 over mT5/mBART.** Akkadian transliteration contains hyphens, diacritics (ṣ, š, ṭ, ḫ), subscript numbers (₄), and mixed scripts. BPE tokenization fragments these into meaningless subwords. ByT5 operates at the byte level, preserving the morphological structure that carries semantic information. Trade-off: longer sequences and slower training, but no information loss at the input boundary.

**BM25 on the Akkadian side over semantic embedding retrieval.** The initial system embedded the English side of the corpus with MiniLM and used FAISS for retrieval. Diagnostic testing revealed this produced 21% genre precision and 6.5% content overlap — effectively random retrieval. The problem is fundamental: at inference time, you don't have the English translation (that's what you're generating), so English-side similarity is meaningless. Switching to BM25 directly on the Akkadian transliterations with genre-based filtering improved genre precision to 92% and content overlap to 46%.

**Genre-filtered retrieval.** Old Assyrian texts cluster into distinct types (letters, legal documents, commercial records, administrative texts) with different vocabulary and structure. A legal document about a debt dispute has nothing useful to offer a model translating a merchant's shipping letter. The genre classifier uses Akkadian-side heuristics (letter formulas, witness markers, commodity Sumerograms) to hard-filter retrieval candidates before BM25 ranking.

**ByT5-small over base/large.** Counterintuitively, the smallest model performed best. With only 1,589 training pairs, larger models (580M base, 1.2B large) overfit — memorizing training examples without generalizing. ByT5-small achieved the best test scores across all configurations. This is consistent with the empirical finding that model capacity must be matched to data volume, especially in low-resource settings.

**Enriched context over scaffold pre-translation.** An intermediate design attempted to pre-translate deterministic components (names, numbers, Sumerograms, letter formulas) into an English scaffold that the model would complete. Systematic diagnostic testing (Stage 6) revealed that 90% of scaffolds didn't align with gold translation prefixes, creating contradictory training signals. The model learned to copy the scaffold and stop. The final design injects pre-translated information as structured hints in the context rather than as partial output, letting the model decide how to use them.

## Retrieval Redesign: A Case Study

The most impactful improvement came from diagnosing and rebuilding the retrieval system. Here's the before/after:

| Metric | Before (FAISS/English) | After (BM25/Akkadian + Genre) |
|---|---|---|
| Genre precision | 21.2% ± 30.7% | 91.8% ± 27.4% |
| Content overlap | 6.5% ± 5.4% | 46.4% ± 10.1% |
| Self-retrieval | 0 | 0 |
| Duplicate retrieval | 0 | 0 |

The diagnosis was straightforward once the right question was asked: "What are the top-3 retrieved examples for each validation input, and are they actually useful?" Printing them revealed the system was matching letters to legal documents and commercial records to administrative texts. The root cause — embedding the wrong language side — was a design-level error that no amount of hyperparameter tuning could fix.

## Diagnostic Framework

The project includes an 8-stage diagnostic pipeline that tests every component of the system in isolation:

| Stage | Tests | Key Findings |
|---|---|---|
| 1. Raw Input | Encoding, empty texts, split leakage | 100 near-duplicate prefixes in training |
| 2. Preprocessing | Special chars, gaps, idempotency | All edge cases handled correctly |
| 3. Pre-translation | Formula detection, PN resolution, numbers, Sumerograms, determinatives | 20/20 scaffold completeness after fixes |
| 4. RAG Retrieval | Genre filtering, self-match guard, score distribution | Self-retrieval bug caught and fixed |
| 5. Context Assembly | Byte length limits, truncation priority, redundancy | 12% of contexts exceeded input limit |
| 6. Training Target | Scaffold/gold alignment, empty targets, continuation check | 90% scaffold mismatch — led to architecture change |
| 7. Model Inference | Beam search ablation, context ablation | Truncation from training-time output length limit |
| 8. Final Assembly | Join quality, duplication | Clean concatenation verified |

This framework caught multiple critical bugs that would have been invisible from loss curves alone: the self-retrieval leak inflating scores, the scaffold-target mismatch causing the model to learn copy-and-stop, and the output length truncation cutting translations short.

## What Didn't Work (and Why)

**Scaffold pre-translation.** Pre-translating names, numbers, and Sumerograms into a partial English output seemed like a guaranteed win — these are deterministic mappings. But Old Assyrian word order doesn't map linearly to English. "KIŠIB X DUMU Y" becomes "Seal of X son of Y" in the gold, but the scaffold produced "seal son seal son" — isolated translations without structure. The model learned to reproduce scaffold content and emit EOS, achieving low loss on scaffold tokens while never learning to translate the ambiguous body text.

**Larger models.** ByT5-base (580M) and ByT5-large (1.2B) both scored lower than ByT5-small (300M). ByT5-large achieved train loss 0.25 vs eval loss 0.315 — classic overfitting. The extra capacity memorized training examples instead of learning transferable translation patterns. With 1.5k pairs, more parameters means more rope to hang yourself.

**Publication extraction.** 900 scholarly publications were processed with OCR and LLM correction to extract parallel pairs. Yield: 28 pairs (0.55%). Most publications discuss tablets rather than providing full translations, or provide paraphrased rather than literal translations. The effort was not worthwhile relative to improving retrieval and model training.

## Results

Competition metric: geometric mean of BLEU × chrF++.

| Configuration | Test Score | BLEU | chrF++ | PN Acc |
|---|---|---|---|---|
| ByT5-small + BM25/genre RAG (best) | 6.46 | 2.22 | 18.78 | 2.11% |
| ByT5-small + old FAISS RAG | 5.41 | 1.68 | 17.40 | 2.63% |
| ByT5-base + enriched hints | 3.28 | 0.72 | 14.92 | 2.63% |
| ByT5-large + BM25/genre RAG | 3.37 | 0.72 | 15.66 | 3.16% |
| Zero-shot baseline | 0.00 | 0.00 | 4.88 | 2.94% |

For reference, Akkademia (2023) achieved BLEU 36–37 using a CNN architecture trained on ~56,000 pairs from the ORACC corpus — 35× more training data than available in this challenge.

## Sample Translations

**Input:** `a-na e-lá-ma qí-bi₄-ma um-ma a-šur-SIPA-ma 4 GÚ 20 ma-na ANNA ku-nu-ki 1 me-at 21 TÚG mì-ma a-ni-im a-ra-de₈-a-kum`

**Model output:** `Say to Elamma, thus Aššur-rē'ī: 4 talents 20 minas of tin under seals, 72 textiles,`

**Gold:** `Say to Elamma, thus Aššur-rē'ī: 4 talent 20 minas of tin under seals (and) 121 textiles, all this I am leading to you.`

The model correctly identifies the letter formula, resolves the proper names, translates the commodity quantities (4 talents 20 minas of tin), and understands "under seals." It fails on the number "1 me-at 21" (121 → 72) and truncates before the sentence-final verb — both known issues traceable to the number parser and training-time output length limit.

## Repository Structure

```
├── configs/
│   └── training.yaml           # All hyperparameters
├── data/
│   ├── processed/              # Cleaned corpus
│   └── indices/                # FAISS + BM25 indices
├── diagnostics/
│   ├── stage1_raw_input.py
│   ├── stage2_preprocessing.py
│   ├── stage3_pretranslation.py
│   ├── stage4_rag_retrieval.py
│   ├── stage5_context_assembly.py
│   ├── stage6_training_target.py
│   ├── stage7_model_inference.py
│   ├── stage8_final_assembly.py
│   └── run_all.py
├── src/
│   ├── preprocessing/
│   │   ├── normalizer.py       # Scribal notation handling
│   │   └── pretranslator.py    # Formula/PN/number/Sumerogram
│   ├── retrieval/
│   │   ├── retriever.py        # BM25 + genre-filtered retrieval
│   │   ├── bm25_index.py
│   │   ├── genre.py            # Rule-based genre classifier
│   │   ├── lexicon.py          # PN/Sumerogram/dictionary lookup
│   │   ├── embedder.py
│   │   └── index_builder.py
│   ├── modeling/
│   │   ├── byt5_trainer.py     # Training loop
│   │   └── context_assembler.py
│   └── utils/
├── scripts/
│   ├── train.py
│   └── evaluate.py
└── models/
    └── byt5_finetuned/
```

## Running

```bash
# Install dependencies
pip install -r requirements.txt

# Run diagnostic suite
python diagnostics/run_all.py

# Train
python scripts/train.py

# Evaluate existing checkpoint
python scripts/evaluate.py --checkpoint models/byt5_finetuned/final
```

## Future Work

- **Number parsing.** The sexagesimal/multiplicative number system (`me-at` = hundred) is consistently mishandled. A rule-based number parser integrated as context hints would fix recurring errors like 121 → 72.
- **Proper noun copy mechanism.** PN accuracy never exceeded 9% across all runs. A pointer-network or constrained decoding approach could let the model copy names from the input rather than generating them byte-by-byte.
- **Enriched hints (revisited).** The hint-based context format underperformed in initial testing, but the diagnostic framework identified specific bugs (float garbage in numbers, dictionary definitions leaking into glosses) that were fixed. A clean A/B test on the corrected hints is warranted.
- **Data augmentation.** Back-translation, span corruption, and more aggressive synthetic gap generation could stretch the 1,589 pairs further.
- **Ensemble approaches.** Multiple ByT5-small checkpoints trained with different random seeds, with output selected by confidence scoring.

## References

- Akkademia (2023): CNN-based Akkadian NMT, BLEU 36–37 on ORACC corpus
- Xue et al. (2022): ByT5 — Token-Free Pre-trained Models
- BabyLemmatizer: 94–96% accuracy on Akkadian morphological analysis
- SLAB-NLP/Akk: 89% hit@5 on gap-filling with BERT MLM
