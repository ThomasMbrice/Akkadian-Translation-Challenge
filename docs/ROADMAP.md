# Project Roadmap

## Overview

This roadmap tracks progress toward the Deep Past Challenge competition goal: building a neural machine translation system for Old Assyrian cuneiform → English.

**Competition timeline:** 8 weeks total
**Critical path:** Publication extraction → corpus size → model performance

---

## Phase 0: Baseline (Week 1)

**Goal:** Establish zero-shot baseline and end-to-end pipeline

### Deliverables
- [ ] Competition data downloaded to `data/raw/`
- [ ] Basic preprocessing implemented (`src/preprocessing/normalizer.py`)
- [ ] Zero-shot ByT5 baseline tested
- [ ] Submission pipeline functional (`scripts/submit.py`)
- [ ] Baseline scores recorded

### Key Metrics (Expected)
| Metric | Target |
|--------|--------|
| BLEU | 5-10 |
| chrF++ | 20-30 |
| Proper noun accuracy | ~40% |

### Success Criteria
✓ Submission generated and uploaded
✓ Pipeline runs without errors
✓ Baseline scores documented in `notebooks/02_baseline.ipynb`

### Blockers / Risks
- Competition data format changes
- Zero-shot ByT5 may produce empty outputs (very low-resource)

---

## Phase 1: Data Extraction (Weeks 2-3)

**Goal:** Extract 2-3x more training data from 900 scholarly publications

**This is the highest-leverage activity for model performance.**

### Deliverables
- [ ] OCR correction pipeline (`src/data_extraction/ocr_corrector.py`)
- [ ] Publication parser (`src/data_extraction/publication_parser.py`)
- [ ] Sentence alignment (`src/preprocessing/aligner.py`)
- [ ] Deduplication logic (`src/data_extraction/deduplicator.py`)
- [ ] 20-50k parallel sentence pairs in `data/processed/combined_corpus.csv`

### Tasks Breakdown

#### Week 2: OCR Correction & Extraction
- [ ] Implement LLM-based OCR error correction
- [ ] Parse publications.csv for Akkadian-Translation pairs
- [ ] Match document IDs with published_texts.csv
- [ ] Convert non-English translations → English (French, German, Turkish)

#### Week 3: Alignment & Quality Control
- [ ] Sentence-level alignment using Sentences_Oare_FirstWord_LinNum.csv
- [ ] Deduplication (same tablet across multiple publications)
- [ ] Quality filtering (literal translations only, not paraphrases)
- [ ] Manual inspection of 50 random samples

### Key Metrics
| Metric | Target |
|--------|--------|
| Total parallel pairs extracted | 20k-50k |
| Sentence-level alignment accuracy | 90%+ |
| Duplicate rate | <10% |

### Success Criteria
✓ Combined corpus size 2-3x larger than train.csv
✓ Proper nouns match OA_Lexicon_eBL.csv
✓ Sumerograms preserved correctly
✓ Manual inspection passes quality bar

### Blockers / Risks
- OCR quality in publications.csv may be poor
- Multi-language translations may require careful LLM prompting
- Sentence alignment heuristics may fail for complex tablets

---

## Phase 2: Retrieval System (Weeks 3-4)

**Goal:** Build translation memory (RAG) and lexicon lookup

### Deliverables
- [ ] Lexicon builder (`src/retrieval/lexicon.py`)
- [ ] English embedder (`src/retrieval/embedder.py`)
- [ ] FAISS index builder (`src/retrieval/index_builder.py`)
- [ ] Retriever interface (`src/retrieval/retriever.py`)
- [ ] FAISS index saved to `data/indices/faiss_index.bin`

### Tasks Breakdown

#### Week 3: Lexicon & Embeddings
- [ ] Load OA_Lexicon_eBL.csv → proper noun dictionary
- [ ] Load eBL_Dictionary.csv → Sumerogram mappings
- [ ] Implement fuzzy matching for inflected forms
- [ ] Embed English side of corpus with sentence-transformers

#### Week 4: Indexing & Retrieval
- [ ] Build FAISS index over embeddings
- [ ] Implement k=5 retrieval
- [ ] Test retrieval on validation set (letter formulas, contracts, etc.)

### Key Metrics
| Metric | Target |
|--------|--------|
| Lexicon coverage | 90%+ of proper nouns |
| Retrieval precision@5 | 60%+ relevant examples |
| Retrieval latency | <100ms per query |

### Success Criteria
✓ Given input, retriever returns 5 similar Akkadian-English pairs
✓ Proper nouns resolved via lexicon lookup
✓ Sumerograms mapped correctly (e.g., DUMU → "son")

### Blockers / Risks
- No Akkadian embedder exists (embedding English side is heuristic)
- Retrieval quality hard to evaluate without labeled data

---

## Phase 3: Model Fine-Tuning (Weeks 4-6)

**Goal:** Train ByT5 with RAG context integration

### Deliverables
- [ ] Data augmentation (`src/modeling/augmentation.py`)
- [ ] Context assembly (`src/modeling/context_assembler.py`)
- [ ] ByT5 trainer (`src/modeling/byt5_trainer.py`)
- [ ] Trained model checkpoints in `models/byt5_finetuned/`
- [ ] Validation scores tracked

### Tasks Breakdown

#### Week 4-5: Data Prep & Initial Training
- [ ] Implement synthetic gap augmentation
- [ ] Format RAG context (lexicon + retrieved examples + input)
- [ ] Set up training pipeline (Hugging Face Transformers + Accelerate)
- [ ] Initial training run (5 epochs)

#### Week 5-6: Hyperparameter Tuning
- [ ] Experiment with learning rates (5e-5, 1e-4)
- [ ] Experiment with batch sizes (8, 16, 32)
- [ ] Try `byt5-base` vs `byt5-large`
- [ ] Validation-based early stopping

### Key Metrics
| Metric | Target |
|--------|--------|
| Validation BLEU | 25-35 |
| Validation chrF++ | 45-55 |
| Proper noun accuracy | 75-85% |

### Success Criteria
✓ Validation BLEU > 25
✓ Model handles gaps (`<gap>`, `<big_gap>`) correctly
✓ Proper nouns preserved in output

### Blockers / Risks
- Training may require 16GB+ VRAM (A100/V100 access)
- Overfitting possible with small corpus (regularization needed)
- RAG context may exceed ByT5 max length (1024 bytes)

---

## Phase 4: Iteration & Submission (Weeks 6-8)

**Goal:** Error analysis, ensembling, final submission

### Deliverables
- [ ] Error analysis report (`notebooks/03_error_analysis.ipynb`)
- [ ] Targeted fixes implemented
- [ ] Final model(s) trained
- [ ] Test set predictions generated
- [ ] Competition submission uploaded

### Tasks Breakdown

#### Week 6-7: Error Analysis
- [ ] Categorize errors: proper nouns, Sumerograms, morphology, fluency
- [ ] Identify systematic failures
- [ ] Implement targeted fixes
  - If proper noun accuracy low → improve lexicon lookup
  - If gap handling poor → more augmentation
  - If fluency poor → adjust beam search / temperature

#### Week 7-8: Final Training & Submission
- [ ] Retrain with fixes
- [ ] Optional: Ensemble multiple models (different seeds, hyperparameters)
- [ ] Generate test.csv predictions with `scripts/inference.py`
- [ ] Create submission with `scripts/submit.py`
- [ ] Upload to Kaggle

### Key Metrics (Final)
| Metric | Baseline | Target | Stretch |
|--------|----------|--------|---------|
| Test BLEU | 5-10 | 30+ | 35+ |
| Test chrF++ | 20-30 | 50+ | 55+ |
| Proper noun accuracy | ~40% | 85%+ | 90%+ |

### Success Criteria
✓ Test BLEU ≥ 30 (competitive with Akkademia baseline)
✓ Submission uploaded before competition deadline
✓ All code documented and reproducible

### Blockers / Risks
- Test set may have different distribution (genre, time period)
- Competition leaderboard may not reflect final standings (overfitting public LB)

---

## Milestones Summary

| Week | Phase | Key Deliverable |
|------|-------|-----------------|
| 1 | Baseline | Zero-shot submission uploaded |
| 2-3 | Data | 20-50k parallel pairs extracted |
| 3-4 | Retrieval | FAISS index + lexicon operational |
| 4-6 | Training | ByT5 fine-tuned, validation BLEU > 25 |
| 6-8 | Iteration | Final submission, test BLEU ≥ 30 |

---

## Critical Path

```
Publication extraction → Corpus size → Model performance
```

**Focus:** Weeks 2-3 (data extraction) are the highest-leverage activities. Every additional parallel pair improves model performance.

---

## Current Status

**Phase:** [TO BE UPDATED]
**Week:** [TO BE UPDATED]
**Completed deliverables:** [TO BE UPDATED]
**Blockers:** [TO BE UPDATED]

---

## Reference Benchmarks

| System | Corpus Size | BLEU | Notes |
|--------|-------------|------|-------|
| **Akkademia (2023)** | 56k pairs | 36-37 | CNN-based, ORACC data |
| **Zero-shot ByT5** | 0 (zero-shot) | 5-10 | Our baseline |
| **Target (this project)** | 20-50k pairs | 30+ | Fine-tuned ByT5 + RAG |

---

*This roadmap is a living document. Update as phases complete and blockers arise.*
