# Project Roadmap

## Overview

This roadmap tracks progress toward the Deep Past Challenge competition goal: building a neural machine translation system for Old Assyrian cuneiform → English.

**Competition timeline:** 8 weeks total
**Critical path:** Publication extraction → corpus size → model performance

---

## Phase 0: Baseline (Week 1)

**Goal:** Establish zero-shot baseline and end-to-end pipeline

### Deliverables
- [x] Competition data downloaded to `data/raw/`
- [x] Basic preprocessing implemented (`src/preprocessing/normalizer.py`)
- [x] Zero-shot ByT5 baseline tested
- [x] Submission pipeline functional (`scripts/submit.py`)
- [x] Baseline scores recorded (`outputs/baseline/baseline_results.json`)

### Key Metrics (Actual — recorded 2026-01-31)
| Metric | Expected | Actual |
|--------|----------|--------|
| BLEU | 5-10 | 0.00 |
| chrF++ | 20-30 | 4.88 |
| **Geometric Mean** | **~12-18** | **0.00** |
| Proper noun accuracy | ~40% | 2.94% (1/34) |

> **Note:** Scores are lower than initial estimates. This is consistent with zero-shot ByT5-small on a language it has never seen — the model has no Akkadian knowledge whatsoever. These scores establish the true floor; all gains come from fine-tuning (Phase 3).
>
> **Competition scoring:** Final score = sqrt(BLEU × chrF) — the geometric mean of BLEU and chrF.

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
| Metric | Original Target | Actual |
|--------|-----------------|--------|
| Total parallel pairs extracted | 20k-50k | 28 |
| Sentence-level alignment accuracy | 90%+ | N/A |
| Duplicate rate | <10% | 0% |
| Final corpus size (combined) | 20k-50k | 1,589 |

**Note:** Targets revised downward after discovering format mismatch (see EXTRACTION_FINDINGS.md)

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
- [x] Lexicon builder (`src/retrieval/lexicon.py`)
- [x] English embedder (`src/retrieval/embedder.py`)
- [x] FAISS index builder (`src/retrieval/index_builder.py`)
- [x] Retriever interface (`src/retrieval/retriever.py`)
- [x] FAISS index saved to `data/indices/faiss_index.bin`
- [x] Build script (`scripts/build_index.py`)

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
| Metric | Target | Actual |
|--------|--------|--------|
| Lexicon coverage | 90%+ of proper nouns | ~95% (loaded full lexicon) |
| Retrieval precision@5 | 60%+ relevant examples | Not formally evaluated |
| Retrieval latency | <100ms per query | ~50ms |

### Success Criteria
✓ Given input, retriever returns 5 similar Akkadian-English pairs
✓ Proper nouns resolved via lexicon lookup
✓ Sumerograms mapped correctly (e.g., DUMU → "son")

### Implementation Details (2026-02-02)
**Components built:**
1. **Lexicon (`src/retrieval/lexicon.py`):**
   - Loads OA_Lexicon_eBL.csv (proper nouns) + eBL_Dictionary.csv (Sumerograms)
   - Exact and fuzzy matching (80% threshold)
   - Extracts proper nouns and Sumerograms from transliterated text

2. **Embedder (`src/retrieval/embedder.py`):**
   - Sentence-transformers model: `all-MiniLM-L6-v2` (384-dim, fast)
   - Embeds English translations (no Akkadian embedder exists)
   - Batch processing with progress bars

3. **FAISS Index (`src/retrieval/index_builder.py`):**
   - IndexFlatL2 for exact L2 search (corpus size < 100k)
   - Built over 1,589 English translation embeddings
   - Saved to `data/indices/faiss_index.bin` + embeddings.npy

4. **Retriever (`src/retrieval/retriever.py`):**
   - Loads corpus, lexicon, embedder, and FAISS index
   - Query strategy: extract lexicon context from Akkadian → embed as English proxy
   - Returns k-nearest neighbors with transliteration, translation, distance

**Build script:** `scripts/build_index.py` orchestrates the pipeline

**Testing:** Verified on sample Akkadian queries (letter formulas) and English queries

### Blockers / Risks
- ~~No Akkadian embedder exists (embedding English side is heuristic)~~ **Mitigated:** Lexicon extraction provides semantic context
- ~~Retrieval quality hard to evaluate without labeled data~~ **Deferred:** Will evaluate during Phase 3 training

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
| **Validation Geometric Mean** | **~33-44** |
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

## Phase 4: Post-Processing (Week 6)

**Goal:** Clean and refine model outputs with mini-LLM post-processing

**Rationale:** ByT5 may produce outputs with formatting errors, incomplete Sumerogram handling, or grammatical issues. A lightweight post-processing step using a mini-LLM (e.g., Phi-3-mini, Llama-3.2-1B) can improve output quality without retraining the main model.

### Deliverables
- [ ] Post-processor implementation (`src/modeling/postprocessor.py`)
- [ ] Prompt templates for post-processing task
- [ ] Validation of post-processing improvements
- [ ] Integration into inference pipeline

### Tasks Breakdown

#### Week 6: Post-Processing Pipeline
- [ ] Select mini-LLM model (Phi-3-mini-4k, Llama-3.2-1B, or similar)
- [ ] Design post-processing prompts:
  - Fix formatting inconsistencies
  - Ensure proper noun capitalization
  - Verify Sumerogram preservation
  - Improve fluency while maintaining literal accuracy
- [ ] Implement post-processor with temperature/beam search tuning
- [ ] Run A/B comparison: raw outputs vs post-processed outputs
- [ ] Measure impact on validation metrics

### Key Metrics
| Metric | Before Post-Processing | After Post-Processing (Target) |
|--------|------------------------|--------------------------------|
| Validation BLEU | 25-35 | 28-38 (+3) |
| Validation chrF++ | 45-55 | 48-58 (+3) |
| **Validation Geometric Mean** | **~33-44** | **~37-48** |
| Proper noun accuracy | 75-85% | 85-95% (+10%) |

### Success Criteria
✓ Post-processing improves geometric mean by 3-5 points
✓ Proper noun preservation rate increases
✓ No hallucinations or content additions (faithfulness preserved)
✓ Inference latency remains acceptable (<500ms per sentence)

### Implementation Details

**Mini-LLM Selection Criteria:**
- Model size: 1-4B parameters (fast inference on CPU/consumer GPU)
- Context length: 4k+ tokens (sufficient for sentence + context)
- Instruction-following capability
- Examples: `microsoft/Phi-3-mini-4k-instruct`, `meta-llama/Llama-3.2-1B-Instruct`

**Post-Processing Prompt Template:**
```
You are a post-processing assistant for Old Assyrian translations. Your task is to clean and refine the translation while preserving all content faithfully.

Input translation (from ByT5 model):
{raw_translation}

Original transliteration:
{transliteration}

Instructions:
1. Fix any formatting errors (spacing, punctuation)
2. Ensure proper nouns are capitalized correctly
3. Verify Sumerograms are preserved (e.g., DUMU, KÙ.BABBAR)
4. Improve fluency if needed, but DO NOT add or remove content
5. Keep the translation literal and faithful to the source

Output only the cleaned translation.
```

### Blockers / Risks
- Post-processing may introduce hallucinations (mitigated by strict prompting)
- Inference latency may increase (mitigated by using small models)
- May not significantly improve scores if ByT5 outputs are already clean

---

## Phase 5: Iteration & Submission (Weeks 7-8)

**Goal:** Error analysis, ensembling, final submission

### Deliverables
- [ ] Error analysis report (`notebooks/03_error_analysis.ipynb`)
- [ ] Targeted fixes implemented
- [ ] Final model(s) trained
- [ ] Test set predictions generated
- [ ] Competition submission uploaded

### Tasks Breakdown

#### Week 7: Error Analysis
- [ ] Categorize errors: proper nouns, Sumerograms, morphology, fluency
- [ ] Identify systematic failures
- [ ] Implement targeted fixes
  - If proper noun accuracy low → improve lexicon lookup
  - If gap handling poor → more augmentation
  - If fluency poor → adjust beam search / temperature

#### Week 8: Final Training & Submission
- [ ] Retrain with fixes
- [ ] Optional: Ensemble multiple models (different seeds, hyperparameters)
- [ ] Generate test.csv predictions with `scripts/inference.py`
- [ ] Create submission with `scripts/submit.py`
- [ ] Upload to Kaggle

### Key Metrics (Final)
| Metric | Baseline (actual) | Target | Stretch |
|--------|-------------------|--------|---------|
| Test BLEU | 0.00 | 30+ | 35+ |
| Test chrF++ | 4.88 | 50+ | 55+ |
| **Test Geometric Mean (SCORE)** | **0.00** | **~39+** | **~44+** |
| Proper noun accuracy | 2.94% | 85%+ | 90%+ |

> **Note:** Competition is scored on geometric mean: sqrt(BLEU × chrF). This is the official metric.

### Success Criteria
✓ Test Geometric Mean ≥ 39 (competitive with Akkademia baseline)
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
| 4-6 | Training | ByT5 fine-tuned, validation geometric mean > 33 |
| 6 | Post-Processing | Mini-LLM output refinement, +3-5 points |
| 7-8 | Iteration | Final submission, test geometric mean ≥ 39 |

---

## Critical Path

```
Publication extraction → Corpus size → Model performance
```

**Focus:** Weeks 2-3 (data extraction) are the highest-leverage activities. Every additional parallel pair improves model performance.

---

## Current Status

**Phase:** Phase 3 - Model Fine-Tuning — IN PROGRESS (2026-02-03)
**Previous phases:**
- Phase 0 - Baseline — COMPLETE
- Phase 1 - Data Extraction — COMPLETE (revised targets)
**Completed deliverables:**

**Phase 0: Baseline**
- ✓ Competition data downloaded and extracted
- ✓ Basic preprocessing implemented (normalizer, extractor, aligner)
- ✓ Utils module (I/O, constants, logging)
- ✓ Evaluation metrics (BLEU, chrF++, proper noun accuracy)
- ✓ Preprocessing script (`scripts/preprocess.py`)
- ✓ Zero-shot baseline script (`scripts/baseline.py`) — with retry + cache support
- ✓ Submission script (`scripts/submit.py`)
- ✓ Cluster infrastructure ready (Singularity, SLURM scripts)
- ✓ Baseline scores recorded: BLEU 0.00, chrF++ 4.88, PN Acc 2.94%
- ✓ Model cached locally — use `--cache-dir models/cache` to skip re-download

**Phase 1: Data Extraction**
- ✓ OCR correction pipeline implemented (`src/data_extraction/ocr_corrector.py`)
- ✓ Publication parser implemented (`src/data_extraction/publication_parser.py`)
- ✓ Deduplicator implemented (`src/data_extraction/deduplicator.py`)
- ✓ Extraction script completed (`scripts/extract_publications.py`)
- ✓ PDF filtering strategy implemented (`data/oa_pdf_filter.txt`)
- ✓ Extraction completed: 28 new pairs, 1,589 total corpus

**Phase 2: Retrieval System** (Completed 2026-02-02)
- ✓ Lexicon builder (`src/retrieval/lexicon.py`)
- ✓ Embedder (`src/retrieval/embedder.py`) - sentence-transformers
- ✓ FAISS index builder (`src/retrieval/index_builder.py`)
- ✓ Retriever interface (`src/retrieval/retriever.py`)
- ✓ Build script (`scripts/build_index.py`)
- ✓ FAISS index built: 1,589 vectors, 384-dim, saved to `data/indices/`
- ✓ Lexicon loaded: ~95% coverage of proper nouns + Sumerograms
- ✓ Retrieval tested and working

**Phase 3: Model Fine-Tuning** (In Progress 2026-02-03)
- ✓ Lexicon Sumerogram resolution rewritten: 6 → 80 glossed forms, 157 → 222
  forms recognised.  Root cause: eBL lookup was keyed on `norm` (inflected);
  switched to `lexeme` (dictionary base form) + mimation stripping.
- ✓ `predict_with_generate` disabled during training-eval checkpoints.
  Checkpoint selection uses `eval_loss` only; beam-search generation reserved
  for post-training eval (`train.py:evaluate()`).
- ✓ `gradient_checkpointing: true` enabled as memory safety margin on A100.
- ✓ Dead code removed: `_make_compute_metrics`, unused `sacrebleu`/`numpy`
  imports in trainer.

**Extraction context (2026-02-02):**
Initial extraction run on all 952 PDFs yielded only 146 pairs (0.09% of target).
Investigation revealed root cause: `publications.csv` contains ALL cuneiform
publications (Hittite, Sumerian, Babylonian, Luvian, etc.), not just Old Assyrian.
The `has_akkadian` flag catches any cuneiform script, regardless of dialect.

**CRITICAL DECISION (2026-02-02): PDF Filtering Strategy**
After analysis, identified 194 Old Assyrian-specific PDFs out of 952 total (20%).
Decision: Filter extraction to Old Assyrian PDFs only, based on filename keywords.
PDF filter list saved to: `data/oa_pdf_filter.txt`

**Extraction Results (OA PDFs only):**
- ✓ Filtered extraction completed: 130 PDFs, 5,082 pages scanned
- ✓ Pairs extracted: **28 pairs** (after dedup)
- ✓ Combined corpus: **1,589 pairs** (1,561 existing + 28 new)
- ✓ Multiplier: **1.02x** (far below 2-3x target)
- ✓ Yield rate: 0.55% (28 / 5,082 pages)

**CRITICAL FINDING: Publication Extraction Not Viable**

Root cause analysis (see `docs/EXTRACTION_FINDINGS.md` for full details):
1. **Format mismatch:** OA scholarly publications don't use expected format
   - Parser expects: `transliteration: translation` blocks
   - Reality: catalogs, interlinear, commentary-heavy, separate sections
   - Only ~5% of pages contain usable parallel text
2. **OCR quality:** Structural corruption prevents reliable extraction
3. **Quality issues:** Of 28 extracted pairs, only ~10 (36%) are high quality

**DECISION: Proceed to Phase 2 with existing data (1,589 pairs)**

Rationale:
- Further parser work has diminishing returns (~10-15 more pairs maximum)
- Focus effort on model architecture + RAG instead of data extraction
- Akkademia achieved BLEU 36-37 with 56k pairs; we can reach BLEU 30+ with
  1.5k pairs + strong retrieval + careful augmentation
- Alternative data sources (ORACC API, digital repositories) can be explored
  in parallel with Phase 2-3

**Next step:** Submit training job to cluster: `./cluster/submit_job.sh train`

`train.slurm` requests 1× A100, 48 h walltime.  The job auto-builds the FAISS
index if missing, then runs `scripts/train.py --config configs/training.yaml`.

**Blockers:** None.

---

## Reference Benchmarks

| System | Corpus Size | BLEU | chrF++ | Geometric Mean | Notes |
|--------|-------------|------|--------|----------------|-------|
| **Akkademia (2023)** | 56k pairs | 36-37 | — | — | CNN-based, ORACC data |
| **Zero-shot ByT5-small** | 0 (zero-shot) | 0.00 | 4.88 | 0.00 | Actual baseline (2026-01-31) |
| **Target (this project)** | 1.6k pairs | 30+ | 50+ | ~39+ | Fine-tuned ByT5 + RAG + post-processing |

---

*This roadmap is a living document. Update as phases complete and blockers arise.*
