# Akkadian NMT Project Documentation

**Welcome!** This is the documentation hub for the Old Assyrian cuneiform → English translation project.

## For New Claude Instances

If you're a new Claude instance starting work on this project:

1. **Start here** → Read this page for navigation
2. **Understand the architecture** → [ARCHITECTURE.md](ARCHITECTURE.md)
3. **Learn the data formats** → [DATA_FORMATS.md](DATA_FORMATS.md)
4. **See how to develop** → [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md)
5. **Check project status** → [ROADMAP.md](ROADMAP.md)

---

## Project Context

**What:** Neural machine translation from Old Assyrian (transliterated cuneiform) → English

**Why:** Unlock 10,000+ untranslated Bronze Age tablets documenting ancient Mesopotamian trade networks

**Competition:** [Kaggle Deep Past Challenge](https://www.kaggle.com/competitions/the-deep-past)

**Timeline:** 8 weeks (Phases 0-4)

**Current Phase:** [TO BE UPDATED]

---

## Documentation Guide

### [ARCHITECTURE.md](ARCHITECTURE.md)
**Read when:** You need to understand the technical approach

**Contains:**
- Problem statement (low-resource, morphologically complex Semitic language)
- System architecture diagram (preprocessing → retrieval → generation)
- Key technical decisions (ByT5, RAG, data augmentation)
- Data pipeline (sources, processing, target corpus size)
- Evaluation targets (BLEU, chrF++, proper noun accuracy)

**Read this if you're asking:** "How does the system work?"

---

### [DATA_FORMATS.md](DATA_FORMATS.md)
**Read when:** You're working on data preprocessing or extraction

**Contains:**
- Competition data files (train.csv, test.csv, publications.csv, etc.)
- Preprocessing rules (what to remove, preserve, normalize)
- Data extraction workflow (publications → parallel pairs)
- Example transformations
- File organization in `data/`

**Read this if you're asking:** "How should I process this data?"

---

### [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md)
**Read when:** You're implementing code or debugging

**Contains:**
- Project structure (where files live)
- Development workflow (phase-by-phase tasks)
- Coding conventions (Python style, config management)
- Environment setup (dependencies, GPU requirements)
- Common tasks (running scripts, training models)
- Debugging tips

**Read this if you're asking:** "How do I work on this project?"

---

### [ROADMAP.md](ROADMAP.md)
**Read when:** You need to know where we are and what's next

**Contains:**
- Phase 0: Baseline (Week 1)
- Phase 1: Data Extraction (Weeks 2-3) ← **Critical path**
- Phase 2: Retrieval System (Weeks 3-4)
- Phase 3: Model Fine-Tuning (Weeks 4-6)
- Phase 4: Post-Processing (Week 6) ← **New: Mini-LLM refinement**
- Phase 5: Iteration & Submission (Weeks 7-8)
- Current status and blockers

**Read this if you're asking:** "What should I work on next?"

---

## Quick Reference

### Key Concepts

**Old Assyrian:** Bronze Age dialect of Akkadian (2000-1700 BCE), used by Mesopotamian merchants

**Transliteration:** Cuneiform → Latin alphabet (e.g., `a-na A-šùr-i-mì-tí DUMU qí-bi-ma`)

**Sumerogram:** All-caps logogram (e.g., `DUMU` = "son", `KÙ.BABBAR` = "silver")

**Determinative:** Semantic classifier in braces (e.g., `{d}` = divine, `{f}` = female)

**ByT5:** Byte-level T5 model, handles multilingual/mixed-script text without tokenization issues

**RAG (Retrieval-Augmented Generation):** Translation memory that provides similar examples as context

### Critical Numbers

| Metric | Value |
|--------|-------|
| Competition texts | ~8,000 |
| Texts with translations | ~4,000 (50%) |
| Publications to extract | 900 PDFs |
| Target corpus size | 20-50k sentence pairs (actual: 1.6k) |
| Baseline geometric mean (zero-shot) | 0.00 |
| Target geometric mean (fine-tuned + post-processing) | ~39+ |
| **Competition scoring** | **sqrt(BLEU × chrF)** |

### Critical Path

```
Publication extraction → Corpus size → Model performance
```

Weeks 2-3 (data extraction) are the **highest-leverage activities**. Every additional parallel pair improves model performance.

---

## File Structure Overview

```
akklang/
├── docs/              # You are here!
├── data/              # Competition data (gitignored)
├── src/               # Source code
│   ├── preprocessing/
│   ├── data_extraction/  ← Critical path work happens here
│   ├── retrieval/
│   ├── modeling/
│   └── evaluation/
├── scripts/           # Executable pipelines
├── notebooks/         # Exploratory analysis
├── configs/           # YAML configuration files
└── tests/             # Unit tests
```

---

## Resources

### Competition
- [Deep Past Challenge (Kaggle)](https://www.kaggle.com/competitions/the-deep-past)
- [OARE Database](https://oracc.museum.upenn.edu/oare/)
- [CDLI (Cuneiform Digital Library)](https://cdli.mpiwg-berlin.mpg.de/)
- [eBL (electronic Babylonian Library)](https://www.ebl.lmu.de/)

### Technical
- [ByT5 Paper (Xue et al., 2021)](https://arxiv.org/abs/2105.13626)
- [Akkademia (Gordin et al., 2023)](https://arxiv.org/abs/2310.12715) - BLEU 36-37 baseline
- [BabyLemmatizer](https://github.com/gaigutherz/BabyLemmatizer) - Akkadian morphology
- [SLAB-NLP/Akk](https://github.com/SLAB-NLP/Akk) - Akkadian BERT

---

## Getting Help

When asking for help or context in a new conversation:

1. **Specify what you're working on:**
   "I'm working on data extraction (Phase 1, Week 2)"

2. **Reference relevant docs:**
   "I've read ARCHITECTURE.md and DATA_FORMATS.md"

3. **Describe the issue:**
   "OCR correction is producing nonsense for French translations"

4. **Include context:**
   "See `src/data_extraction/ocr_corrector.py:45`"

This helps future Claude instances pick up where you left off!

---

## Updating Documentation

As the project progresses:

- **Update ROADMAP.md** when completing phases or hitting blockers
- **Update this README.md** if adding new documentation files
- **Update DEVELOPMENT_GUIDE.md** if adding new scripts or changing workflow
- **Update DATA_FORMATS.md** if discovering new data quirks

Keep docs synchronized with code!

---

**Happy translating!**
