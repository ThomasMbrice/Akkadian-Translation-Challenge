# Development Guide

## Quick Start for New Claude Instances

When starting a new conversation on this project:

1. **Read this file first** to understand the project structure
2. **Read ARCHITECTURE.md** to understand the technical approach
3. **Read DATA_FORMATS.md** if working with data preprocessing
4. **Read ROADMAP.md** to see where we are in the timeline

## Project Structure

```
akklang/
├── docs/                          # Documentation (START HERE)
│   ├── README.md                  # Navigation hub
│   ├── ARCHITECTURE.md            # Technical architecture
│   ├── DATA_FORMATS.md            # Data specifications
│   ├── DEVELOPMENT_GUIDE.md       # This file
│   └── ROADMAP.md                 # Phases and milestones
│
├── data/                          # Data directory (gitignored)
│   ├── raw/                       # Competition data as-is
│   ├── processed/                 # Cleaned, aligned data
│   ├── augmented/                 # Synthetic augmentation
│   └── indices/                   # FAISS indices, lexicons
│
├── src/                           # Source code
│   ├── preprocessing/             # Data cleaning, normalization
│   │   ├── normalizer.py         # Strip notations, normalize gaps
│   │   ├── extractor.py          # Extract proper nouns, Sumerograms
│   │   └── aligner.py            # Sentence alignment
│   │
│   ├── data_extraction/           # Publication → parallel pairs
│   │   ├── ocr_corrector.py      # LLM-based OCR correction
│   │   ├── publication_parser.py # Extract from publications.csv
│   │   └── deduplicator.py       # Remove duplicate tablets
│   │
│   ├── retrieval/                 # RAG / Translation memory
│   │   ├── embedder.py           # Embed English translations
│   │   ├── index_builder.py      # Build FAISS index
│   │   ├── retriever.py          # Retrieve k-nearest neighbors
│   │   └── lexicon.py            # Proper noun/Sumerogram lookups
│   │
│   ├── modeling/                  # Model training and inference
│   │   ├── byt5_trainer.py       # ByT5 fine-tuning
│   │   ├── augmentation.py       # Back-translation, synthetic gaps
│   │   ├── context_assembler.py  # Combine RAG + lexicon + input
│   │   └── inference.py          # Generate translations
│   │
│   ├── evaluation/                # Metrics and analysis
│   │   ├── metrics.py            # BLEU, chrF++, proper noun accuracy
│   │   ├── error_analysis.py     # Analyze failure modes
│   │   └── validator.py          # Validate outputs
│   │
│   └── utils/                     # Shared utilities
│       ├── io.py                 # File I/O helpers
│       └── constants.py          # Competition formatting rules
│
├── scripts/                       # Executable scripts
│   ├── preprocess.py             # Run preprocessing pipeline
│   ├── extract_publications.py   # Extract from publications
│   ├── build_index.py            # Build retrieval index
│   ├── train.py                  # Train model
│   ├── inference.py              # Generate predictions
│   └── submit.py                 # Create submission file
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration.ipynb # Explore competition data
│   ├── 02_baseline.ipynb         # Zero-shot baseline
│   └── 03_error_analysis.ipynb   # Analyze model errors
│
├── tests/                         # Unit tests
│   ├── test_preprocessing.py     # Test preprocessing logic
│   ├── test_retrieval.py         # Test RAG components
│   └── test_modeling.py          # Test model utilities
│
├── configs/                       # Configuration files
│   ├── preprocessing.yaml        # Preprocessing parameters
│   ├── training.yaml             # Training hyperparameters
│   └── inference.yaml            # Inference settings
│
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
├── .gitignore                     # Git ignore rules
└── README.md                      # Brief project intro
```

## Development Workflow

### Phase 0: Baseline (Week 1)
**Goal:** Establish zero-shot baseline and submission pipeline

**Tasks:**
1. Download competition data to `data/raw/`
2. Implement basic preprocessing in `src/preprocessing/normalizer.py`
3. Test zero-shot translation with base ByT5 model
4. Generate baseline submission with `scripts/submit.py`
5. Record BLEU, chrF++ scores in `notebooks/02_baseline.ipynb`

**Success criteria:** Baseline submission uploaded, scores documented

---

### Phase 1: Data Extraction (Weeks 2-3)
**Goal:** Extract 2-3x more training data from publications

**Critical path:** Publication extraction is the highest-leverage activity

**Tasks:**
1. **OCR correction** (`src/data_extraction/ocr_corrector.py`)
   - Use LLM to fix OCR errors in publications.csv
   - Preserve formatting, transliterations, translations

2. **Publication parsing** (`src/data_extraction/publication_parser.py`)
   - Extract parallel Akkadian-Translation pairs
   - Match document IDs with published_texts.csv
   - Handle multi-language translations (French, German, Turkish)

3. **Translation to English** (inline in parser)
   - Use LLM to convert non-English → English
   - Preserve literal word-for-word style (not paraphrases)

4. **Sentence alignment** (`src/preprocessing/aligner.py`)
   - Split document-level pairs → sentence-level
   - Use Sentences_Oare_FirstWord_LinNum.csv as aid

5. **Deduplication** (`src/data_extraction/deduplicator.py`)
   - Same tablet appears in multiple publications
   - Keep highest-quality translation

**Quality checks:**
- Verify proper nouns match OA_Lexicon_eBL.csv
- Verify Sumerograms preserved correctly
- Sample 50 pairs for manual inspection

**Target:** 20-50k parallel sentence pairs in `data/processed/combined_corpus.csv`

---

### Phase 2: Retrieval System (Weeks 3-4)
**Goal:** Build translation memory and lexicon lookup

**Tasks:**
1. **Lexicon builder** (`src/retrieval/lexicon.py`)
   - Load OA_Lexicon_eBL.csv → proper noun dictionary
   - Load eBL_Dictionary.csv → Sumerogram mappings
   - Implement fuzzy matching for inflected forms

2. **Embedder** (`src/retrieval/embedder.py`)
   - Embed English side of corpus (no Akkadian embedder exists)
   - Use sentence-transformers (e.g., `all-MiniLM-L6-v2`)

3. **FAISS index builder** (`src/retrieval/index_builder.py`)
   - Build FAISS index over English embeddings
   - Save to `data/indices/faiss_index.bin`

4. **Retriever** (`src/retrieval/retriever.py`)
   - Given Akkadian input, embed English query (use simple heuristics)
   - Retrieve k=5 most similar (Akkadian, English) pairs
   - Return context for model

**Testing:**
- Query: "a-na A-šùr-i-mì-tí DUMU Ṣí-lí-{d}UTU qí-bi-ma"
- Expected retrieval: Similar letter opening formulas

---

### Phase 3: Model Fine-Tuning (Weeks 4-6)
**Goal:** Train ByT5 with RAG context

**Tasks:**
1. **Data augmentation** (`src/modeling/augmentation.py`)
   - Synthetic gaps: randomly mask spans with `<gap>` / `<big_gap>`
   - Back-translation: if cycle-consistency model available

2. **Context assembly** (`src/modeling/context_assembler.py`)
   - Format: Lexicon entries + retrieved examples + input
   - Keep total context under ByT5 max length (1024 bytes)

3. **Training** (`src/modeling/byt5_trainer.py`)
   - Base model: `google/byt5-base` or `google/byt5-large`
   - Framework: Hugging Face Transformers + Accelerate
   - Loss: Standard seq2seq cross-entropy
   - Optional: Multi-task with lemmatization / POS tagging

4. **Hyperparameter tuning**
   - Learning rate: 5e-5 to 1e-4
   - Batch size: 8-32 (depends on GPU)
   - Epochs: 5-10 (early stopping on validation)

**Config:** Store hyperparameters in `configs/training.yaml`

**Checkpoints:** Save to `models/byt5_finetuned/`

---

### Phase 4: Iteration (Weeks 6-8)
**Goal:** Error analysis, ensembling, final submission

**Tasks:**
1. **Error analysis** (`src/evaluation/error_analysis.py`)
   - Categorize errors: proper nouns, Sumerograms, morphology, fluency
   - Identify systematic failures

2. **Targeted fixes**
   - If proper noun accuracy low → improve lexicon lookup
   - If gap handling poor → more augmentation
   - If fluency poor → adjust generation parameters (beam search, temperature)

3. **Ensembling** (optional)
   - Train multiple models (different seeds, hyperparameters)
   - Ensemble predictions (e.g., majority vote, MBRT)

4. **Final submission**
   - Run inference on test.csv with `scripts/inference.py`
   - Generate submission with `scripts/submit.py`
   - Upload to competition

---

## Coding Conventions

### General
- Python 3.9+
- Type hints everywhere (`from typing import ...`)
- Docstrings for all public functions (Google style)
- Black formatting (line length 100)
- Use `pathlib.Path` for file paths, not strings

### Configuration
- All hyperparameters in YAML files under `configs/`
- Load with PyYAML or OmegaConf
- No hardcoded paths; use `data/`, `models/`, etc.

### Logging
- Use Python `logging` module (not print statements)
- Log level: INFO for key events, DEBUG for verbose

### Testing
- Write tests for data processing logic (preprocessing is error-prone)
- Use pytest
- Run tests with `pytest tests/`

---

## Environment Setup

### Dependencies
Core packages (see `requirements.txt` for full list):
- `transformers` - Hugging Face ByT5
- `datasets` - Data loading
- `faiss-cpu` or `faiss-gpu` - Similarity search
- `sentence-transformers` - Embedding English translations
- `sacrebleu`, `evaluate` - Metrics
- `pandas`, `numpy` - Data manipulation
- `pyyaml` - Config loading

### Installation
```bash
pip install -r requirements.txt
```

### GPU Requirements
- Training: Recommended 16GB+ VRAM (A100, V100, or equivalent)
- Inference: Can run on CPU, but GPU preferred

---

## Common Tasks

### Run preprocessing
```bash
python scripts/preprocess.py --config configs/preprocessing.yaml
```

### Extract publications
```bash
python scripts/extract_publications.py --output data/processed/extracted_pairs.csv
```

### Build retrieval index
```bash
python scripts/build_index.py --corpus data/processed/combined_corpus.csv --output data/indices/
```

### Train model
```bash
python scripts/train.py --config configs/training.yaml
```

### Generate predictions
```bash
python scripts/inference.py --model models/byt5_finetuned/ --input data/raw/test.csv --output predictions.csv
```

### Create submission
```bash
python scripts/submit.py --predictions predictions.csv --output submission.csv
```

---

## Debugging Tips

### Preprocessing issues
- Check for unhandled Unicode characters (Akkadian has many diacritics)
- Verify gap normalization logic with `tests/test_preprocessing.py`

### Retrieval not working
- Check embedding dimensions match FAISS index
- Verify proper nouns extracted correctly (capitalization matters)

### Model not learning
- Check learning rate (too high causes divergence)
- Verify loss decreasing over time
- Inspect predictions on validation set (are they empty? nonsense? close?)

### Out of memory
- Reduce batch size
- Use gradient accumulation
- Try `byt5-base` instead of `byt5-large`

---

## Key Principles

1. **Data quality > Model complexity**
   The 900 publications are the critical path. Extract them well.

2. **Preserve linguistic features**
   Proper nouns, Sumerograms, determinatives must survive preprocessing.

3. **Evaluation beyond BLEU**
   chrF++ more reliable for morphologically rich languages. Manual inspection essential.

4. **Competition constraints**
   Remember: test data is sentence-level, train is document-level. Alignment matters.

---

## Resources

- **Competition:** [Kaggle Deep Past Challenge](https://www.kaggle.com/competitions/the-deep-past)
- **OARE Database:** [https://oracc.museum.upenn.edu/oare/](https://oracc.museum.upenn.edu/oare/)
- **CDLI:** [https://cdli.mpiwg-berlin.mpg.de/](https://cdli.mpiwg-berlin.mpg.de/)
- **eBL Dictionary:** [https://www.ebl.lmu.de/](https://www.ebl.lmu.de/)
- **ByT5 Paper:** [https://arxiv.org/abs/2105.13626](https://arxiv.org/abs/2105.13626)

---

*For technical architecture details, see ARCHITECTURE.md. For data format specifications, see DATA_FORMATS.md.*
