# Akkadian NMT: Old Assyrian → English Translation

Neural machine translation system for transliterated Old Assyrian cuneiform → English, built for the [Kaggle Deep Past Challenge](https://www.kaggle.com/competitions/the-deep-past).

## Goal

Unlock 10,000+ untranslated Bronze Age tablets documenting ancient Mesopotamian trade networks (2000-1700 BCE). These texts are the unfiltered voices of ancient merchants and their families—letters, contracts, invoices—sitting untranslated in museum collections.

## Approach

- **Model:** ByT5 (byte-level T5) fine-tuned on Old Assyrian-English parallel pairs
- **Data:** Extract 20-50k training pairs from 900 scholarly publications via OCR + LLM alignment
- **Retrieval:** Translation memory (RAG) + lexicon lookup for proper nouns and Sumerograms
- **Evaluation:** BLEU, chrF++, proper noun accuracy

**Target:** BLEU 30+ (competitive with state-of-the-art Akkademia baseline of BLEU 36-37)

## Quick Start

### For New Claude Instances
**Start here:** [docs/README.md](docs/README.md)

Read in order:
1. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - System design
2. [docs/DATA_FORMATS.md](docs/DATA_FORMATS.md) - Data specifications
3. [docs/DEVELOPMENT_GUIDE.md](docs/DEVELOPMENT_GUIDE.md) - How to develop
4. [docs/ROADMAP.md](docs/ROADMAP.md) - Project phases and status

### For Humans
```bash
# Install dependencies
pip install -r requirements.txt

# Run preprocessing
python scripts/preprocess.py --config configs/preprocessing.yaml

# Extract publications (critical path!)
python scripts/extract_publications.py

# Build retrieval index
python scripts/build_index.py

# Train model
python scripts/train.py --config configs/training.yaml

# Generate predictions
python scripts/inference.py --model models/byt5_finetuned/ --input data/raw/test.csv
```

## Project Structure

```
akklang/
├── docs/              # Documentation (START HERE)
├── data/              # Competition data (gitignored)
├── src/               # Source code
│   ├── preprocessing/
│   ├── data_extraction/  # Critical path: extract 900 publications
│   ├── retrieval/        # RAG + lexicon lookup
│   ├── modeling/         # ByT5 training + inference
│   └── evaluation/       # Metrics
├── scripts/           # Executable pipelines
├── notebooks/         # Exploratory analysis
├── configs/           # YAML configs
└── tests/             # Unit tests
```

## Critical Path

```
Publication extraction → Corpus size → Model performance
```

The 900 scholarly publications are the data multiplier. Extracting high-quality parallel pairs from these PDFs is the highest-leverage activity.

## Context

**Language:** Old Assyrian (2000-1700 BCE), an early dialect of Akkadian, the oldest documented Semitic language

**Challenge:** Low-resource (~8k texts, 50% translated), morphologically complex (single words encode full English clauses), no native speakers for evaluation

**Impact:** Blueprint for translating thousands of endangered and overlooked languages—ancient and modern—that the AI age has yet to reach

## Resources

- [Deep Past Challenge (Kaggle)](https://www.kaggle.com/competitions/the-deep-past)
- [OARE Database](https://oracc.museum.upenn.edu/oare/)
- [ByT5 Paper](https://arxiv.org/abs/2105.13626)
- [Akkademia Baseline](https://arxiv.org/abs/2310.12715) (BLEU 36-37)

---

**For detailed documentation, see [docs/README.md](docs/README.md)**
