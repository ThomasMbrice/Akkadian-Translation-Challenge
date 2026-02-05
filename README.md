# Akkadian NMT: Old Assyrian → English Translation

Neural machine translation system for transliterated Old Assyrian cuneiform → English, built for the [Kaggle Deep Past Challenge](https://www.kaggle.com/competitions/the-deep-past).

## Goal

Unlock 10,000+ untranslated Bronze Age tablets documenting ancient Mesopotamian trade networks (2000-1700 BCE). These texts are the unfiltered voices of ancient merchants and their families—letters, contracts, invoices—sitting untranslated in museum collections.

## Approach

- **Model:** ByT5 (byte-level T5) fine-tuned on Old Assyrian-English parallel pairs
- **Data:** Extract 20-50k training pairs from 900 scholarly publications via OCR + LLM alignment
- **Retrieval:** Translation memory (RAG) + lexicon lookup for proper nouns and Sumerograms
- **Post-processing:** Mini-LLM refinement (Phi-3/Llama) for output quality
- **Evaluation:** Geometric mean of BLEU and chrF (competition metric)

**Target:** Geometric mean ~39+ (sqrt(BLEU × chrF), competitive with Akkademia baseline)

## Quick Start

### For New Claude Instances
**Start here:** [docs/README.md](docs/README.md)

Read in order:
1. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - System design
2. [docs/DATA_FORMATS.md](docs/DATA_FORMATS.md) - Data specifications
3. [docs/DEVELOPMENT_GUIDE.md](docs/DEVELOPMENT_GUIDE.md) - How to develop
4. [docs/ROADMAP.md](docs/ROADMAP.md) - Project phases and status
5. [docs/CLUSTER_SETUP.md](docs/CLUSTER_SETUP.md) - GPU cluster setup

### For GPU Cluster (Recommended)
```bash
# 1. Upload to cluster
rsync -avz akklang/ user@cluster:/home/user/akklang/

# 2. Build container (one-time, ~15 min)
ssh user@cluster
cd ~/akklang
./cluster/build_container.sh

# 3. Run pipeline
./cluster/submit_job.sh baseline   # Phase 0: Week 1
./cluster/submit_job.sh extract    # Phase 1: Weeks 2-3
./cluster/submit_job.sh train      # Phase 3: Weeks 4-6
./cluster/submit_job.sh inference  # Phases 4-5: Weeks 6-8
```

See [docs/CLUSTER_SETUP.md](docs/CLUSTER_SETUP.md) for full instructions.

### For Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run preprocessing
python scripts/preprocess.py --config configs/preprocessing.yaml

# Run baseline (use --cache-dir to cache the model and skip re-downloading)
python scripts/baseline.py --model google/byt5-small \
    --test-file data/processed/train_sentences.csv \
    --output outputs/baseline \
    --sample 100 \
    --cache-dir models/cache

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

## Critical Path (Updated 2026-02-02)

```
Publication extraction → Corpus size → Model performance
```

**Update:** After implementing and testing publication extraction (Phase 1), we discovered that scholarly OA publications don't contain parallel text in extractable formats. Only 28 usable pairs were extracted from 5,082 pages (0.55% yield).

**Revised strategy:** Proceed with existing 1,589 pairs. Focus on strong retrieval (RAG) + data augmentation + model architecture rather than corpus expansion. See `docs/EXTRACTION_FINDINGS.md` for full analysis.

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
