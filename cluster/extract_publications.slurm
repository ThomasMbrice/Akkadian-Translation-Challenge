#!/bin/bash
#SBATCH --job-name=akklang-extract
#SBATCH --output=logs/slurm/extract_%j.out
#SBATCH --error=logs/slurm/extract_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=cpu

# Akkadian NMT: Publication Data Extraction
# Phase 1: Weeks 2-3
# Critical Path: Extract 20-50k parallel pairs from 900 publications

set -e

echo "=========================================="
echo "Akkadian NMT - Publication Extraction"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Project paths
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/akklang}"
CONTAINER="$PROJECT_ROOT/akklang.sif"

# Check container exists
if [ ! -f "$CONTAINER" ]; then
    echo "Error: Container not found at $CONTAINER"
    echo "Build it first: ./cluster/build_container.sh"
    exit 1
fi

# Create log directory
mkdir -p "$PROJECT_ROOT/logs/slurm"
mkdir -p "$PROJECT_ROOT/data/processed"

# Bind paths for Singularity
export SINGULARITY_BIND="$PROJECT_ROOT:/akklang"

echo "Extracting parallel pairs from publications.csv..."
echo "Target: 20-50k pairs (2-3x multiplication)"
echo ""

singularity exec "$CONTAINER" \
    python /akklang/scripts/extract_publications.py \
    --publications /akklang/data/raw/deep-past-initiative-machine-translation/publications.csv \
    --published-texts /akklang/data/raw/deep-past-initiative-machine-translation/published_texts.csv \
    --sentence-alignment /akklang/data/raw/deep-past-initiative-machine-translation/Sentences_Oare_FirstWord_LinNum.csv \
    --lexicon /akklang/data/raw/deep-past-initiative-machine-translation/OA_Lexicon_eBL.csv \
    --output /akklang/data/processed/extracted_corpus.csv \
    --workers 16

echo ""
echo "Deduplicating corpus..."
singularity exec "$CONTAINER" \
    python /akklang/scripts/deduplicate.py \
    --input /akklang/data/processed/extracted_corpus.csv \
    --output /akklang/data/processed/deduplicated_corpus.csv

echo ""
echo "Combining with train.csv..."
singularity exec "$CONTAINER" \
    python /akklang/scripts/combine_corpus.py \
    --train /akklang/data/raw/deep-past-initiative-machine-translation/train.csv \
    --extracted /akklang/data/processed/deduplicated_corpus.csv \
    --output /akklang/data/processed/combined_corpus.csv

echo ""
echo "=========================================="
echo "Extraction completed!"
echo "End time: $(date)"
echo ""
echo "Results:"
echo "  - Extracted corpus: $PROJECT_ROOT/data/processed/extracted_corpus.csv"
echo "  - Deduplicated: $PROJECT_ROOT/data/processed/deduplicated_corpus.csv"
echo "  - Combined corpus: $PROJECT_ROOT/data/processed/combined_corpus.csv"
echo ""
echo "Next steps:"
echo "  1. Run quality analysis: python scripts/analyze_corpus.py"
echo "  2. Build retrieval index: sbatch cluster/train.slurm"
echo "=========================================="
