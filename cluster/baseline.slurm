#!/bin/bash
#SBATCH --job-name=akklang-baseline
#SBATCH --output=logs/slurm/baseline_%j.out
#SBATCH --error=logs/slurm/baseline_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a100

# Akkadian NMT: Zero-shot ByT5 Baseline
# Phase 0: Week 1
# Expected: BLEU 5-10, chrF++ 20-30

set -e

echo "=========================================="
echo "Akkadian NMT - Zero-Shot Baseline"
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

# Bind paths for Singularity
export SINGULARITY_BIND="$PROJECT_ROOT:/akklang"

# Step 1: Preprocessing
echo "Running preprocessing..."
if ! singularity exec --nv "$CONTAINER" \
    python /akklang/scripts/preprocess.py \
    --config /akklang/configs/preprocessing.yaml; then
    echo "ERROR: Preprocessing failed"
    exit 1
fi

# Verify preprocessing output
if [ ! -f "$PROJECT_ROOT/data/processed/train_sentences.csv" ]; then
    echo "ERROR: Preprocessing did not create train_sentences.csv"
    exit 1
fi
echo "✓ Preprocessing completed"

# Step 2: Zero-shot baseline evaluation
echo ""
echo "Running zero-shot baseline evaluation..."
if ! singularity exec --nv "$CONTAINER" \
    python /akklang/scripts/baseline.py \
    --model google/byt5-base \
    --data /akklang/data/processed/train_sentences.csv \
    --output /akklang/outputs/baseline_results.json; then
    echo "ERROR: Baseline evaluation failed"
    exit 1
fi

# Verify baseline results
if [ ! -f "$PROJECT_ROOT/outputs/baseline_results.json" ]; then
    echo "ERROR: Baseline evaluation did not create results file"
    exit 1
fi
echo "✓ Baseline evaluation completed"

# Step 3: Generate test predictions
echo ""
echo "Generating test predictions..."
if ! singularity exec --nv "$CONTAINER" \
    python /akklang/scripts/inference.py \
    --config /akklang/configs/inference.yaml \
    --model google/byt5-base \
    --zero-shot; then
    echo "ERROR: Test prediction generation failed"
    exit 1
fi

# Verify predictions
if [ ! -f "$PROJECT_ROOT/predictions.csv" ]; then
    echo "ERROR: Inference did not create predictions.csv"
    exit 1
fi
echo "✓ Test predictions generated"

echo ""
echo "=========================================="
echo "Baseline job completed!"
echo "End time: $(date)"
echo "Results: $PROJECT_ROOT/outputs/baseline_results.json"
echo "Submission: $PROJECT_ROOT/predictions.csv"
echo "=========================================="
