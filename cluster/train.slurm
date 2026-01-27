#!/bin/bash
#SBATCH --job-name=akklang-train
#SBATCH --output=logs/slurm/train_%j.out
#SBATCH --error=logs/slurm/train_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu

# Akkadian NMT: ByT5 Fine-tuning with RAG
# Phase 3: Weeks 4-6
# Target: Validation BLEU > 25

set -e

echo "=========================================="
echo "Akkadian NMT - ByT5 Fine-tuning"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
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

# Create necessary directories
mkdir -p "$PROJECT_ROOT/logs/slurm"
mkdir -p "$PROJECT_ROOT/models"
mkdir -p "$PROJECT_ROOT/logs/training"

# Bind paths for Singularity
export SINGULARITY_BIND="$PROJECT_ROOT:/akklang"

# Build retrieval index if not exists
if [ ! -f "$PROJECT_ROOT/data/indices/faiss_index.bin" ]; then
    echo "Building FAISS index for RAG..."
    singularity exec --nv "$CONTAINER" \
        python /akklang/scripts/build_index.py \
        --corpus /akklang/data/processed/combined_corpus.csv \
        --output /akklang/data/indices/faiss_index.bin
    echo ""
fi

echo "Starting ByT5 fine-tuning..."
singularity exec --nv "$CONTAINER" \
    python /akklang/scripts/train.py \
    --config /akklang/configs/training.yaml

echo ""
echo "=========================================="
echo "Training completed!"
echo "End time: $(date)"
echo "Model checkpoint: $PROJECT_ROOT/models/byt5_finetuned/"
echo "Training logs: $PROJECT_ROOT/logs/training/"
echo "=========================================="
