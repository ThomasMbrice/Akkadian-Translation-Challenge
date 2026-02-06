#!/bin/bash
#SBATCH --job-name=akklang-inference
#SBATCH --output=logs/slurm/inference_%j.out
#SBATCH --error=logs/slurm/inference_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a100

# Akkadian NMT: Generate test predictions
# Phase 4: Weeks 6-8

set -e

echo "=========================================="
echo "Akkadian NMT - Inference"
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

# Verify trained model exists
if [ ! -d "$PROJECT_ROOT/models/byt5_finetuned" ]; then
    echo "ERROR: Trained model not found. Run train.slurm first."
    exit 1
fi

# Step 1: Generate predictions
echo "Generating test set predictions..."
if ! singularity exec --nv "$CONTAINER" \
    python /akklang/scripts/inference.py \
    --config /akklang/configs/inference.yaml; then
    echo "ERROR: Inference failed"
    exit 1
fi

# Verify predictions were created
if [ ! -f "$PROJECT_ROOT/predictions.csv" ]; then
    echo "ERROR: Inference did not create predictions.csv"
    exit 1
fi
echo "✓ Predictions generated"

# Step 2: Create submission file
echo ""
echo "Creating competition submission..."
if ! singularity exec --nv "$CONTAINER" \
    python /akklang/scripts/submit.py \
    --predictions /akklang/predictions.csv \
    --output /akklang/submission.csv; then
    echo "ERROR: Submission file creation failed"
    exit 1
fi

# Verify submission file was created
if [ ! -f "$PROJECT_ROOT/submission.csv" ]; then
    echo "ERROR: submit.py did not create submission.csv"
    exit 1
fi
echo "✓ Submission file created"

echo ""
echo "=========================================="
echo "Inference completed!"
echo "End time: $(date)"
echo "Submission file: $PROJECT_ROOT/submission.csv"
echo ""
echo "Upload to Kaggle with:"
echo "  kaggle competitions submit -c deep-past-initiative-machine-translation -f submission.csv -m 'ByT5 with RAG'"
echo "=========================================="
