#!/bin/bash
#SBATCH --job-name=akklang-eval
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scratch/%u/Akkadian-Translation-Challenge/logs/slurm/eval_nocontainer_%j.out
#SBATCH --error=/gpfs/scratch/%u/Akkadian-Translation-Challenge/logs/slurm/eval_nocontainer_%j.err

# Re-evaluate existing checkpoint without retraining.
# Bumps max_output_length to 512 to test whether output truncation was the bottleneck.

set -euo pipefail

PROJECT_ROOT="/gpfs/scratch/tmbrice/Akkadian-Translation-Challenge"
VENV_DIR="$PROJECT_ROOT/venv"
CHECKPOINT="$PROJECT_ROOT/models/byt5_finetuned/final"

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST started at $(date)"
echo "Project root: $PROJECT_ROOT"
echo "Checkpoint:   $CHECKPOINT"

mkdir -p "$PROJECT_ROOT/logs/slurm" "$PROJECT_ROOT/logs/training"

# Python environment
module load python3
source "$VENV_DIR/bin/activate"
echo "Python: $(python --version)"

# Verify CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Environment
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"

# Prerequisites
if [ ! -f "$PROJECT_ROOT/data/processed/combined_corpus.csv" ]; then
    echo "ERROR: combined_corpus.csv not found"
    exit 1
fi

if [ ! -d "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found at $CHECKPOINT"
    exit 1
fi

# Run inference-only evaluation with max_output_length=512
echo "Starting eval-only inference at $(date)"
python "$PROJECT_ROOT/scripts/train.py" \
    --config "$PROJECT_ROOT/configs/training.yaml" \
    --eval-only "$CHECKPOINT" \
    --max-output-length 512

echo "Eval complete at $(date)"
echo "Results: $PROJECT_ROOT/models/byt5_finetuned/eval_results.json"
