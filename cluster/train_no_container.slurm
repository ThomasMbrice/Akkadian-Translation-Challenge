#!/bin/bash
#SBATCH --job-name=akklang-train
#SBATCH --partition=a100-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/scratch/%u/Akkadian-Translation-Challenge/logs/slurm/train_nocontainer_%j.out
#SBATCH --error=/gpfs/scratch/%u/Akkadian-Translation-Challenge/logs/slurm/train_nocontainer_%j.err

set -euo pipefail

PROJECT_ROOT="/gpfs/scratch/tmbrice/Akkadian-Translation-Challenge"
VENV_DIR="$PROJECT_ROOT/venv"

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST started at $(date)"
echo "Project root: $PROJECT_ROOT"

mkdir -p "$PROJECT_ROOT/logs/slurm" "$PROJECT_ROOT/logs/training" \
         "$PROJECT_ROOT/models" "$PROJECT_ROOT/outputs" \
         "$PROJECT_ROOT/.cache/huggingface" "$PROJECT_ROOT/.cache/torch"

# Python environment
module load python3

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
echo "Python: $(python --version)"

# Dependencies
pip install -q --upgrade pip setuptools wheel
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -q -r "$PROJECT_ROOT/requirements.txt"

# Verify CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Environment
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"

# Prerequisites
if [ ! -f "$PROJECT_ROOT/data/processed/combined_corpus.csv" ]; then
    echo "ERROR: combined_corpus.csv not found at $PROJECT_ROOT/data/processed/"
    exit 1
fi

# Train
echo "Starting training at $(date)"
python "$PROJECT_ROOT/scripts/train.py" --config "$PROJECT_ROOT/configs/training.yaml"

if [ ! -d "$PROJECT_ROOT/models/byt5_finetuned" ]; then
    echo "ERROR: model checkpoint not created"
    exit 1
fi

echo "Training complete at $(date)"
echo "Model: $PROJECT_ROOT/models/byt5_finetuned/"
