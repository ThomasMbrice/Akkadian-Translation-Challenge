#!/bin/bash
#SBATCH --job-name=akklang-predict
#SBATCH --partition=a100-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scratch/%u/Akkadian-Translation-Challenge/logs/slurm/predict_%j.out
#SBATCH --error=/gpfs/scratch/%u/Akkadian-Translation-Challenge/logs/slurm/predict_%j.err

set -euo pipefail

PROJECT_ROOT="/gpfs/scratch/tmbrice/Akkadian-Translation-Challenge"
VENV_DIR="$PROJECT_ROOT/venv"

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST started at $(date)"

mkdir -p "$PROJECT_ROOT/logs/slurm"

module load python3
source "$VENV_DIR/bin/activate"
echo "Python: $(python --version)"

python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"

if [ ! -d "$PROJECT_ROOT/models/byt5_finetuned/final" ]; then
    echo "ERROR: checkpoint not found at $PROJECT_ROOT/models/byt5_finetuned/final"
    exit 1
fi

echo "Running inference at $(date)"
python "$PROJECT_ROOT/scripts/predict.py" \
    --checkpoint "$PROJECT_ROOT/models/byt5_finetuned/final" \
    --output "$PROJECT_ROOT/submission.csv" \
    --max-length 512 \
    --num-beams 5

echo "Done at $(date)"
echo "Submission: $PROJECT_ROOT/submission.csv"
