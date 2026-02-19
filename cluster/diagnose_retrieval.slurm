#!/bin/bash
#SBATCH --job-name=diag_retrieval
#SBATCH --output=/gpfs/scratch/%u/Akkadian-Translation-Challenge/logs/slurm/train_nocontainer_%j.out
#SBATCH --error=/gpfs/scratch/%u/Akkadian-Translation-Challenge/logs/slurm/train_nocontainer_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=short

# ============================================================================
# Retrieval Diagnostic for Akkadian NMT RAG System
# ============================================================================

set -e  # Exit on error

# Project paths
PROJECT_ROOT=/gpfs/scratch/tmbrice/Akkadian-Translation-Challenge
cd "$PROJECT_ROOT" || exit 1

# Python environment
module purge
module load python/3.9.20
source .venv/bin/activate

# Ensure log directory exists
mkdir -p logs/slurm

# Run diagnostic
echo "=========================================="
echo "Retrieval Quality Diagnostic"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

python3 scripts/diagnose_retrieval.py --k 3

echo ""
echo "End time: $(date)"
echo "=========================================="

# Print summary file location
if [ -f logs/retrieval_diagnostic_summary.txt ]; then
    echo ""
    echo "Summary saved to: logs/retrieval_diagnostic_summary.txt"
    echo ""
    cat logs/retrieval_diagnostic_summary.txt
fi
