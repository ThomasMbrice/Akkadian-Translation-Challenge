#!/bin/bash
#SBATCH --job-name=akklang-test
#SBATCH --output=logs/slurm/test_%j.out
#SBATCH --error=logs/slurm/test_%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=a100

# Akkadian NMT: Container Test
# Verifies Singularity container and all dependencies

set -e

echo "=========================================="
echo "Akkadian NMT - Container Test"
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
    echo "=========================================="
    echo "ERROR: Container not found!"
    echo "=========================================="
    echo "Expected location: $CONTAINER"
    echo ""
    echo "Build it first with:"
    echo "  cd $PROJECT_ROOT"
    echo "  ./cluster/build_container.sh"
    echo ""
    exit 1
fi

echo "Container found: $CONTAINER"
echo "Container size: $(du -h $CONTAINER | cut -f1)"
echo ""

# Create log directory
mkdir -p "$PROJECT_ROOT/logs/slurm"

# Bind paths for Singularity
export SINGULARITY_BIND="$PROJECT_ROOT:/akklang"

echo "Running container tests..."
echo ""

# Run test with explicit error handling
if ! singularity exec --nv "$CONTAINER" python /akklang/scripts/test_container.py; then
    EXIT_CODE=$?
    echo ""
    echo "Container test failed with exit code $EXIT_CODE"
else
    EXIT_CODE=0
fi

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ CONTAINER TEST PASSED"
    echo "=========================================="
    echo ""
    echo "Container is ready! Next steps:"
    echo "  1. Run baseline: sbatch cluster/baseline.slurm"
    echo "  2. Extract data: sbatch cluster/extract_publications.slurm"
    echo "  3. Train model: sbatch cluster/train.slurm"
else
    echo "✗ CONTAINER TEST FAILED"
    echo "=========================================="
    echo ""
    echo "Check the output above for errors."
    echo "You may need to rebuild the container."
fi
echo ""
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
