#!/bin/bash
#
# Helper script to submit jobs to the cluster
# Usage: ./submit_job.sh [baseline|train|inference|extract]
#

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Create logs directory
mkdir -p logs/slurm

if [ $# -eq 0 ]; then
    echo "Usage: $0 [test|baseline|train|inference|extract]"
    echo ""
    echo "Jobs:"
    echo "  test      - Test container setup (~15min, recommended first!)"
    echo "  baseline  - Run zero-shot baseline (Phase 0, ~2h)"
    echo "  extract   - Extract publications (Phase 1, ~24h, CPU-only)"
    echo "  train     - Fine-tune ByT5 (Phase 3, ~48h, requires GPU)"
    echo "  inference - Generate test predictions (Phase 4, ~1h)"
    exit 1
fi

JOB_TYPE=$1

case $JOB_TYPE in
    test)
        echo "Submitting container test job..."
        sbatch cluster/test.slurm
        ;;
    baseline)
        echo "Submitting baseline job..."
        sbatch cluster/baseline.slurm
        ;;
    extract)
        echo "Submitting publication extraction job..."
        sbatch cluster/extract_publications.slurm
        ;;
    train)
        echo "Submitting training job..."
        sbatch cluster/train.slurm
        ;;
    inference)
        echo "Submitting inference job..."
        sbatch cluster/inference.slurm
        ;;
    *)
        echo "Error: Unknown job type '$JOB_TYPE'"
        echo "Valid options: test, baseline, extract, train, inference"
        exit 1
        ;;
esac

echo ""
echo "Job submitted! Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/slurm/*.out"
