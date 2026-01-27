#!/bin/bash
#
# Build Singularity container on cluster
# Usage: ./build_container.sh
#

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Building Singularity container..."
echo "This may take 10-20 minutes..."

# Build container (requires sudo or fakeroot)
singularity build --fakeroot akklang.sif singularity.def

if [ $? -eq 0 ]; then
    echo "✓ Container built successfully: akklang.sif"
    echo ""
    echo "Test the container with:"
    echo "  singularity exec --nv akklang.sif python -c 'import torch; print(torch.cuda.is_available())'"
else
    echo "✗ Container build failed"
    exit 1
fi
