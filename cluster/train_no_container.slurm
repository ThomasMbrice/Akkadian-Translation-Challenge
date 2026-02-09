#!/bin/bash
#SBATCH --job-name=akklang-train-nocontainer
#SBATCH --partition=a100-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/train_nocontainer_%j.out
#SBATCH --error=logs/slurm/train_nocontainer_%j.err

# Akkadian NMT: ByT5 Fine-tuning WITHOUT Singularity Container
# Uses cluster modules + venv instead

set -e

echo "=========================================="
echo "Akkadian NMT - ByT5 Fine-tuning (No Container)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'nvidia-smi unavailable')"
echo "Start time: $(date)"
echo ""

# Project paths
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/Akkadian-Translation-Challenge}"
VENV_DIR="$PROJECT_ROOT/venv"

echo "Project root: $PROJECT_ROOT"
echo ""

# Create necessary directories
mkdir -p "$PROJECT_ROOT/logs/slurm"
mkdir -p "$PROJECT_ROOT/models"
mkdir -p "$PROJECT_ROOT/logs/training"
mkdir -p "$PROJECT_ROOT/outputs"

# Load Python module
echo "Loading Python module..."
module load python3
python3 --version
echo ""

# Create and activate virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    echo "✓ Virtual environment created"
    echo ""
fi

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"
which python
python --version
echo ""

# Install/upgrade dependencies
echo "Installing dependencies..."
echo "(This will be slow on first run, cached on subsequent runs)"
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support (bundled CUDA libs)
echo "Installing PyTorch with CUDA 11.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project requirements
if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
    echo "Installing project requirements..."
    pip install -r "$PROJECT_ROOT/requirements.txt"
else
    echo "WARNING: requirements.txt not found, installing core packages manually"
    pip install transformers==4.36.0 accelerate==0.25.0 datasets==2.15.0
    pip install sentencepiece==0.1.99 protobuf==3.20.3
    pip install sacrebleu==2.3.1 evaluate==0.4.1
    pip install sentence-transformers==2.2.2 faiss-cpu==1.7.4
    pip install pandas==2.1.4 numpy==1.26.2 scikit-learn==1.3.2
    pip install tqdm==4.66.1 pyyaml==6.0.1
fi
echo "✓ Dependencies installed"
echo ""

# Test CUDA availability
echo "Testing CUDA availability..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Set environment variables
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export TRANSFORMERS_CACHE="$PROJECT_ROOT/.cache/huggingface"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"

# Create cache directories
mkdir -p "$TRANSFORMERS_CACHE"
mkdir -p "$TORCH_HOME"

# Verify combined corpus exists (prerequisite for training)
if [ ! -f "$PROJECT_ROOT/data/processed/combined_corpus.csv" ]; then
    echo "=========================================="
    echo "ERROR: Combined corpus not found!"
    echo "=========================================="
    echo "Expected: $PROJECT_ROOT/data/processed/combined_corpus.csv"
    echo ""
    echo "Run extract_publications job first, or check your data directory."
    echo ""
    exit 1
fi
echo "✓ Combined corpus found"
echo ""

# Build retrieval index if not exists
if [ ! -f "$PROJECT_ROOT/data/indices/faiss_index.bin" ]; then
    echo "Building FAISS index for RAG..."
    if ! python "$PROJECT_ROOT/scripts/build_index.py" \
        --corpus "$PROJECT_ROOT/data/processed/combined_corpus.csv" \
        --output "$PROJECT_ROOT/data/indices/faiss_index.bin"; then
        echo "ERROR: FAISS index building failed"
        exit 1
    fi

    # Verify index was created
    if [ ! -f "$PROJECT_ROOT/data/indices/faiss_index.bin" ]; then
        echo "ERROR: FAISS index was not created"
        exit 1
    fi
    echo "✓ FAISS index built"
    echo ""
fi

# Start training
echo "=========================================="
echo "Starting ByT5 fine-tuning..."
echo "=========================================="
echo ""

if ! python "$PROJECT_ROOT/scripts/train.py" \
    --config "$PROJECT_ROOT/configs/training.yaml"; then
    echo ""
    echo "=========================================="
    echo "ERROR: Training failed"
    echo "=========================================="
    exit 1
fi

# Verify model checkpoint was created
if [ ! -d "$PROJECT_ROOT/models/byt5_finetuned" ]; then
    echo ""
    echo "=========================================="
    echo "ERROR: Model checkpoint directory was not created"
    echo "=========================================="
    echo "Expected: $PROJECT_ROOT/models/byt5_finetuned/"
    echo ""
    exit 1
fi
echo "✓ Training completed"

echo ""
echo "=========================================="
echo "TRAINING JOB COMPLETED!"
echo "=========================================="
echo "End time: $(date)"
echo "Model checkpoint: $PROJECT_ROOT/models/byt5_finetuned/"
echo "Training logs: $PROJECT_ROOT/logs/training/"
echo ""
echo "Next steps:"
echo "  1. Check validation metrics in logs/training/"
echo "  2. Run inference: sbatch cluster/inference.slurm"
echo "=========================================="
