# GPU Cluster Setup for Akkadian NMT

This directory contains scripts for running the Akkadian NMT pipeline on a remote GPU cluster using Singularity containers and SLURM.

## Quick Start

```bash
# 1. Upload project to cluster
rsync -avz --exclude='*.sif' --exclude='data/' akklang/ user@cluster:/home/user/akklang/

# 2. SSH to cluster
ssh user@cluster

# 3. Build container (one-time setup, ~15 minutes)
cd ~/akklang
./cluster/build_container.sh

# 4. Submit baseline job
./cluster/submit_job.sh baseline

# 5. Monitor job
squeue -u $USER
tail -f logs/slurm/baseline_*.out
```

---

## Infrastructure

### Singularity Container (`singularity.def`)

**Base:** NVIDIA CUDA 11.8 with cuDNN 8
**Python:** 3.10 via Miniconda
**Key packages:**
- PyTorch 2.1+ with CUDA 11.8
- Hugging Face Transformers (ByT5)
- FAISS-GPU for vector search
- Sentence transformers for embeddings

**Build command:**
```bash
singularity build --fakeroot akklang.sif singularity.def
```

**Test command:**
```bash
singularity exec --nv akklang.sif python -c "import torch; print(torch.cuda.is_available())"
```

---

## SLURM Jobs

All jobs use the `akklang.sif` container and bind the project directory to `/akklang` inside the container.

### 1. Baseline (`baseline.slurm`)

**Phase:** 0 (Week 1)
**Duration:** ~2 hours
**Resources:** 1 GPU, 32GB RAM, 4 CPUs
**Purpose:** Zero-shot ByT5 baseline

```bash
sbatch cluster/baseline.slurm
```

**Outputs:**
- `outputs/baseline_results.json` - Metrics (BLEU, chrF++)
- `predictions.csv` - Test set predictions

**Expected scores:**
| Metric | Target |
|--------|--------|
| BLEU | 5-10 |
| chrF++ | 20-30 |

---

### 2. Extract Publications (`extract_publications.slurm`)

**Phase:** 1 (Weeks 2-3)
**Duration:** ~24 hours
**Resources:** 16 CPUs, 128GB RAM (CPU-only)
**Purpose:** Extract 20-50k parallel pairs from 900 publications

```bash
sbatch cluster/extract_publications.slurm
```

**Outputs:**
- `data/processed/extracted_corpus.csv` - Raw extraction
- `data/processed/deduplicated_corpus.csv` - After deduplication
- `data/processed/combined_corpus.csv` - Merged with train.csv

**Critical path:** This step multiplies training data by 2-3x.

---

### 3. Training (`train.slurm`)

**Phase:** 3 (Weeks 4-6)
**Duration:** ~48 hours
**Resources:** 1 A100 GPU, 64GB RAM, 8 CPUs
**Purpose:** Fine-tune ByT5 with RAG

```bash
sbatch cluster/train.slurm
```

**Outputs:**
- `models/byt5_finetuned/` - Model checkpoints
- `logs/training/` - Training logs (loss, metrics)

**Target validation BLEU:** 25+

**Hyperparameters** (see `configs/training.yaml`):
- Learning rate: 5e-5
- Batch size: 8 per device, 2 gradient accumulation steps
- Epochs: 10
- RAG: Top-3 retrieved examples

---

### 4. Inference (`inference.slurm`)

**Phase:** 4 (Weeks 6-8)
**Duration:** ~1 hour
**Resources:** 1 GPU, 32GB RAM, 4 CPUs
**Purpose:** Generate test set predictions for submission

```bash
sbatch cluster/inference.slurm
```

**Outputs:**
- `predictions.csv` - Model predictions
- `submission.csv` - Competition-ready submission

**Upload to Kaggle:**
```bash
kaggle competitions submit \
  -c deep-past-initiative-machine-translation \
  -f submission.csv \
  -m "ByT5 + RAG, BLEU XX"
```

---

## Job Management

### Submit jobs
```bash
./cluster/submit_job.sh baseline   # Phase 0
./cluster/submit_job.sh extract    # Phase 1
./cluster/submit_job.sh train      # Phase 3
./cluster/submit_job.sh inference  # Phase 4
```

### Monitor jobs
```bash
# List your jobs
squeue -u $USER

# Show job details
scontrol show job <job_id>

# Cancel job
scancel <job_id>

# View live output
tail -f logs/slurm/baseline_*.out

# Check GPU usage (on compute node)
nvidia-smi
```

### Resource requests

| Job | Time | GPU | CPUs | RAM | Notes |
|-----|------|-----|------|-----|-------|
| Baseline | 2h | 1x V100 | 4 | 32GB | Zero-shot |
| Extract | 24h | None | 16 | 128GB | CPU-heavy |
| Train | 48h | 1x A100 | 8 | 64GB | Prefers A100 |
| Inference | 1h | 1x V100 | 4 | 32GB | Any GPU |

---

## Directory Structure

```
akklang/
├── singularity.def         # Container definition
├── akklang.sif            # Built container (gitignored)
├── cluster/
│   ├── README.md          # This file
│   ├── build_container.sh # Build Singularity image
│   ├── submit_job.sh      # Job submission helper
│   ├── baseline.slurm     # Phase 0: Baseline
│   ├── extract_publications.slurm  # Phase 1: Data
│   ├── train.slurm        # Phase 3: Training
│   └── inference.slurm    # Phase 4: Submission
├── logs/slurm/            # Job outputs (auto-created)
├── data/                  # Data files (sync to cluster)
├── models/                # Model checkpoints (large!)
└── configs/               # YAML configs
```

---

## Cluster-Specific Adjustments

### SLURM Partition Names

Update `#SBATCH --partition=` in `.slurm` files if your cluster uses different names:

```bash
# Common partition names:
gpu       # GPU partition (baseline, train, inference)
cpu       # CPU partition (extract)
a100      # A100-specific partition
v100      # V100-specific partition
short     # Short jobs (<4h)
long      # Long jobs (>24h)
```

### GPU Types

Update `#SBATCH --gres=gpu:` to request specific GPUs:

```bash
gpu:1         # Any GPU
gpu:v100:1    # V100 (16GB VRAM)
gpu:a100:1    # A100 (40GB VRAM, preferred)
gpu:a100_80:1 # A100 80GB
```

### Resource Limits

Check your cluster's limits:
```bash
# Show partition info
sinfo

# Show your resource limits
sacctmgr show assoc user=$USER format=User,Account,Partition,MaxJobs,MaxSubmit

# Show QOS limits
sacctmgr show qos format=Name,MaxWall,MaxSubmit,MaxJobsPU
```

---

## Data Transfer

### Initial upload (from local machine)
```bash
# Sync project (exclude large files)
rsync -avz --progress \
  --exclude='*.sif' \
  --exclude='data/raw/*.zip' \
  --exclude='models/' \
  --exclude='outputs/' \
  akklang/ user@cluster:/home/user/akklang/

# Or use SCP for specific files
scp -r data/raw/*.csv user@cluster:/home/user/akklang/data/raw/
```

### Download results (to local machine)
```bash
# Download predictions
scp user@cluster:/home/user/akklang/submission.csv .

# Download model checkpoint
rsync -avz --progress \
  user@cluster:/home/user/akklang/models/byt5_finetuned/ \
  models/byt5_finetuned/

# Download logs
rsync -avz user@cluster:/home/user/akklang/logs/ logs/
```

---

## Troubleshooting

### Container build fails
```bash
# If you don't have fakeroot:
singularity build --remote akklang.sif singularity.def

# Or use Sylabs cloud build:
# 1. Create account at https://cloud.sylabs.io
# 2. Generate token: singularity remote login
# 3. Build remotely: singularity build --remote akklang.sif singularity.def
```

### Job fails immediately
```bash
# Check error log
cat logs/slurm/baseline_<job_id>.err

# Common issues:
# - Container not found: Run ./cluster/build_container.sh first
# - Data not found: Check PROJECT_ROOT environment variable
# - Permission denied: chmod +x cluster/*.sh
```

### Out of memory (OOM)
```bash
# Reduce batch size in configs/training.yaml:
per_device_train_batch_size: 4  # Was 8
gradient_accumulation_steps: 4  # Was 2

# Enable gradient checkpointing:
gradient_checkpointing: true
```

### CUDA out of memory
```bash
# Use smaller model
model:
  name: google/byt5-small  # Was google/byt5-base

# Or use CPU for baseline (slow!)
#SBATCH --gres=gpu:0
#SBATCH --partition=cpu
```

### Job stuck in queue
```bash
# Check queue status
squeue

# Check why pending
squeue -u $USER --start

# Request different resources if queue is full
#SBATCH --gres=gpu:v100:1  # Try V100 instead of A100
```

---

## Testing Container Locally (Optional)

If you have Singularity on your local machine:

```bash
# Build locally
singularity build akklang.sif singularity.def

# Test Python
singularity exec akklang.sif python -c "import torch; print(torch.__version__)"

# Test with GPU (requires NVIDIA driver)
singularity exec --nv akklang.sif python -c "import torch; print(torch.cuda.is_available())"

# Run interactive shell
singularity shell --nv akklang.sif
```

---

## Next Steps After Setup

1. **Phase 0 (Week 1):** Run baseline
   ```bash
   ./cluster/submit_job.sh baseline
   ```

2. **Phase 1 (Weeks 2-3):** Extract publications
   ```bash
   ./cluster/submit_job.sh extract
   ```

3. **Phase 2 (Weeks 3-4):** Build retrieval index (runs automatically in training job)

4. **Phase 3 (Weeks 4-6):** Train model
   ```bash
   ./cluster/submit_job.sh train
   ```

5. **Phase 4 (Weeks 6-8):** Generate submission
   ```bash
   ./cluster/submit_job.sh inference
   ```

See `docs/ROADMAP.md` for detailed timeline and milestones.

---

## Support

- **SLURM docs:** https://slurm.schedmd.com/
- **Singularity docs:** https://sylabs.io/docs/
- **Cluster-specific help:** Contact your HPC support team

**Common commands:**
- `man sbatch` - SLURM batch submission
- `man squeue` - Queue status
- `man scancel` - Cancel jobs
- `singularity --help` - Singularity help
