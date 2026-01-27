# GPU Cluster Quick Start Guide

This guide covers running the Akkadian NMT pipeline on a remote GPU cluster using Singularity and SLURM.

## Prerequisites

- Access to a SLURM-managed GPU cluster
- Singularity installed on the cluster (most HPC centers have this)
- SSH access to the cluster
- Kaggle API credentials (for data download and submission)

---

## Initial Setup (One-Time)

### 1. Upload Project to Cluster

From your local machine:

```bash
# Sync entire project
rsync -avz --progress \
  --exclude='*.sif' \
  --exclude='data/raw/*.zip' \
  --exclude='models/' \
  akklang/ user@cluster.edu:/home/user/akklang/

# Replace 'user@cluster.edu' with your cluster address
```

### 2. Build Singularity Container

SSH to the cluster and build the container:

```bash
ssh user@cluster.edu
cd ~/akklang

# Build container (~15 minutes)
./cluster/build_container.sh

# Verify build
singularity exec --nv akklang.sif python -c \
  "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**Expected output:**
```
CUDA: True
```

### 3. Verify Data Files

Ensure competition data is present:

```bash
ls -lh data/raw/deep-past-initiative-machine-translation/
```

Should show:
- `train.csv`
- `test.csv`
- `publications.csv`
- `published_texts.csv`
- `OA_Lexicon_eBL.csv`
- `eBL_Dictionary.csv`
- And other files...

If missing, download from Kaggle:
```bash
# Configure Kaggle API
mkdir -p ~/.kaggle
vi ~/.kaggle/kaggle.json  # Paste your API token

# Download competition data
kaggle competitions download -c deep-past-initiative-machine-translation
unzip deep-past-initiative-machine-translation.zip -d data/raw/deep-past-initiative-machine-translation/
```

---

## Running the Pipeline

### Phase 0: Baseline (Week 1)

**Goal:** Establish zero-shot ByT5 baseline
**Duration:** ~2 hours
**Resources:** 1 GPU, 32GB RAM

```bash
./cluster/submit_job.sh baseline

# Monitor progress
tail -f logs/slurm/baseline_*.out
```

**Outputs:**
- `outputs/baseline_results.json` - Metrics
- `predictions.csv` - Test predictions

**Expected scores:** BLEU 5-10, chrF++ 20-30

---

### Phase 1: Extract Publications (Weeks 2-3)

**Goal:** Extract 20-50k parallel pairs from 900 publications
**Duration:** ~24 hours (can be longer)
**Resources:** 16 CPUs, 128GB RAM (no GPU needed)

```bash
./cluster/submit_job.sh extract

# Monitor progress
tail -f logs/slurm/extract_*.out
```

**Outputs:**
- `data/processed/combined_corpus.csv` - Final training corpus

**Critical:** This step multiplies training data by 2-3x. Essential for good model performance.

---

### Phase 3: Training (Weeks 4-6)

**Goal:** Fine-tune ByT5 with RAG
**Duration:** ~48 hours
**Resources:** 1 A100 GPU (or V100), 64GB RAM

```bash
./cluster/submit_job.sh train

# Monitor progress
tail -f logs/slurm/train_*.out

# Check training curves (if using TensorBoard)
tensorboard --logdir logs/training/
```

**Outputs:**
- `models/byt5_finetuned/` - Model checkpoint

**Target:** Validation BLEU > 25

---

### Phase 4: Inference (Weeks 6-8)

**Goal:** Generate competition submission
**Duration:** ~1 hour
**Resources:** 1 GPU, 32GB RAM

```bash
./cluster/submit_job.sh inference

# Monitor progress
tail -f logs/slurm/inference_*.out
```

**Outputs:**
- `submission.csv` - Ready for Kaggle upload

**Submit to competition:**
```bash
kaggle competitions submit \
  -c deep-past-initiative-machine-translation \
  -f submission.csv \
  -m "ByT5 + RAG (BLEU XX.X)"
```

---

## Job Monitoring Commands

```bash
# List your jobs
squeue -u $USER

# Show job details
scontrol show job <job_id>

# Cancel a job
scancel <job_id>

# View job output in real-time
tail -f logs/slurm/<job_name>_<job_id>.out

# Check GPU usage (when on compute node)
nvidia-smi

# View job history
sacct -u $USER --format=JobID,JobName,State,Elapsed,ReqMem,MaxRSS
```

---

## Downloading Results

From your local machine:

```bash
# Download submission file
scp user@cluster.edu:/home/user/akklang/submission.csv .

# Download model checkpoint (large!)
rsync -avz --progress \
  user@cluster.edu:/home/user/akklang/models/byt5_finetuned/ \
  models/byt5_finetuned/

# Download all logs
rsync -avz user@cluster.edu:/home/user/akklang/logs/ logs/
```

---

## Troubleshooting

### Job immediately fails

**Check error log:**
```bash
cat logs/slurm/<job_name>_<job_id>.err
```

**Common issues:**
- **Container not found:** Build it with `./cluster/build_container.sh`
- **Data missing:** Verify `data/raw/` has competition files
- **Permission error:** Run `chmod +x cluster/*.sh`

### Out of Memory (OOM)

**Edit `configs/training.yaml`:**
```yaml
training:
  per_device_train_batch_size: 4  # Reduce from 8
  gradient_accumulation_steps: 4  # Increase from 2

hardware:
  gradient_checkpointing: true  # Enable memory optimization
```

### CUDA Out of Memory

**Use smaller model in `configs/training.yaml`:**
```yaml
model:
  name: google/byt5-small  # Instead of byt5-base
```

### Job stuck in queue

**Check queue status:**
```bash
squeue
squeue -u $USER --start  # See estimated start time
```

**If queue is full, try:**
- Different partition: Edit `#SBATCH --partition=` in `.slurm` file
- Different GPU type: Change `#SBATCH --gres=gpu:v100:1`
- Shorter time limit: Reduce `#SBATCH --time=`

---

## Customizing for Your Cluster

### Update Partition Names

Edit `.slurm` files if your cluster uses different partition names:

```bash
# Example for cluster with 'gpuq' and 'cpuq' partitions
sed -i 's/partition=gpu/partition=gpuq/g' cluster/*.slurm
sed -i 's/partition=cpu/partition=cpuq/g' cluster/*.slurm
```

### Update GPU Requests

```bash
# Request specific GPU type
#SBATCH --gres=gpu:v100:1    # V100 (16GB VRAM)
#SBATCH --gres=gpu:a100:1    # A100 (40GB VRAM, preferred)
#SBATCH --gres=gpu:a100_80:1 # A100 80GB
```

### Check Available Resources

```bash
# Show cluster info
sinfo

# Show your account limits
sacctmgr show assoc user=$USER format=User,Account,Partition,MaxJobs

# Show available GPUs
sinfo -o "%20N %10c %10m %25f %10G"
```

---

## Resource Requirements Summary

| Phase | Job | Duration | GPU | CPUs | RAM | Partition |
|-------|-----|----------|-----|------|-----|-----------|
| 0 | Baseline | 2h | 1x V100 | 4 | 32GB | `gpu` |
| 1 | Extract | 24h | None | 16 | 128GB | `cpu` |
| 3 | Training | 48h | 1x A100 | 8 | 64GB | `gpu` |
| 4 | Inference | 1h | 1x V100 | 4 | 32GB | `gpu` |

---

## Pipeline Timeline

```
Week 1:  Phase 0 - Baseline (~2h job)
Week 2:  Phase 1 - Extract publications (~24h job)
Week 3:  Phase 1 - Quality analysis + fixes
Week 4:  Phase 3 - Start training (~48h job)
Week 5:  Phase 3 - Hyperparameter tuning
Week 6:  Phase 3 - Final training run
Week 7:  Phase 4 - Error analysis + fixes
Week 8:  Phase 4 - Final submission (~1h job)
```

**Total GPU time:** ~100 hours (baseline + training iterations + inference)

---

## Best Practices

1. **Test locally first** - Validate scripts on small data before submitting long jobs
2. **Use checkpoints** - Training resumes from checkpoints if interrupted
3. **Monitor early** - Check first 10 minutes of output for errors
4. **Save logs** - Keep all `logs/slurm/` files for debugging
5. **Version models** - Tag checkpoints with date/hyperparameters
6. **Backup results** - Download important outputs regularly

---

## Next Steps

After setup is complete:

1. ✓ Container built
2. ✓ Data uploaded
3. → **Run baseline:** `./cluster/submit_job.sh baseline`
4. → **Review results:** Check `outputs/baseline_results.json`
5. → **Continue to Phase 1:** See `docs/ROADMAP.md`

---

## Additional Resources

- **Full cluster docs:** `cluster/README.md`
- **Project roadmap:** `docs/ROADMAP.md`
- **Architecture:** `docs/ARCHITECTURE.md`
- **SLURM documentation:** https://slurm.schedmd.com/
- **Singularity documentation:** https://sylabs.io/docs/
