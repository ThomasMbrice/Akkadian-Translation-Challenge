# GPU Cluster Quick Start Guide

This guide covers running the Akkadian NMT pipeline on a remote GPU cluster using Singularity and SLURM.

## Prerequisites

- Access to a SLURM-managed GPU cluster
- Singularity installed on the cluster (most HPC centers have this)
- SSH access to the cluster
- Kaggle API credentials (for data download and submission)

---

## Initial Setup

### 1. Upload Project

```bash
rsync -avz --exclude='*.sif' --exclude='models/' \
  akklang/ user@cluster:/home/user/akklang/
```

### 2. Build Container

```bash
ssh user@cluster
cd ~/akklang
./cluster/build_container.sh
```

### 3. Test Container

```bash
./cluster/submit_job.sh test
```

---

## Running Jobs

```bash
# Test container (15 min)
./cluster/submit_job.sh test

# Zero-shot baseline (2 hours)
./cluster/submit_job.sh baseline

# Extract data (24 hours, CPU-only)
./cluster/submit_job.sh extract

# Train model (48 hours)
./cluster/submit_job.sh train

# Generate submission (1 hour)
./cluster/submit_job.sh inference
```

### Monitor Jobs

```bash
# Check job status
squeue -u $USER

# View output
tail -f logs/slurm/*.out
```

---

## Partition Configuration

Scripts are configured for:
- GPU jobs: `a100`, `a100-long` partitions
- CPU jobs: `long-40core-shared` partition

To check your cluster's partitions:
```bash
sinfo
```

To change partitions, edit the `#SBATCH --partition=` line in each `.slurm` file.

## Resource Requirements

| Job | Time | Partition | GPU | CPUs | RAM |
|-----|------|-----------|-----|------|-----|
| test | 15m | a100 | 1 | 2 | 16GB |
| baseline | 2h | a100 | 1 | 4 | 32GB |
| extract | 24h | long-40core-shared | - | 16 | 128GB |
| train | 48h | a100-long | 1 | 8 | 64GB |
| inference | 1h | a100 | 1 | 4 | 32GB |
