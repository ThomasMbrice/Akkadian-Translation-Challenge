# GPU Cluster Setup for Akkadian NMT

This directory contains scripts for running the Akkadian NMT pipeline on a remote GPU cluster using Singularity containers and SLURM.

## Quick Start

```bash
# 1. Upload to cluster
rsync -avz --exclude='*.sif' akklang/ user@cluster:/home/user/akklang/

# 2. Build container
ssh user@cluster
cd ~/akklang
./cluster/build_container.sh

# 3. Test
./cluster/submit_job.sh test

# 4. Run baseline
./cluster/submit_job.sh baseline
```

## Jobs

### Submit
```bash
./cluster/submit_job.sh test       # Test container (15 min)
./cluster/submit_job.sh baseline   # Zero-shot baseline (2 hours)
./cluster/submit_job.sh extract    # Extract data (24 hours, CPU)
./cluster/submit_job.sh train      # Train model (48 hours)
./cluster/submit_job.sh inference  # Generate submission (1 hour)
```

### Monitor
```bash
squeue -u $USER                    # Check status
tail -f logs/slurm/*.out           # View output
scancel <job_id>                   # Cancel job
```

## Resources

| Job | Time | Partition | GPU | CPUs | RAM |
|-----|------|-----------|-----|------|-----|
| test | 15m | a100 | 1 | 2 | 16GB |
| baseline | 2h | a100 | 1 | 4 | 32GB |
| extract | 24h | long-40core-shared | - | 16 | 128GB |
| train | 48h | a100-long | 1 | 8 | 64GB |
| inference | 1h | a100 | 1 | 4 | 32GB |

## Partitions

Scripts use: `a100`, `a100-long`, `long-40core-shared`

Check your cluster:
```bash
sinfo
```

Edit `#SBATCH --partition=` in `.slurm` files if different.
