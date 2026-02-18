#!/usr/bin/env python3
"""
Quick pre-flight check — run on the login node before sbatch.

Checks:
  1. Python version (must be 3.9+)
  2. All required pip packages import correctly
  3. All required data files exist
  4. Config loads and has required keys

Usage:
    python scripts/check_deps.py
    python scripts/check_deps.py --config configs/training.yaml
"""
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PASS = "[OK]"
FAIL = "[FAIL]"


def check(label, fn, warn_only=False):
    try:
        fn()
        print(f"  {PASS}  {label}")
        return True
    except Exception as e:
        tag = "[WARN]" if warn_only else FAIL
        print(f"  {tag}  {label}: {e}")
        return warn_only  # warn_only=True means don't count as failure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training.yaml")
    args = parser.parse_args()

    failures = 0

    # ------------------------------------------------------------------
    # 1. Python version
    # ------------------------------------------------------------------
    print("\n--- Python ---")
    v = sys.version_info
    if v >= (3, 9):
        print(f"  {PASS}  Python {v.major}.{v.minor}.{v.micro}")
    else:
        print(f"  {FAIL}  Python {v.major}.{v.minor}.{v.micro} — need 3.9+")
        failures += 1

    # ------------------------------------------------------------------
    # 2. Package imports
    # ------------------------------------------------------------------
    print("\n--- Packages ---")
    packages = [
        ("torch", "import torch"),
        ("torch", "import torch"),
        ("transformers", "import transformers"),
        ("datasets", "import datasets"),
        ("pandas", "import pandas"),
        ("sklearn", "import sklearn"),
        ("faiss", "import faiss"),
        ("sentence_transformers", "import sentence_transformers"),
        ("sacrebleu", "import sacrebleu"),
        ("yaml", "import yaml"),
    ]
    for label, stmt in packages:
        if not check(label, lambda s=stmt: exec(s)):
            failures += 1

    check(
        "torch CUDA (login nodes have no GPU — warning only)",
        lambda: (_ for _ in ()).throw(AssertionError("no CUDA")) if not __import__("torch").cuda.is_available() else None,
        warn_only=True,
    )

    # ------------------------------------------------------------------
    # 3. Project module imports
    # ------------------------------------------------------------------
    print("\n--- Project modules ---")
    modules = [
        ("src.utils.io", "from src.utils.io import setup_logging, load_yaml, save_json"),
        ("src.modeling", "from src.modeling import ByT5Trainer, ContextAssembler, Augmenter"),
        ("src.retrieval", "from src.retrieval import Retriever"),
        ("src.evaluation.metrics", "from src.evaluation.metrics import MetricsCalculator"),
    ]
    for label, stmt in modules:
        if not check(label, lambda s=stmt: exec(s)):
            failures += 1

    # ------------------------------------------------------------------
    # 4. Data files
    # ------------------------------------------------------------------
    print("\n--- Data files ---")
    config_path = PROJECT_ROOT / args.config
    data_files = [
        config_path,
        PROJECT_ROOT / "data/processed/combined_corpus.csv",
        PROJECT_ROOT / "data/raw/deep-past-initiative-machine-translation/OA_Lexicon_eBL.csv",
        PROJECT_ROOT / "data/raw/deep-past-initiative-machine-translation/eBL_Dictionary.csv",
        PROJECT_ROOT / "data/indices/faiss_index.bin",
    ]
    for f in data_files:
        rel = f.relative_to(PROJECT_ROOT)
        if not check(str(rel), lambda p=f: (_ for _ in ()).throw(FileNotFoundError(p)) if not p.exists() else None):
            failures += 1

    # ------------------------------------------------------------------
    # 5. Config keys
    # ------------------------------------------------------------------
    print("\n--- Config ---")
    try:
        import yaml
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh)
        required_keys = ["model", "data", "retrieval", "training", "output"]
        for k in required_keys:
            if not check(f"config['{k}']", lambda key=k: (_ for _ in ()).throw(KeyError(key)) if key not in cfg else None):
                failures += 1
    except Exception as e:
        print(f"  {FAIL}  Could not load config: {e}")
        failures += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    if failures == 0:
        print("All checks passed — safe to sbatch.")
    else:
        print(f"{failures} check(s) FAILED — fix before submitting.")
        sys.exit(1)


if __name__ == "__main__":
    main()
