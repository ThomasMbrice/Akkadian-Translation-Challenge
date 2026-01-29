"""
File I/O utilities for the Akkadian NMT project.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_csv(
    file_path: str | Path,
    required_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load CSV file with validation.

    Args:
        file_path: Path to CSV file
        required_columns: List of required column names

    Returns:
        DataFrame with loaded data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Loading CSV: {file_path}")
    df = pd.read_csv(file_path)

    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def save_csv(
    df: pd.DataFrame,
    file_path: str | Path,
    create_dirs: bool = True,
) -> None:
    """
    Save DataFrame to CSV.

    Args:
        df: DataFrame to save
        file_path: Output path
        create_dirs: Create parent directories if they don't exist
    """
    file_path = Path(file_path)

    if create_dirs:
        file_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving CSV: {file_path} ({len(df)} rows)")
    df.to_csv(file_path, index=False)


def load_yaml(file_path: str | Path) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        file_path: Path to YAML file

    Returns:
        Dictionary with configuration

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Loading YAML: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def save_json(
    data: Dict | List,
    file_path: str | Path,
    create_dirs: bool = True,
    indent: int = 2,
) -> None:
    """
    Save data to JSON file.

    Args:
        data: Data to save (dict or list)
        file_path: Output path
        create_dirs: Create parent directories if they don't exist
        indent: JSON indentation
    """
    file_path = Path(file_path)

    if create_dirs:
        file_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving JSON: {file_path}")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(file_path: str | Path) -> Dict | List:
    """
    Load JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Loaded data (dict or list)

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Loading JSON: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def ensure_dir(dir_path: str | Path) -> Path:
    """
    Ensure directory exists, create if necessary.

    Args:
        dir_path: Directory path

    Returns:
        Path object for the directory
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def setup_logging(
    log_file: Optional[str | Path] = None,
    level: str = "INFO",
    format_str: Optional[str] = None,
) -> None:
    """
    Setup logging configuration.

    Args:
        log_file: Optional log file path
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        format_str: Custom format string
    """
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers = [logging.StreamHandler()]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str,
        handlers=handlers,
    )
