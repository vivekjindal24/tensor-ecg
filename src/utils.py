"""
Utility functions for ECG data I/O and common operations.
"""

import os
import json
import logging
import random
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


def safe_makedirs(path: Union[str, Path]) -> None:
    """
    Create directory safely if it does not exist.

    Args:
        path: Directory path to create
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def read_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read JSON file safely.

    Args:
        path: Path to JSON file

    Returns:
        Dictionary containing JSON data
    """
    with open(path, 'r') as f:
        return json.load(f)


def write_json(path: Union[str, Path], data: Dict[str, Any], indent: int = 2) -> None:
    """
    Write dictionary to JSON file.

    Args:
        path: Output path
        data: Dictionary to serialize
        indent: JSON indentation level
    """
    safe_makedirs(Path(path).parent)
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across numpy and torch.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def print_device_info() -> str:
    """
    Print and return device information (CUDA GPU or CPU).

    Returns:
        Device string ('cuda' or 'cpu')
    """
    try:
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"Using CUDA device: {gpu_name}")
            logger.info(f"GPU memory: {gpu_memory:.2f} GB")
            return device
        else:
            logger.info("CUDA not available, using CPU")
            return 'cpu'
    except ImportError:
        logger.warning("PyTorch not installed, defaulting to CPU")
        return 'cpu'


def safe_save_npz(path: Union[str, Path], signal: np.ndarray, label: int) -> None:
    """
    Save signal and label to compressed NPZ file safely.

    Args:
        path: Output file path
        signal: Signal array
        label: Integer label
    """
    safe_makedirs(Path(path).parent)
    np.savez_compressed(path, signal=signal, label=label)


def robust_path_variants(path: Union[str, Path]) -> list:
    """
    Generate multiple path variants for robust matching.
    Handles different path separators and relative path formats.

    Args:
        path: Input path

    Returns:
        List of path string variants
    """
    p = Path(path)
    variants = []

    # Normalize separators to forward slash
    normalized = str(p).replace('\\', '/')
    variants.append(normalized)

    # Without dataset prefix (assumes dataset is first component)
    parts = Path(normalized).parts
    if len(parts) > 1:
        variants.append('/'.join(parts[1:]))

    # Last two components
    if len(parts) >= 2:
        variants.append('/'.join(parts[-2:]))

    # Basename without extension
    stem = p.stem
    variants.append(stem)

    # Full relative path without extension
    variants.append(str(p.with_suffix('')).replace('\\', '/'))

    return list(set(variants))


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent


