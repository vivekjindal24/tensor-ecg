"""
ECG Preprocessing Module
Handles loading, filtering, resampling, and normalization of ECG signals.
Implements streaming preprocessing with unified label mapping.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import wfdb
from scipy import signal as scipy_signal
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .utils import read_json, robust_path_variants, safe_makedirs, safe_save_npz, write_json

logger = logging.getLogger(__name__)


def load_wfdb_record(record_path: Union[str, Path]) -> Tuple[np.ndarray, float]:
    """
    Load WFDB format ECG record (.hea/.dat files).

    Args:
        record_path: Path to record (without extension)

    Returns:
        Tuple of (signal array, sampling frequency)

    Raises:
        ValueError: If record cannot be read
    """
    try:
        record = wfdb.rdrecord(str(Path(record_path).with_suffix('')))
        if record.p_signal is None:
            raise ValueError(f"No signal data in record: {record_path}")

        # Flatten multi-lead to single channel (mean across leads)
        signal = record.p_signal.mean(axis=1) if record.p_signal.ndim > 1 else record.p_signal
        return signal.astype(np.float32), float(record.fs)
    except Exception as e:
        raise ValueError(f"Failed to read WFDB record {record_path}: {e}")


def load_mat_record(mat_path: Union[str, Path]) -> Tuple[np.ndarray, float]:
    """
    Load MAT format ECG record.

    Args:
        mat_path: Path to .mat file

    Returns:
        Tuple of (signal array, sampling frequency)

    Raises:
        ValueError: If MAT file cannot be read
    """
    try:
        mat_data = loadmat(str(mat_path))

        # Common MAT file structures
        signal = None
        fs = 500  # Default fallback

        # Try common field names
        for key in ['val', 'data', 'ecg', 'signal']:
            if key in mat_data:
                signal = mat_data[key]
                break

        if signal is None:
            # Try first non-metadata field
            for key, value in mat_data.items():
                if not key.startswith('__') and isinstance(value, np.ndarray):
                    signal = value
                    break

        if signal is None:
            raise ValueError(f"No signal data found in MAT file: {mat_path}")

        # Flatten and ensure 1D
        signal = signal.flatten().astype(np.float32)

        # Try to extract sampling frequency
        for key in ['fs', 'Fs', 'freq', 'frequency']:
            if key in mat_data:
                fs = float(mat_data[key].item())
                break

        return signal, fs
    except Exception as e:
        raise ValueError(f"Failed to read MAT file {mat_path}: {e}")


def resample_signal(signal: np.ndarray, original_fs: float, target_fs: float = 500) -> np.ndarray:
    """
    Resample signal to target frequency using scipy.signal.resample.

    Args:
        signal: Input signal array
        original_fs: Original sampling frequency
        target_fs: Target sampling frequency

    Returns:
        Resampled signal
    """
    if abs(original_fs - target_fs) < 1e-3:
        return signal

    num_samples = int(len(signal) * target_fs / original_fs)
    return scipy_signal.resample(signal, num_samples).astype(np.float32)


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Z-score normalize signal.

    Args:
        signal: Input signal array

    Returns:
        Normalized signal
    """
    mean = np.mean(signal)
    std = np.std(signal)
    if std < 1e-8:
        logger.warning("Signal has near-zero std, skipping normalization")
        return signal
    return ((signal - mean) / std).astype(np.float32)


def pad_or_truncate(signal: np.ndarray, target_samples: int) -> np.ndarray:
    """
    Pad with zeros or truncate signal to target length.

    Args:
        signal: Input signal array
        target_samples: Desired length

    Returns:
        Signal of length target_samples
    """
    current_length = len(signal)
    if current_length < target_samples:
        pad_width = target_samples - current_length
        return np.pad(signal, (0, pad_width), mode='constant').astype(np.float32)
    else:
        return signal[:target_samples].astype(np.float32)


def build_mapping_index(unified_csv: Union[str, Path]) -> Dict[str, str]:
    """
    Build robust mapping index from unified label CSV.

    Args:
        unified_csv: Path to unified_label_mapping.csv

    Returns:
        Dictionary mapping path variants to mapped_label
    """
    try:
        df = pd.read_csv(unified_csv)
        required_cols = ['dataset', 'record_id', 'mapped_label']
        missing = set(required_cols) - set(df.columns)
        if missing:
            logger.warning(f"Missing columns in unified CSV: {missing}, using available columns")

        mapping = {}
        for _, row in df.iterrows():
            dataset = row.get('dataset', '')
            record_id = str(row.get('record_id', ''))
            mapped_label = row.get('mapped_label', 'OTHER')

            # Build full relative path
            full_path = f"{dataset}/{record_id}" if dataset else record_id

            # Generate variants
            variants = robust_path_variants(full_path)
            for variant in variants:
                mapping[variant] = mapped_label

        logger.info(f"Built mapping index with {len(mapping)} path variants from {len(df)} records")
        return mapping
    except FileNotFoundError:
        logger.warning(f"Unified label mapping not found: {unified_csv}, all records will be marked OTHER")
        return {}
    except Exception as e:
        logger.error(f"Error reading unified CSV: {e}")
        return {}


def lookup_label(file_path: Path, dataset_dir: Path, mapping_index: Dict[str, str]) -> str:
    """
    Lookup mapped label for a file using robust path matching.

    Args:
        file_path: Full path to file
        dataset_dir: Root dataset directory
        mapping_index: Mapping dictionary from build_mapping_index

    Returns:
        Mapped label string (default: 'OTHER')
    """
    try:
        rel_path = file_path.relative_to(dataset_dir)
    except ValueError:
        rel_path = file_path

    # Generate variants
    variants = robust_path_variants(rel_path)

    for variant in variants:
        if variant in mapping_index:
            return mapping_index[variant]

    return 'OTHER'


def sanitize_filename(path: str) -> str:
    """
    Sanitize path for use as filename.

    Args:
        path: Input path string

    Returns:
        Sanitized filename safe for filesystem
    """
    # Replace separators and special chars with underscore
    sanitized = re.sub(r'[\\/:*?"<>|]', '_', path)
    # Remove multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    return sanitized.strip('_')


def run_streaming_preprocess(
    dataset_dir: Union[str, Path],
    unified_csv: Union[str, Path],
    out_dir: Union[str, Path],
    target_fs: int = 500,
    target_samples: int = 5000,
    label_order: Optional[List[str]] = None,
    limit: Optional[int] = None
) -> Dict[str, any]:
    """
    Run streaming preprocessing on dataset with unified label mapping.
    Processes files one at a time, saves individually, and builds manifest.

    Args:
        dataset_dir: Root directory containing datasets
        unified_csv: Path to unified_label_mapping.csv
        out_dir: Output directory for processed records
        target_fs: Target sampling frequency (Hz)
        target_samples: Target number of samples
        label_order: List of label names in order (for encoding)
        limit: Optional limit on number of files to process (for testing)

    Returns:
        Dictionary with processing statistics and paths
    """
    if label_order is None:
        label_order = ['MI', 'AF', 'BBB', 'NORM', 'OTHER']

    dataset_dir = Path(dataset_dir)
    out_dir = Path(out_dir)
    records_dir = out_dir / 'records'
    safe_makedirs(records_dir)

    # Build mapping index
    mapping_index = build_mapping_index(unified_csv)

    # Label encoding
    label_to_int = {label: idx for idx, label in enumerate(label_order)}

    # Find all ECG files
    hea_files = list(dataset_dir.rglob('*.hea'))
    mat_files = list(dataset_dir.rglob('*.mat'))

    all_files = hea_files + mat_files
    if limit:
        all_files = all_files[:limit]

    logger.info(f"Found {len(hea_files)} .hea files and {len(mat_files)} .mat files")
    logger.info(f"Processing {len(all_files)} files total")

    manifest = []
    label_counts = {label: 0 for label in label_order}
    failed_count = 0

    for file_path in tqdm(all_files, desc="Processing records"):
        try:
            # Read signal
            if file_path.suffix == '.hea':
                signal, fs = load_wfdb_record(file_path.with_suffix(''))
            elif file_path.suffix == '.mat':
                signal, fs = load_mat_record(file_path)
            else:
                continue

            # Resample if needed
            if abs(fs - target_fs) > 1e-3:
                signal = resample_signal(signal, fs, target_fs)

            # Normalize
            signal = normalize_signal(signal)

            # Pad or truncate
            signal = pad_or_truncate(signal, target_samples)

            # Lookup label
            mapped_label = lookup_label(file_path, dataset_dir, mapping_index)
            label_int = label_to_int.get(mapped_label, label_to_int['OTHER'])

            # Generate output filename
            try:
                rel_path = file_path.relative_to(dataset_dir)
            except ValueError:
                rel_path = file_path

            dataset_name = rel_path.parts[0] if len(rel_path.parts) > 1 else 'unknown'
            sanitized = sanitize_filename(str(rel_path.with_suffix('')))
            out_filename = f"{dataset_name}__{sanitized}.npz"
            out_path = records_dir / out_filename

            # Save
            safe_save_npz(out_path, signal, label_int)

            # Add to manifest
            manifest.append({
                'path': f"records/{out_filename}",
                'label': int(label_int),
                'mapped_label': mapped_label,
                'dataset': dataset_name
            })

            label_counts[mapped_label] = label_counts.get(mapped_label, 0) + 1

        except Exception as e:
            logger.warning(f"Failed to process {file_path}: {e}")
            failed_count += 1
            continue

    logger.info(f"Successfully processed {len(manifest)} records, {failed_count} failures")
    logger.info(f"Label distribution: {label_counts}")

    # Create stratified splits
    labels_list = [item['label'] for item in manifest]
    indices = np.arange(len(manifest))

    if len(manifest) > 10:
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.2, stratify=labels_list, random_state=42
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, stratify=[labels_list[i] for i in temp_idx], random_state=42
        )
    else:
        # Too few samples for stratification
        train_idx = indices
        val_idx = np.array([])
        test_idx = np.array([])

    splits = {
        'train': [manifest[i] for i in train_idx],
        'val': [manifest[i] for i in val_idx] if len(val_idx) > 0 else [],
        'test': [manifest[i] for i in test_idx] if len(test_idx) > 0 else [],
        'label_order': label_order,
        'label_to_int': label_to_int,
        'counts': label_counts
    }

    # Save outputs
    write_json(out_dir / 'splits.json', splits)
    write_json(out_dir / 'label_map.json', {'label_order': label_order, 'label_to_int': label_to_int})
    np.save(out_dir / 'labels.npy', np.array(labels_list))

    logger.info(f"Saved splits.json, label_map.json, and labels.npy to {out_dir}")

    return {
        'total_processed': len(manifest),
        'total_failed': failed_count,
        'splits': splits,
        'label_counts': label_counts,
        'output_dir': str(out_dir)
    }

