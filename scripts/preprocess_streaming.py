"""
Memory-safe, idempotent streaming preprocessing for ECG datasets.
Saves individual .npy files with metadata and creates manifest.jsonl for lazy loading.
"""
import os
import sys
import json
import time
import logging
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.signal import resample
from scipy.io import loadmat
from tqdm import tqdm

# Windows asyncio fix
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Configuration
ROOT = Path(__file__).parent.parent
DATASET_DIR = ROOT / "Dataset"
ARTIFACTS_DIR = ROOT / "artifacts"
PROCESSED_DIR = ARTIFACTS_DIR / "processed"
RECORDS_DIR = PROCESSED_DIR / "records"
LOGS_DIR = ROOT / "logs"

# Create directories
for d in [ARTIFACTS_DIR, PROCESSED_DIR, RECORDS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Setup logging
log_file = LOGS_DIR / "preprocess_automation.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
TARGET_FS = 500
TARGET_SAMPLES = 5000
LABEL_ORDER = ['MI', 'AF', 'BBB', 'NORM', 'OTHER']
LABEL_TO_INT = {name: i for i, name in enumerate(LABEL_ORDER)}
INT_TO_LABEL = {i: name for name, i in LABEL_TO_INT.items()}

# Environment variable for smoke testing
PREPROCESS_LIMIT = int(os.getenv('ECG_PREPROCESS_LIMIT', '0'))

logger.info("="*80)
logger.info("ECG STREAMING PREPROCESSING")
logger.info("="*80)
logger.info(f"ROOT: {ROOT}")
logger.info(f"DATASET_DIR: {DATASET_DIR}")
logger.info(f"RECORDS_DIR: {RECORDS_DIR}")
logger.info(f"PREPROCESS_LIMIT: {PREPROCESS_LIMIT if PREPROCESS_LIMIT > 0 else 'UNLIMITED'}")


def zscore_normalize(arr: np.ndarray) -> np.ndarray:
    """Z-score normalization"""
    arr = arr.astype(np.float32)
    mean = arr.mean()
    std = arr.std()
    if std < 1e-8:
        std = 1.0
    return ((arr - mean) / std).astype(np.float32)


def pad_or_truncate(x: np.ndarray, target_length: int) -> np.ndarray:
    """Pad with zeros or truncate"""
    if x.size >= target_length:
        return x[:target_length]
    pad_width = target_length - x.size
    return np.pad(x, (0, pad_width), mode='constant', constant_values=0).astype(np.float32)


def resample_signal(x: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
    """Resample signal"""
    if original_fs is None or np.isclose(original_fs, target_fs):
        return x
    new_length = int(round(x.size * target_fs / original_fs))
    if new_length <= 0:
        return x
    return resample(x, new_length).astype(np.float32)


def read_wfdb(hea_path: Path):
    """Read WFDB format (.hea/.dat)"""
    try:
        import wfdb
        record_path = str(hea_path.with_suffix(''))
        record = wfdb.rdsamp(record_path)
        signal = np.asarray(record[0], dtype=np.float32)
        fs = float(record[1].get('fs', TARGET_FS))

        # Convert to 1D
        if signal.ndim == 2:
            signal_1d = signal.mean(axis=1) if signal.shape[1] > 1 else signal[:, 0]
        else:
            signal_1d = signal.reshape(-1)

        return signal_1d.astype(np.float32), fs
    except Exception as e:
        raise RuntimeError(f"Failed to read WFDB {hea_path.name}: {e}")


def read_mat(mat_path: Path):
    """Read MATLAB .mat file"""
    try:
        mat_data = loadmat(str(mat_path))

        signal = None
        for key in ['val', 'data', 'signal', 'ecg']:
            if key in mat_data:
                signal = np.asarray(mat_data[key], dtype=np.float32)
                break

        if signal is None:
            for value in mat_data.values():
                if isinstance(value, np.ndarray) and value.size > 100:
                    signal = value.astype(np.float32)
                    break

        if signal is None:
            raise RuntimeError("No signal array found")

        # Convert to 1D
        if signal.ndim == 2:
            signal_1d = signal.mean(axis=0) if signal.shape[0] > 1 else signal.reshape(-1)
        else:
            signal_1d = signal.reshape(-1)

        return signal_1d.astype(np.float32), None
    except Exception as e:
        raise RuntimeError(f"Failed to read MAT {mat_path.name}: {e}")


def load_mapping_index():
    """Load unified label mapping"""
    mapping_file = LOGS_DIR / "unified_label_mapping.csv"
    mapping_index = {}

    if not mapping_file.exists():
        logger.warning(f"Mapping file not found: {mapping_file}")
        return mapping_index

    df = pd.read_csv(mapping_file, dtype=str).fillna("")

    for _, row in df.iterrows():
        dataset = str(row.get("dataset", "")).strip()
        record_id = str(row.get("record_id", "")).strip().replace("\\", "/").strip("/")
        mapped_label = str(row.get("mapped_label", "")).strip().upper()

        if not dataset or not record_id:
            continue

        if dataset not in mapping_index:
            mapping_index[dataset] = {}

        # Add full path
        mapping_index[dataset][record_id] = mapped_label

        # Add variants
        parts = record_id.split("/")
        if len(parts) >= 1:
            mapping_index[dataset][parts[-1]] = mapped_label
        if len(parts) >= 2:
            mapping_index[dataset]["/".join(parts[-2:])] = mapped_label

    logger.info(f"Loaded mapping for {len(mapping_index)} datasets")
    return mapping_index


def lookup_mapped_label(path: Path, mapping_index: dict) -> str:
    """Look up mapped label"""
    try:
        rel_path = path.relative_to(DATASET_DIR).with_suffix("")
    except Exception:
        rel_path = path.with_suffix("")

    parts = rel_path.as_posix().split("/")
    dataset = parts[0] if parts else ""

    if dataset not in mapping_index:
        return "OTHER"

    index = mapping_index[dataset]

    candidates = [
        rel_path.as_posix(),
        "/".join(parts[1:]) if len(parts) > 1 else "",
        "/".join(parts[-2:]) if len(parts) >= 2 else "",
        rel_path.name
    ]

    for key in candidates:
        if key and key in index:
            label = index[key].upper()
            return label if label in LABEL_TO_INT else "OTHER"

    # CinC special case
    if "CinC" in dataset and len(parts) >= 3:
        alt_key = "/".join(parts[2:])
        if alt_key in index:
            label = index[alt_key].upper()
            return label if label in LABEL_TO_INT else "OTHER"

    return "OTHER"


def load_progress_checkpoint():
    """Load progress checkpoint"""
    checkpoint_file = PROCESSED_DIR / "progress.json"
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return {"processed_files": set(), "last_index": 0, "processed_count": 0}


def save_progress_checkpoint(processed_files: set, last_index: int, processed_count: int):
    """Save progress checkpoint"""
    checkpoint_file = PROCESSED_DIR / "progress.json"
    checkpoint = {
        "processed_files": list(processed_files),
        "last_index": last_index,
        "processed_count": processed_count,
        "timestamp": datetime.utcnow().isoformat()
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def main():
    """Main preprocessing pipeline"""
    logger.info("Starting preprocessing...")

    # Load mapping
    mapping_index = load_mapping_index()

    # Load progress checkpoint
    checkpoint = load_progress_checkpoint()
    processed_files = set(checkpoint.get("processed_files", []))
    logger.info(f"Loaded checkpoint: {len(processed_files)} files already processed")

    # Find all files
    logger.info(f"Scanning {DATASET_DIR}...")
    hea_files = sorted(DATASET_DIR.rglob("*.hea"))
    mat_files = sorted(DATASET_DIR.rglob("*.mat"))
    all_files = hea_files + mat_files

    logger.info(f"Found {len(hea_files)} .hea and {len(mat_files)} .mat files")

    # Apply limit for smoke testing
    if PREPROCESS_LIMIT > 0:
        all_files = all_files[:PREPROCESS_LIMIT]
        logger.info(f"LIMITED TO: {len(all_files)} files (ECG_PREPROCESS_LIMIT={PREPROCESS_LIMIT})")

    # Filter out already processed
    all_files = [f for f in all_files if str(f) not in processed_files]
    logger.info(f"Files to process (after skipping existing): {len(all_files)}")

    if not all_files:
        logger.info("No new files to process!")
        return

    # Open manifest file in append mode
    manifest_file = PROCESSED_DIR / "manifest.jsonl"
    manifest_fp = open(manifest_file, 'a', encoding='utf-8')

    # Processing loop
    label_counts = Counter()
    skipped = 0
    start_time = time.time()
    save_times = []

    progress_bar = tqdm(all_files, desc="Processing", unit="file")

    for idx, file_path in enumerate(progress_bar):
        try:
            # Generate record ID
            try:
                rel_path = file_path.relative_to(DATASET_DIR).with_suffix("")
                record_id = rel_path.as_posix().replace("/", "__")
            except Exception:
                record_id = file_path.stem

            # Check if already exists
            npy_file = RECORDS_DIR / f"{record_id}.npy"
            meta_file = RECORDS_DIR / f"{record_id}.meta.json"
            label_file = RECORDS_DIR / f"{record_id}.label"

            if npy_file.exists() and meta_file.exists() and label_file.exists():
                processed_files.add(str(file_path))
                continue

            # Read signal
            read_start = time.time()
            if file_path.suffix.lower() == '.hea':
                signal, fs = read_wfdb(file_path)
            else:
                signal, fs = read_mat(file_path)

            # Resample
            if fs is not None and not np.isclose(fs, TARGET_FS):
                signal = resample_signal(signal, fs, TARGET_FS)

            # Normalize
            signal = zscore_normalize(signal)

            # Pad/truncate
            signal = pad_or_truncate(signal, TARGET_SAMPLES)

            # Add channel dimension
            signal = signal[np.newaxis, :]

            # Lookup label
            mapped_label = lookup_mapped_label(file_path, mapping_index)
            label_int = LABEL_TO_INT.get(mapped_label, LABEL_TO_INT["OTHER"])

            # Save files
            save_start = time.time()
            np.save(npy_file, signal, allow_pickle=False)

            with open(label_file, 'w') as f:
                f.write(str(label_int))

            meta = {
                "dataset": rel_path.parts[0] if len(rel_path.parts) > 0 else "unknown",
                "source_path": str(file_path.relative_to(DATASET_DIR)),
                "original_fs": float(fs) if fs is not None else None,
                "mapped_label": mapped_label,
                "label_int": int(label_int)
            }
            with open(meta_file, 'w') as f:
                json.dump(meta, f)

            # Append to manifest
            manifest_entry = {
                "path": f"records/{npy_file.name}",
                "label": int(label_int)
            }
            manifest_fp.write(json.dumps(manifest_entry) + '\n')
            manifest_fp.flush()

            save_times.append(time.time() - save_start)

            # Update counters
            label_counts[label_int] += 1
            processed_files.add(str(file_path))

            # Periodic checkpoint save
            if (idx + 1) % 1000 == 0:
                save_progress_checkpoint(processed_files, idx, len(processed_files))
                elapsed = time.time() - start_time
                speed = len(processed_files) / elapsed
                avg_save_time = np.mean(save_times[-100:]) if save_times else 0
                progress_bar.set_postfix({
                    'rec/s': f'{speed:.1f}',
                    'skip': skipped,
                    'save_ms': f'{avg_save_time*1000:.1f}'
                })

        except Exception as e:
            skipped += 1
            if skipped <= 20:
                logger.error(f"Error processing {file_path.name}: {e}")
            progress_bar.set_postfix({'skipped': skipped})

    progress_bar.close()
    manifest_fp.close()

    # Final checkpoint
    save_progress_checkpoint(processed_files, len(all_files), len(processed_files))

    # Print summary
    elapsed = time.time() - start_time
    logger.info("="*80)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("="*80)
    logger.info(f"Total processed: {len(processed_files):,}")
    logger.info(f"Skipped (errors): {skipped:,}")
    logger.info(f"Time elapsed: {elapsed:.1f}s ({len(processed_files)/elapsed:.1f} rec/s)")
    logger.info(f"\nLabel distribution:")
    for idx, label_name in enumerate(LABEL_ORDER):
        count = label_counts[idx]
        pct = (count / len(processed_files) * 100) if processed_files else 0
        logger.info(f"  {idx}={label_name:5s}: {count:6,d} ({pct:5.1f}%)")

    # Create splits
    create_splits()

    logger.info("\nPreprocessing complete!")


def create_splits():
    """Create stratified train/val/test splits"""
    logger.info("\nCreating stratified splits...")

    # Read manifest
    manifest_file = PROCESSED_DIR / "manifest.jsonl"
    if not manifest_file.exists():
        logger.error("Manifest file not found!")
        return

    manifest = []
    with open(manifest_file, 'r') as f:
        for line in f:
            if line.strip():
                manifest.append(json.loads(line))

    logger.info(f"Loaded {len(manifest)} entries from manifest")

    # Group by label
    by_label = defaultdict(list)
    for entry in manifest:
        by_label[entry['label']].append(entry)

    # Stratified split
    train_list, val_list, test_list = [], [], []
    rng = np.random.default_rng(seed=42)

    for label, entries in by_label.items():
        entries_copy = entries.copy()
        rng.shuffle(entries_copy)
        n = len(entries_copy)
        n_train = int(n * 0.80)
        n_val = int(n * 0.10)

        train_list.extend(entries_copy[:n_train])
        val_list.extend(entries_copy[n_train:n_train + n_val])
        test_list.extend(entries_copy[n_train + n_val:])

    # Save splits
    splits_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "label_order": LABEL_ORDER,
        "label_to_int": LABEL_TO_INT,
        "train": train_list,
        "val": val_list,
        "test": test_list,
        "counts": {
            "train": len(train_list),
            "val": len(val_list),
            "test": len(test_list),
            "total": len(manifest)
        }
    }

    splits_file = PROCESSED_DIR / "splits.json"
    with open(splits_file, 'w') as f:
        json.dump(splits_data, f, indent=2)

    # Save label map
    label_map = {
        "label_to_int": LABEL_TO_INT,
        "int_to_label": INT_TO_LABEL
    }
    label_map_file = PROCESSED_DIR / "label_map.json"
    with open(label_map_file, 'w') as f:
        json.dump(label_map, f, indent=2)

    # Save labels array
    np.save(PROCESSED_DIR / "labels.npy", np.array(LABEL_ORDER, dtype=object))

    logger.info(f"Saved splits to {splits_file}")
    logger.info(f"  Train: {len(train_list):,}")
    logger.info(f"  Val:   {len(val_list):,}")
    logger.info(f"  Test:  {len(test_list):,}")


if __name__ == "__main__":
    main()

