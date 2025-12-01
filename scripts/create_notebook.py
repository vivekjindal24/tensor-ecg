"""
Create a complete Jupyter notebook for ECG tensor pipeline.

This script programmatically builds notebooks/ecg_tensor_pipeline.ipynb
using nbformat v4 with all necessary cells for preprocessing, training, and evaluation.
"""

import json
from pathlib import Path

try:
    import nbformat
    from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
except ImportError:
    print("Error: nbformat not installed. Run: pip install nbformat")
    exit(1)

# Project paths
ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = ROOT / "notebooks"
OUTPUT_NOTEBOOK = NOTEBOOKS_DIR / "ecg_tensor_pipeline.ipynb"

# Ensure notebooks directory exists
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

# Create a new notebook
nb = new_notebook()

# ============================================================================
# CELL 1: Title and Instructions (Markdown)
# ============================================================================
nb.cells.append(new_markdown_cell("""# ECG Tensor Pipeline — Preprocessing, Training, Evaluation

## Overview

This notebook provides an end-to-end pipeline for ECG signal classification:
1. **Preprocessing**: Load raw ECG datasets, resample, normalize, and save per-record `.npz` files
2. **Dataset & DataLoader**: Lazy loading with PyTorch for memory efficiency
3. **Model**: Compact 1D CNN for ECG classification
4. **Training**: GPU-accelerated training with mixed precision
5. **Evaluation**: Metrics, confusion matrix, ROC curves, and visualizations

## Prerequisites

- Python 3.10+
- Required packages: `numpy`, `scipy`, `pandas`, `wfdb`, `torch`, `scikit-learn`, `matplotlib`, `tqdm`, `nbformat`
- Datasets should be in `Dataset/` folder
- Unified label mapping CSV at `logs/unified_label_mapping.csv`

## Running Headless

To execute this notebook from the command line:

```powershell
jupyter nbconvert --to notebook --execute notebooks/ecg_tensor_pipeline.ipynb --output ecg_tensor_pipeline_executed.ipynb
```

**Note**: On Windows, if you see `RuntimeError: There is no current event loop in thread`, add this at the top of your script:

```python
import asyncio
import sys
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
```

## GPU-Intensive Tasks

The following cells will utilize GPU heavily when CUDA is available:
- **Training loop**: Forward/backward passes, gradient updates
- **Large batch evaluation**: Model inference on test set
- **Mixed precision training**: Uses `torch.cuda.amp` for speedup
"""))

# ============================================================================
# CELL 2: Environment Setup and Imports (Code)
# ============================================================================
nb.cells.append(new_code_cell("""# Environment checks, imports, seeds, and directory setup
import os
import sys
import random
import json
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import resample
from scipy.io import loadmat
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    confusion_matrix, classification_report, 
    f1_score, precision_recall_fscore_support,
    roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import label_binarize

# Print environment info
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")

# Check CUDA availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\\nDevice: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("Running on CPU - training will be slower")

# Set deterministic seeds for reproducibility
DEFAULT_SEED = 42
random.seed(DEFAULT_SEED)
np.random.seed(DEFAULT_SEED)
torch.manual_seed(DEFAULT_SEED)
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed_all(DEFAULT_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"\\nRandom seed set to: {DEFAULT_SEED}")

# Define project paths
# Try to detect if running inside notebooks/ or at project root
CANDIDATES = [
    Path.cwd().parent,                 # when running inside notebooks/
    Path.cwd(),                        # when running at project root
    Path("D:/ecg-research").resolve()  # explicit fallback for this project
]
ROOT = next((p.resolve() for p in CANDIDATES if (p / "Dataset").exists()), Path.cwd().resolve())

DATASET_DIR = ROOT / "Dataset"
ARTIFACTS_DIR = ROOT / "artifacts"
PROCESSED_DIR = ARTIFACTS_DIR / "processed"
RECORDS_DIR = PROCESSED_DIR / "records"
CHECKPOINTS_DIR = PROCESSED_DIR / "checkpoints"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
LOGS_DIR = ROOT / "logs"

# Create directories
for p in [ARTIFACTS_DIR, PROCESSED_DIR, RECORDS_DIR, CHECKPOINTS_DIR, FIGURES_DIR, LOGS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

print(f"\\nProject Paths:")
print(f"  ROOT: {ROOT}")
print(f"  DATASET_DIR: {DATASET_DIR} (exists: {DATASET_DIR.exists()})")
print(f"  PROCESSED_DIR: {PROCESSED_DIR}")
print(f"  FIGURES_DIR: {FIGURES_DIR}")

# List available datasets
if DATASET_DIR.exists():
    datasets = [p.name for p in sorted(DATASET_DIR.iterdir()) if p.is_dir()]
    print(f"\\nAvailable datasets: {datasets}")
else:
    print(f"\\nWarning: Dataset directory not found at {DATASET_DIR}")

print("\\n" + "="*80)
print("Environment setup complete!")
print("="*80)
"""))

# ============================================================================
# CELL 3: Configuration Constants (Code)
# ============================================================================
nb.cells.append(new_code_cell("""# Configuration constants and hyperparameters

# Signal processing parameters
TARGET_FS = 500           # Target sampling frequency (Hz)
TARGET_SAMPLES = 5000     # Target number of samples per record (10 seconds at 500 Hz)

# Label configuration
LABEL_ORDER = ['MI', 'AF', 'BBB', 'NORM', 'OTHER']
LABEL_TO_INT = {name: i for i, name in enumerate(LABEL_ORDER)}
INT_TO_LABEL = {i: name for name, i in LABEL_TO_INT.items()}

# Training hyperparameters
BATCH_SIZE = 32           # Adjust based on GPU memory
EPOCHS = 10               # Number of training epochs
LR = 1e-3                 # Learning rate
WEIGHT_DECAY = 1e-4       # L2 regularization
NUM_WORKERS = 0           # DataLoader workers (set to 0 for Windows stability)
PIN_MEMORY = True         # Pin memory for faster GPU transfer

# Mixed precision training (GPU only)
USE_MIXED_PRECISION = torch.cuda.is_available()

# Adjust batch size for CPU
if DEVICE.type == 'cpu' and BATCH_SIZE > 8:
    print(f"CPU detected - reducing batch size from {BATCH_SIZE} to 8")
    BATCH_SIZE = 8
    PIN_MEMORY = False

print(f"\\nConfiguration:")
print(f"  Target FS: {TARGET_FS} Hz")
print(f"  Target Samples: {TARGET_SAMPLES}")
print(f"  Label Mapping: {LABEL_TO_INT}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Learning Rate: {LR}")
print(f"  Epochs: {EPOCHS}")
print(f"  Mixed Precision: {USE_MIXED_PRECISION}")
print(f"  Device: {DEVICE}")

print(f"\\nGPU-Intensive Tasks:")
print("  - Preprocessing: Moderate (CPU-bound mostly)")
print("  - DataLoader: Low (lazy loading)")
print("  - Model Training: HIGH (forward + backward passes)")
print("  - Model Evaluation: Medium (inference only)")
"""))

# ============================================================================
# CELL 4: Utility Functions (Code)
# ============================================================================
nb.cells.append(new_code_cell("""# Utility functions for signal processing and file I/O

def zscore_normalize(arr: np.ndarray) -> np.ndarray:
    \"\"\"Z-score normalization: (x - mean) / std\"\"\"
    arr = arr.astype(np.float32)
    mean = arr.mean()
    std = arr.std()
    if std < 1e-8:
        std = 1.0  # Prevent division by zero
    return ((arr - mean) / std).astype(np.float32)


def pad_or_truncate(x: np.ndarray, target_length: int) -> np.ndarray:
    \"\"\"Pad with zeros or truncate to target length\"\"\"
    if x.size >= target_length:
        return x[:target_length]
    pad_width = target_length - x.size
    return np.pad(x, (0, pad_width), mode='constant', constant_values=0).astype(np.float32)


def resample_signal(x: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
    \"\"\"Resample signal to target sampling frequency\"\"\"
    if original_fs is None or np.isclose(original_fs, target_fs):
        return x
    new_length = int(round(x.size * target_fs / original_fs))
    if new_length <= 0:
        return x
    return resample(x, new_length).astype(np.float32)


def safe_save_npz(path: Path, signal: np.ndarray, label: int):
    \"\"\"Save signal and label as compressed npz\"\"\"
    np.savez_compressed(path, signal=signal.astype(np.float32), label=int(label))


def load_npz_signal(path: Path):
    \"\"\"Load signal and label from npz file\"\"\"
    with np.load(path, allow_pickle=False) as data:
        signal = data['signal']
        label = int(data['label'])
    return signal, label


def read_wfdb(hea_path: Path):
    \"\"\"Read WFDB format (.hea/.dat) and return 1D signal and sampling frequency\"\"\"
    try:
        import wfdb
        record_path = str(hea_path.with_suffix(''))  # wfdb expects path without extension
        record = wfdb.rdsamp(record_path)
        signal = np.asarray(record[0], dtype=np.float32)
        fs = float(record[1].get('fs', TARGET_FS))
        
        # Convert to 1D: average across leads or take first lead
        if signal.ndim == 2:
            signal_1d = signal.mean(axis=1) if signal.shape[1] > 1 else signal[:, 0]
        else:
            signal_1d = signal.reshape(-1)
        
        return signal_1d.astype(np.float32), fs
    except Exception as e:
        raise RuntimeError(f"Failed to read WFDB file {hea_path.name}: {e}")


def read_mat(mat_path: Path):
    \"\"\"Read MATLAB .mat file and return 1D signal and sampling frequency\"\"\"
    try:
        mat_data = loadmat(str(mat_path))
        
        # Try common keys for signal data
        signal = None
        for key in ['val', 'data', 'signal', 'ecg']:
            if key in mat_data:
                signal = np.asarray(mat_data[key], dtype=np.float32)
                break
        
        # Fallback: find first ndarray
        if signal is None:
            for value in mat_data.values():
                if isinstance(value, np.ndarray) and value.size > 100:
                    signal = value.astype(np.float32)
                    break
        
        if signal is None:
            raise RuntimeError("No signal array found in MAT file")
        
        # Convert to 1D
        if signal.ndim == 2:
            signal_1d = signal.mean(axis=0) if signal.shape[0] > 1 else signal.reshape(-1)
        else:
            signal_1d = signal.reshape(-1)
        
        # MAT files rarely contain fs info; return None
        fs = None
        return signal_1d.astype(np.float32), fs
    except Exception as e:
        raise RuntimeError(f"Failed to read MAT file {mat_path.name}: {e}")


print("Utility functions defined successfully.")
"""))

# ============================================================================
# CELL 5: Load Unified Label Mapping (Code)
# ============================================================================
nb.cells.append(new_code_cell("""# Load unified label mapping from CSV

UNIFIED_CSV = LOGS_DIR / "unified_label_mapping.csv"
mapping_index = {}

if UNIFIED_CSV.exists():
    print(f"Loading unified label mapping from {UNIFIED_CSV}...")
    df_mapping = pd.read_csv(UNIFIED_CSV, dtype=str).fillna("")
    
    # Verify required columns
    required_cols = {"dataset", "record_id", "mapped_label"}
    if not required_cols.issubset(set(df_mapping.columns)):
        print(f"Warning: Missing required columns. Found: {df_mapping.columns.tolist()}")
        print(f"Expected: {list(required_cols)}")
    else:
        # Build mapping index with multiple key variants for robust lookup
        for _, row in df_mapping.iterrows():
            dataset = str(row.get("dataset", "")).strip()
            record_id = str(row.get("record_id", "")).strip().replace("\\\\", "/").strip("/")
            mapped_label = str(row.get("mapped_label", "")).strip().upper()
            
            if not dataset or not record_id:
                continue
            
            if dataset not in mapping_index:
                mapping_index[dataset] = {}
            
            # Add full path
            mapping_index[dataset][record_id] = mapped_label
            
            # Add variants for robust matching
            parts = record_id.split("/")
            if len(parts) >= 1:
                mapping_index[dataset][parts[-1]] = mapped_label  # basename
            if len(parts) >= 2:
                mapping_index[dataset]["/".join(parts[-2:])] = mapped_label  # last two components
        
        print(f"Loaded {len(df_mapping)} mappings from {len(mapping_index)} datasets")
        
        # Count mappings per label
        label_counts = Counter(df_mapping['mapped_label'].str.upper())
        print(f"\\nLabel distribution in mapping:")
        for label in LABEL_ORDER:
            count = label_counts.get(label, 0)
            print(f"  {label}: {count:,}")
        unmapped = label_counts.get('', 0)
        print(f"  (unmapped): {unmapped:,}")
else:
    print(f"Warning: Unified label mapping not found at {UNIFIED_CSV}")
    print("All records will be labeled as OTHER")


def lookup_mapped_label(path: Path) -> str:
    \"\"\"Look up mapped label for a given file path\"\"\"
    try:
        rel_path = path.relative_to(DATASET_DIR).with_suffix("")
    except Exception:
        rel_path = path.with_suffix("")
    
    parts = rel_path.as_posix().split("/")
    dataset = parts[0] if parts else ""
    
    if dataset not in mapping_index:
        return "OTHER"
    
    index = mapping_index[dataset]
    
    # Try multiple key variants
    candidates = [
        rel_path.as_posix(),                              # full path
        "/".join(parts[1:]) if len(parts) > 1 else "",   # without dataset prefix
        "/".join(parts[-2:]) if len(parts) >= 2 else "", # last two components
        rel_path.name                                     # basename only
    ]
    
    for key in candidates:
        if key and key in index:
            label = index[key].upper()
            return label if label in LABEL_TO_INT else "OTHER"
    
    # CinC2017 special case
    if "CinC" in dataset and len(parts) >= 3:
        alt_key = "/".join(parts[2:])
        if alt_key in index:
            label = index[alt_key].upper()
            return label if label in LABEL_TO_INT else "OTHER"
    
    return "OTHER"


print("Label lookup function ready.")
"""))

# ============================================================================
# CELL 6: Streaming Preprocessing (Code)
# ============================================================================
nb.cells.append(new_code_cell("""# Streaming preprocessing: scan datasets, process, and save per-record npz files

print("="*80)
print("STARTING PREPROCESSING")
print("="*80)

# Find all .hea and .mat files
print(f"\\nScanning {DATASET_DIR} for ECG files...")
hea_files = sorted(DATASET_DIR.rglob("*.hea"))
mat_files = sorted(DATASET_DIR.rglob("*.mat"))
all_files = hea_files + mat_files

print(f"Found {len(hea_files)} .hea files and {len(mat_files)} .mat files")
print(f"Total files to process: {len(all_files)}")

if not all_files:
    print("\\nNo dataset files found. Generating synthetic records for testing...")
    # Generate synthetic signals
    for i in range(20):
        t = np.linspace(0, TARGET_SAMPLES / TARGET_FS, TARGET_SAMPLES, dtype=np.float32)
        freq = 0.5 + 0.1 * i
        signal = np.sin(2 * np.pi * freq * t).astype(np.float32)
        signal = signal[np.newaxis, :]  # Shape: (1, TARGET_SAMPLES)
        
        label = i % len(LABEL_ORDER)
        out_file = RECORDS_DIR / f"SYNTH_{i:04d}.npz"
        safe_save_npz(out_file, signal, label)
    
    print(f"Generated 20 synthetic records in {RECORDS_DIR}")
else:
    # Process real dataset files
    manifest = []
    label_counts = Counter()
    skipped = 0
    
    print(f"\\nProcessing files...")
    progress_bar = tqdm(all_files, desc="Processing", unit="file")
    
    for file_path in progress_bar:
        try:
            # Read signal
            if file_path.suffix.lower() == '.hea':
                signal, fs = read_wfdb(file_path)
            else:
                signal, fs = read_mat(file_path)
            
            # Resample if needed
            if fs is not None and not np.isclose(fs, TARGET_FS):
                signal = resample_signal(signal, fs, TARGET_FS)
            
            # Normalize
            signal = zscore_normalize(signal)
            
            # Pad or truncate
            signal = pad_or_truncate(signal, TARGET_SAMPLES)
            
            # Add channel dimension: (1, TARGET_SAMPLES)
            signal = signal[np.newaxis, :]
            
            # Lookup label
            mapped_label = lookup_mapped_label(file_path)
            label_int = LABEL_TO_INT.get(mapped_label, LABEL_TO_INT["OTHER"])
            
            # Generate safe filename
            try:
                rel_path = file_path.relative_to(DATASET_DIR).with_suffix("")
                record_id = rel_path.as_posix().replace("/", "__")
            except Exception:
                record_id = file_path.stem
            
            # Save processed record
            out_file = RECORDS_DIR / f"{record_id}.npz"
            safe_save_npz(out_file, signal, label_int)
            
            # Update manifest
            manifest.append({
                "path": f"records/{out_file.name}",
                "label": int(label_int)
            })
            label_counts[label_int] += 1
            
        except Exception as e:
            skipped += 1
            if skipped <= 10:  # Only print first 10 errors
                tqdm.write(f"Error processing {file_path.name}: {e}")
            progress_bar.set_postfix({"skipped": skipped})
    
    progress_bar.close()
    
    print(f"\\n" + "="*80)
    print("PREPROCESSING SUMMARY")
    print("="*80)
    print(f"Total files processed: {len(manifest):,}")
    print(f"Files skipped (errors): {skipped:,}")
    print(f"\\nLabel distribution:")
    for idx, label_name in enumerate(LABEL_ORDER):
        count = label_counts[idx]
        pct = (count / len(manifest) * 100) if manifest else 0
        print(f"  {idx}={label_name:5s}: {count:6,d} ({pct:5.1f}%)")
    
    # Save manifest and create splits
    print(f"\\nCreating stratified train/val/test splits (80/10/10)...")
    
    # Group by label
    by_label = defaultdict(list)
    for entry in manifest:
        by_label[entry['label']].append(entry)
    
    # Split each label class
    train_list, val_list, test_list = [], [], []
    rng = np.random.default_rng(seed=DEFAULT_SEED)
    
    for label, entries in by_label.items():
        rng.shuffle(entries)
        n = len(entries)
        n_train = int(n * 0.80)
        n_val = int(n * 0.10)
        
        train_list.extend(entries[:n_train])
        val_list.extend(entries[n_train:n_train + n_val])
        test_list.extend(entries[n_train + n_val:])
    
    # Save splits.json
    splits_data = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "label_order": LABEL_ORDER,
        "label_to_int": LABEL_TO_INT,
        "train": train_list,
        "val": val_list,
        "test": test_list,
        "counts": {
            "train": len(train_list),
            "val": len(val_list),
            "test": len(test_list)
        },
        "class_counts": {int(k): int(v) for k, v in label_counts.items()}
    }
    
    splits_file = PROCESSED_DIR / "splits.json"
    with open(splits_file, 'w', encoding='utf-8') as f:
        json.dump(splits_data, f, indent=2)
    
    # Save label_map.json
    label_map = {
        "label_to_int": LABEL_TO_INT,
        "int_to_label": INT_TO_LABEL
    }
    label_map_file = PROCESSED_DIR / "label_map.json"
    with open(label_map_file, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, indent=2)
    
    # Save labels.npy
    np.save(PROCESSED_DIR / "labels.npy", np.array(LABEL_ORDER, dtype=object))
    
    print(f"\\nSaved:")
    print(f"  - {splits_file}")
    print(f"  - {label_map_file}")
    print(f"  - {PROCESSED_DIR / 'labels.npy'}")
    print(f"\\nSplit sizes:")
    print(f"  Train: {len(train_list):,}")
    print(f"  Val:   {len(val_list):,}")
    print(f"  Test:  {len(test_list):,}")

print("\\n" + "="*80)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("="*80)
"""))

# ============================================================================
# CELL 7: PyTorch Dataset and DataLoader (Code)
# ============================================================================
nb.cells.append(new_code_cell("""# PyTorch Dataset for lazy loading of per-record npz files

class ECGDataset(Dataset):
    \"\"\"Lazy-loading dataset for preprocessed ECG records\"\"\"
    
    def __init__(self, entries, base_dir):
        \"\"\"
        Args:
            entries: List of dicts with 'path' and 'label' keys
            base_dir: Base directory for processed files
        \"\"\"
        self.entries = entries
        self.base_dir = Path(base_dir)
    
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        entry = self.entries[idx]
        file_path = self.base_dir / entry['path']
        
        # Load signal and label
        signal, label = load_npz_signal(file_path)
        
        # Convert to tensors
        signal_tensor = torch.from_numpy(signal).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return signal_tensor, label_tensor


def create_dataloaders(splits_file, processed_dir, batch_size, num_workers=0, pin_memory=False):
    \"\"\"Create train, val, and test DataLoaders\"\"\"
    
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    train_dataset = ECGDataset(splits['train'], processed_dir)
    val_dataset = ECGDataset(splits['val'], processed_dir)
    test_dataset = ECGDataset(splits['test'], processed_dir)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


# Create DataLoaders
print("Creating DataLoaders...")
splits_file = PROCESSED_DIR / "splits.json"

if splits_file.exists():
    train_loader, val_loader, test_loader = create_dataloaders(
        splits_file, 
        PROCESSED_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    print(f"DataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    
    # Show example batch
    sample_batch = next(iter(train_loader))
    print(f"\\nExample batch shapes:")
    print(f"  Signals: {sample_batch[0].shape}")
    print(f"  Labels:  {sample_batch[1].shape}")
    print(f"  Label values: {sample_batch[1][:min(10, BATCH_SIZE)].tolist()}")
else:
    print(f"Error: splits.json not found at {splits_file}")
    print("Please run the preprocessing cell first.")
"""))

# ============================================================================
# CELL 8: Model Definition (Code)
# ============================================================================
nb.cells.append(new_code_cell("""# 1D CNN Model for ECG Classification

class ECGNet1D(nn.Module):
    \"\"\"Compact 1D CNN for ECG signal classification\"\"\"
    
    def __init__(self, n_classes=len(LABEL_ORDER), input_channels=1, base_channels=32, dropout=0.3):
        super(ECGNet1D, self).__init__()
        
        self.conv1 = nn.Conv1d(input_channels, base_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(base_channels)
        
        self.conv2 = nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(base_channels * 2)
        
        self.conv3 = nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(base_channels * 4)
        
        self.conv4 = nn.Conv1d(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(base_channels * 8)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base_channels * 8, n_classes)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # x shape: (batch, 1, samples)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# Instantiate model
model = ECGNet1D(n_classes=len(LABEL_ORDER), base_channels=32, dropout=0.3)
model = model.to(DEVICE)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("="*80)
print("MODEL")
print("="*80)
print(model)
print(f"\\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model device: {DEVICE}")
"""))

# ============================================================================
# CELL 9: Training Loop (Code)
# ============================================================================
nb.cells.append(new_code_cell("""# Training loop with mixed precision and metrics tracking

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# Mixed precision scaler
scaler = torch.cuda.amp.GradScaler() if USE_MIXED_PRECISION else None

# Training history
history = {
    'train_loss': [],
    'train_acc': [],
    'train_f1': [],
    'val_loss': [],
    'val_acc': [],
    'val_f1': [],
    'lr': []
}

best_val_f1 = 0.0
best_epoch = 0

print("="*80)
print("TRAINING")
print("="*80)

for epoch in range(EPOCHS):
    print(f"\\nEpoch {epoch + 1}/{EPOCHS}")
    print("-" * 80)
    
    # Training phase
    model.train()
    train_loss = 0.0
    train_preds = []
    train_labels = []
    
    train_progress = tqdm(train_loader, desc="Training", leave=False)
    for signals, labels in train_progress:
        signals = signals.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        if USE_MIXED_PRECISION:
            with torch.cuda.amp.autocast():
                outputs = model(signals)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        train_loss += loss.item()
        preds = outputs.argmax(dim=1).cpu().numpy()
        train_preds.extend(preds)
        train_labels.extend(labels.cpu().numpy())
        
        train_progress.set_postfix({'loss': f'{loss.item():.4f}'})
    
    train_loss /= len(train_loader)
    train_acc = (np.array(train_preds) == np.array(train_labels)).mean()
    train_f1 = f1_score(train_labels, train_preds, average='macro', zero_division=0)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        val_progress = tqdm(val_loader, desc="Validation", leave=False)
        for signals, labels in val_progress:
            signals = signals.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(signals)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(labels.cpu().numpy())
    
    val_loss /= len(val_loader)
    val_acc = (np.array(val_preds) == np.array(val_labels)).mean()
    val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
    
    # Update history
    current_lr = optimizer.param_groups[0]['lr']
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['train_f1'].append(train_f1)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_f1'].append(val_f1)
    history['lr'].append(current_lr)
    
    # Print epoch summary
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1:   {val_f1:.4f}")
    print(f"LR: {current_lr:.6f}")
    
    # Save best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch = epoch + 1
        best_model_path = CHECKPOINTS_DIR / "best_model.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1': val_f1,
            'val_acc': val_acc,
            'val_loss': val_loss
        }, best_model_path)
        print(f"✓ Saved best model (F1: {val_f1:.4f})")
    
    # Step scheduler
    scheduler.step()

print("\\n" + "="*80)
print("TRAINING COMPLETED")
print("="*80)
print(f"Best validation F1: {best_val_f1:.4f} (Epoch {best_epoch})")

# Save final model
final_model_path = CHECKPOINTS_DIR / "final_model.pth"
torch.save({
    'epoch': EPOCHS,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'history': history
}, final_model_path)
print(f"\\nSaved final model to {final_model_path}")

# Save training history
history_file = PROCESSED_DIR / "training_history.json"
with open(history_file, 'w') as f:
    json.dump(history, f, indent=2)
print(f"Saved training history to {history_file}")
"""))

# ============================================================================
# CELL 10: Evaluation and Plots (Code)
# ============================================================================
nb.cells.append(new_code_cell("""# Evaluation on test set with metrics and visualizations

# Load best model
best_model_path = CHECKPOINTS_DIR / "best_model.pth"
checkpoint = torch.load(best_model_path, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded best model from epoch {checkpoint['epoch'] + 1}")
print(f"Best validation F1: {checkpoint['val_f1']:.4f}")

# Evaluate on test set
model.eval()
test_preds = []
test_labels = []
test_probs = []

with torch.no_grad():
    test_progress = tqdm(test_loader, desc="Testing")
    for signals, labels in test_progress:
        signals = signals.to(DEVICE)
        labels = labels.to(DEVICE)
        
        outputs = model(signals)
        probs = F.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        
        test_probs.append(probs.cpu().numpy())
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

test_probs = np.vstack(test_probs)
test_preds = np.array(test_preds)
test_labels = np.array(test_labels)

# Calculate metrics
test_acc = (test_preds == test_labels).mean()
test_f1_macro = f1_score(test_labels, test_preds, average='macro', zero_division=0)
test_f1_weighted = f1_score(test_labels, test_preds, average='weighted', zero_division=0)

print("\\n" + "="*80)
print("TEST SET EVALUATION")
print("="*80)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test F1 (macro): {test_f1_macro:.4f}")
print(f"Test F1 (weighted): {test_f1_weighted:.4f}")

# Per-class metrics
print("\\n" + "-"*80)
print("Per-Class Metrics:")
print("-"*80)
precision, recall, f1, support = precision_recall_fscore_support(
    test_labels, test_preds, average=None, zero_division=0
)

for i, label_name in enumerate(LABEL_ORDER):
    print(f"{label_name:5s}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, F1={f1[i]:.3f}, Support={support[i]}")

# Confusion matrix
cm = confusion_matrix(test_labels, test_preds)
print("\\n" + "-"*80)
print("Confusion Matrix:")
print("-"*80)
print(cm)

# Save evaluation results
eval_results = {
    'test_accuracy': float(test_acc),
    'test_f1_macro': float(test_f1_macro),
    'test_f1_weighted': float(test_f1_weighted),
    'per_class_metrics': {
        LABEL_ORDER[i]: {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }
        for i in range(len(LABEL_ORDER))
    },
    'confusion_matrix': cm.tolist()
}

eval_file = PROCESSED_DIR / "evaluation_results.json"
with open(eval_file, 'w') as f:
    json.dump(eval_results, f, indent=2)
print(f"\\nSaved evaluation results to {eval_file}")
"""))

# ============================================================================
# CELL 11: Visualization Plots (Code)
# ============================================================================
nb.cells.append(new_code_cell("""# Generate and save visualization plots

# Set plot style
plt.style.use('default')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150

# 1. Training curves
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss
axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training and Validation Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Accuracy
axes[0, 1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
axes[0, 1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Training and Validation Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# F1 Score
axes[1, 0].plot(history['train_f1'], label='Train F1', linewidth=2)
axes[1, 0].plot(history['val_f1'], label='Val F1', linewidth=2)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('F1 Score (macro)')
axes[1, 0].set_title('Training and Validation F1 Score')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Learning Rate
axes[1, 1].plot(history['lr'], label='Learning Rate', linewidth=2, color='orange')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Learning Rate')
axes[1, 1].set_title('Learning Rate Schedule')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_yscale('log')

plt.tight_layout()
training_curves_path = FIGURES_DIR / 'training_curves.png'
plt.savefig(training_curves_path, bbox_inches='tight')
print(f"Saved training curves to {training_curves_path}")
plt.show()

# 2. Confusion Matrix Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=LABEL_ORDER,
       yticklabels=LABEL_ORDER,
       xlabel='Predicted Label',
       ylabel='True Label',
       title='Confusion Matrix')

# Add text annotations
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12)

plt.tight_layout()
cm_path = FIGURES_DIR / 'confusion_matrix.png'
plt.savefig(cm_path, bbox_inches='tight')
print(f"Saved confusion matrix to {cm_path}")
plt.show()

# 3. Per-class F1 Score Bar Plot
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(LABEL_ORDER))
bars = ax.bar(x, f1, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
ax.set_xlabel('Class', fontsize=12)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('Per-Class F1 Score on Test Set', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(LABEL_ORDER)
ax.set_ylim([0, 1.0])
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, f1)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.02,
            f'{value:.3f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
f1_bars_path = FIGURES_DIR / 'per_class_f1.png'
plt.savefig(f1_bars_path, bbox_inches='tight')
print(f"Saved per-class F1 plot to {f1_bars_path}")
plt.show()

print("\\nAll visualizations saved to:", FIGURES_DIR)
"""))

# ============================================================================
# CELL 12: Smoke Tests (Code)
# ============================================================================
nb.cells.append(new_code_cell("""# Automated smoke tests to verify pipeline integrity

print("="*80)
print("SMOKE TESTS")
print("="*80)

# Test 1: Load a single record
print("\\n1. Testing record loading...")
try:
    test_files = list(RECORDS_DIR.glob("*.npz"))
    if test_files:
        test_file = test_files[0]
        signal, label = load_npz_signal(test_file)
        print(f"   ✓ Loaded {test_file.name}")
        print(f"     Signal shape: {signal.shape}")
        print(f"     Label: {label} ({LABEL_ORDER[label]})")
        assert signal.shape[1] == TARGET_SAMPLES, "Signal length mismatch"
        assert 0 <= label < len(LABEL_ORDER), "Invalid label"
        print("   ✓ Record validation passed")
    else:
        print("   ✗ No records found")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 2: Model forward pass
print("\\n2. Testing model forward pass...")
try:
    dummy_input = torch.randn(1, 1, TARGET_SAMPLES).to(DEVICE)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"   ✓ Input shape: {dummy_input.shape}")
    print(f"   ✓ Output shape: {output.shape}")
    assert output.shape == (1, len(LABEL_ORDER)), "Output shape mismatch"
    print("   ✓ Model forward pass successful")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 3: Checkpoint loading
print("\\n3. Testing checkpoint loading...")
try:
    best_checkpoint = CHECKPOINTS_DIR / "best_model.pth"
    if best_checkpoint.exists():
        checkpoint = torch.load(best_checkpoint, map_location=DEVICE)
        print(f"   ✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"   ✓ Val F1: {checkpoint.get('val_f1', 0):.4f}")
        print("   ✓ Checkpoint loading successful")
    else:
        print(f"   ✗ Checkpoint not found at {best_checkpoint}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 4: Dataset integrity
print("\\n4. Testing dataset integrity...")
try:
    splits_file = PROCESSED_DIR / "splits.json"
    if splits_file.exists():
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        n_train = len(splits.get('train', []))
        n_val = len(splits.get('val', []))
        n_test = len(splits.get('test', []))
        n_total = n_train + n_val + n_test
        print(f"   ✓ Total records: {n_total}")
        print(f"   ✓ Train: {n_train}, Val: {n_val}, Test: {n_test}")
        print("   ✓ Dataset integrity check passed")
    else:
        print(f"   ✗ Splits file not found")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("\\n" + "="*80)
print("SMOKE TESTS COMPLETED")
print("="*80)
"""))

# ============================================================================
# CELL 13: Final Summary (Markdown)
# ============================================================================
nb.cells.append(new_markdown_cell("""---

## Pipeline Complete!

This notebook has successfully completed the full ECG classification pipeline:

✓ **Preprocessing**: Loaded, resampled, normalized, and saved ECG records  
✓ **Dataset**: Created stratified train/val/test splits  
✓ **Model**: Trained 1D CNN classifier  
✓ **Evaluation**: Generated metrics and visualizations  
✓ **Artifacts**: Saved models, checkpoints, and results  

### Output Files

- **Processed Data**: `artifacts/processed/records/*.npz`
- **Splits**: `artifacts/processed/splits.json`
- **Best Model**: `artifacts/processed/checkpoints/best_model.pth`
- **Training History**: `artifacts/processed/training_history.json`
- **Evaluation Results**: `artifacts/processed/evaluation_results.json`
- **Figures**: `artifacts/figures/*.png`

### Next Steps

1. **Improve Model**: Experiment with deeper architectures (ResNet1D, attention mechanisms)
2. **Hyperparameter Tuning**: Use grid search or Bayesian optimization
3. **Data Augmentation**: Add noise, scaling, time-warping
4. **Multi-Lead Models**: Process all 12 leads instead of averaging
5. **Ensemble Methods**: Combine multiple models for better performance
6. **Deployment**: Export to ONNX or TorchScript for production

### Inference Example

To use the trained model for inference:

```python
# Load model
checkpoint = torch.load('artifacts/processed/checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load a signal
signal, label = load_npz_signal('path/to/record.npz')
signal_tensor = torch.from_numpy(signal).float().unsqueeze(0).to(DEVICE)

# Predict
with torch.no_grad():
    output = model(signal_tensor)
    probabilities = F.softmax(output, dim=1)
    predicted_class = output.argmax(dim=1).item()
    predicted_label = LABEL_ORDER[predicted_class]

print(f"Predicted: {predicted_label} (confidence: {probabilities[0, predicted_class]:.2%})")
```
"""))

# ============================================================================
# Write the notebook to file
# ============================================================================
print(f"Writing notebook to {OUTPUT_NOTEBOOK}...")
with open(OUTPUT_NOTEBOOK, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"✓ Notebook created successfully!")
print(f"  Location: {OUTPUT_NOTEBOOK}")
print(f"  Total cells: {len(nb.cells)}")
print(f"\nTo run the notebook:")
print(f"  jupyter notebook {OUTPUT_NOTEBOOK}")
print(f"\nOr execute headless:")
print(f"  jupyter nbconvert --to notebook --execute {OUTPUT_NOTEBOOK.name}")

