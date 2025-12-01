"""Verify smoke test outputs"""
import json
import numpy as np
from pathlib import Path
import random

ROOT = Path(__file__).parent.parent
PROCESSED_DIR = ROOT / "artifacts" / "processed"
RECORDS_DIR = PROCESSED_DIR / "records"

print("="*80)
print("SMOKE TEST VERIFICATION")
print("="*80)

# Check splits.json
splits_file = PROCESSED_DIR / "splits.json"
if splits_file.exists():
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    print(f"\n✓ splits.json exists")
    print(f"  Train: {splits['counts']['train']}")
    print(f"  Val: {splits['counts']['val']}")
    print(f"  Test: {splits['counts']['test']}")
    print(f"  Total: {splits['counts']['total']}")
else:
    print("\n✗ splits.json missing!")

# Check label_map.json
label_map_file = PROCESSED_DIR / "label_map.json"
if label_map_file.exists():
    with open(label_map_file, 'r') as f:
        label_map = json.load(f)
    print(f"\n✓ label_map.json exists")
    print(f"  Labels: {label_map['label_to_int']}")
else:
    print("\n✗ label_map.json missing!")

# Check labels.npy
labels_file = PROCESSED_DIR / "labels.npy"
if labels_file.exists():
    labels = np.load(labels_file, allow_pickle=True)
    print(f"\n✓ labels.npy exists")
    print(f"  Labels: {list(labels)}")
else:
    print("\n✗ labels.npy missing!")

# Check manifest.jsonl
manifest_file = PROCESSED_DIR / "manifest.jsonl"
if manifest_file.exists():
    with open(manifest_file, 'r') as f:
        manifest = [json.loads(line) for line in f if line.strip()]
    print(f"\n✓ manifest.jsonl exists")
    print(f"  Entries: {len(manifest)}")
else:
    print("\n✗ manifest.jsonl missing!")

# Check record files
record_files = list(RECORDS_DIR.glob("*.npy"))
print(f"\n✓ Found {len(record_files)} .npy files in records/")

# Load and verify 5 random records
if record_files:
    print("\nVerifying 5 random records...")
    samples = random.sample(record_files, min(5, len(record_files)))

    for i, npy_file in enumerate(samples, 1):
        try:
            # Load signal
            signal = np.load(npy_file, allow_pickle=False)

            # Load label
            label_file = npy_file.with_suffix('.label')
            with open(label_file, 'r') as f:
                label = int(f.read().strip())

            # Load metadata
            meta_file = npy_file.parent / f"{npy_file.stem}.meta.json"
            with open(meta_file, 'r') as f:
                meta = json.load(f)

            print(f"\n  {i}. {npy_file.name}")
            print(f"     Signal shape: {signal.shape}")
            print(f"     Label: {label} ({meta.get('mapped_label', 'unknown')})")
            print(f"     Dataset: {meta.get('dataset', 'unknown')}")

            # Verify shape
            assert signal.shape == (1, 5000), f"Expected (1, 5000), got {signal.shape}"
            assert 0 <= label <= 4, f"Invalid label: {label}"

            print(f"     ✓ Valid")

        except Exception as e:
            print(f"\n  {i}. {npy_file.name}")
            print(f"     ✗ Error: {e}")

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)

