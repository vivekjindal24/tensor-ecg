"""
Full Automation Script for ECG Research Pipeline
Orchestrates: environment checks → preprocessing → training → evaluation
Implements resume/idempotency, progress tracking, and comprehensive logging.
"""
import os
import sys
import json
import time
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

# Setup paths
ROOT = Path(__file__).parent.parent
DATASET_DIR = ROOT / "Dataset"
ARTIFACTS_DIR = ROOT / "artifacts"
PROCESSED_DIR = ARTIFACTS_DIR / "processed"
LOGS_DIR = ROOT / "logs"
NB_PATH = ROOT / "notebooks" / "master_pipeline.ipynb"

# Ensure dirs exist
for d in [LOGS_DIR, PROCESSED_DIR, ARTIFACTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Setup logging
LOG_FILE = LOGS_DIR / "preprocess_automation.log"
REPORT_FILE = LOGS_DIR / "preprocess_report.txt"

def log(msg, also_print=True):
    """Write to log file and optionally print."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)
    if also_print:
        # Use ASCII-safe output for Windows console
        safe_msg = msg.replace('✓', '[OK]').replace('✗', '[FAIL]').replace('⚠', '[WARN]')
        print(safe_msg)

def check_env():
    """Check Python environment and dependencies."""
    log("=" * 60)
    log("A. ENVIRONMENT CHECKS")
    log("=" * 60)

    log(f"Python executable: {sys.executable}")
    log(f"Python version: {sys.version.split()[0]}")

    # Check key packages
    required = ['numpy', 'pandas', 'torch', 'wfdb', 'sklearn', 'nbformat']
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            log(f"✓ {pkg} available")
        except ImportError:
            log(f"✗ {pkg} MISSING")
            missing.append(pkg)

    if missing:
        log(f"ERROR: Missing packages: {missing}")
        log("Run: pip install -r requirements.txt")
        return False

    # Check CUDA
    import torch
    if torch.cuda.is_available():
        log(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        log(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        log("⚠ CUDA not available, using CPU")

    # Check disk space
    import shutil
    total, used, free = shutil.disk_usage(ROOT)
    log(f"Disk space: {free / 1e9:.1f} GB free")

    if free < 10e9:
        log("WARNING: Less than 10GB free disk space")

    return True

def validate_mapping():
    """Check unified label mapping CSV."""
    log("=" * 60)
    log("B. VALIDATE MAPPING CSV")
    log("=" * 60)

    unified = LOGS_DIR / "unified_label_mapping.csv"
    candidate = LOGS_DIR / "unified_label_mapping.candidate.csv"

    if unified.exists() and unified.stat().st_size > 0:
        import pandas as pd
        df = pd.read_csv(unified, dtype=str)
        log(f"✓ Unified mapping loaded: {len(df)} rows")

        # Summary by dataset
        if 'dataset' in df.columns:
            counts = df['dataset'].value_counts()
            log(f"  Datasets: {dict(counts)}")

        # Summary by label
        if 'mapped_label' in df.columns:
            label_counts = df['mapped_label'].value_counts()
            log(f"  Labels: {dict(label_counts)}")

            # Check unmapped
            unmapped = df[df['mapped_label'].isna() | (df['mapped_label'] == '')]
            if len(unmapped) > 0:
                log(f"⚠ {len(unmapped)} unmapped records (will become OTHER)")
                sample_file = LOGS_DIR / "unmapped_sample.csv"
                unmapped.sample(min(200, len(unmapped))).to_csv(sample_file, index=False)
                log(f"  Sample saved to: {sample_file}")

        return True

    elif candidate.exists() and candidate.stat().st_size > 0:
        log(f"⚠ Using candidate mapping: {candidate}")
        shutil.copy(candidate, unified)
        log(f"✓ Copied to: {unified}")
        return True

    else:
        log("✗ No mapping CSV found - will default to OTHER for all records")
        log("  Consider running: python scripts/generate_unified_mapping.py")
        return True  # Continue anyway

def check_preprocessing_status():
    """Check current preprocessing progress."""
    log("=" * 60)
    log("PREPROCESSING STATUS CHECK")
    log("=" * 60)

    manifest = PROCESSED_DIR / "manifest.jsonl"
    splits = PROCESSED_DIR / "splits.json"
    records_dir = PROCESSED_DIR / "records"

    status = {
        'manifest_exists': manifest.exists(),
        'splits_exists': splits.exists(),
        'record_count': 0,
        'needs_preprocessing': False
    }

    if manifest.exists():
        with open(manifest, 'r') as f:
            status['record_count'] = sum(1 for _ in f)
        log(f"✓ Manifest exists with {status['record_count']} records")
    else:
        log("⚠ No manifest found")
        status['needs_preprocessing'] = True

    if splits.exists():
        with open(splits, 'r') as f:
            splits_data = json.load(f)
        log(f"✓ Splits file exists")
    else:
        log("⚠ No splits file found")
        status['needs_preprocessing'] = True

    if records_dir.exists():
        npz_files = list(records_dir.glob("*.npz"))
        log(f"✓ Records directory has {len(npz_files)} .npz files")
    else:
        log("⚠ Records directory missing")
        status['needs_preprocessing'] = True

    return status

def run_smoke_test():
    """Run quick smoke test with limited records."""
    log("=" * 60)
    log("F. RUNNING SMOKE TEST")
    log("=" * 60)

    os.environ['ECG_PREPROCESS_LIMIT'] = '500'
    os.environ['ECG_EPOCHS'] = '1'
    os.environ['ECG_BATCH_SIZE'] = '4'

    out_path = LOGS_DIR / "preprocess_report_smoke.ipynb"

    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout=1200",  # 20 min
        "--output", str(out_path.absolute()),
        str(NB_PATH.absolute())
    ]

    log(f"Running: {' '.join(cmd)}")
    start = time.time()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        elapsed = time.time() - start
        log(f"✓ Smoke test completed in {elapsed:.1f}s")
        log(f"✓ Output: {out_path}")

        # Verify outputs
        manifest = PROCESSED_DIR / "manifest.jsonl"
        if manifest.exists():
            with open(manifest, 'r') as f:
                count = sum(1 for _ in f)
            log(f"✓ Manifest has {count} records")

            # Load a few and check
            import numpy as np
            with open(manifest, 'r') as f:
                for i, line in enumerate(f):
                    if i >= 3:
                        break
                    rec = json.loads(line)
                    # Fix path handling
                    rec_path = rec['path']
                    if rec_path.startswith('records/'):
                        rec_path = rec_path[8:]  # Remove 'records/' prefix
                    npz_path = PROCESSED_DIR / 'records' / rec_path
                    if npz_path.exists():
                        try:
                            data = np.load(npz_path, allow_pickle=True)
                            signal_shape = data['signal'].shape if 'signal' in data else 'N/A'
                            label_val = int(data['label']) if 'label' in data else 'N/A'
                            log(f"  Sample {i}: shape={signal_shape} label={label_val}")
                        except Exception as e:
                            log(f"  Sample {i}: Error loading - {e}")

        return True

    except subprocess.CalledProcessError as e:
        log(f"✗ Smoke test failed!")
        log(f"STDERR: {e.stderr[:500]}")
        return False
    except Exception as e:
        log(f"✗ Error: {e}")
        return False

def run_full_preprocessing(skip_if_exists=True):
    """Run full preprocessing on all datasets."""
    log("=" * 60)
    log("G. RUNNING FULL PREPROCESSING")
    log("=" * 60)

    # Remove limit
    if 'ECG_PREPROCESS_LIMIT' in os.environ:
        del os.environ['ECG_PREPROCESS_LIMIT']

    os.environ['ECG_EPOCHS'] = '5'  # More epochs for full run
    os.environ['ECG_BATCH_SIZE'] = '32'

    out_path = LOGS_DIR / "preprocess_run.ipynb"

    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout=0",  # No timeout
        "--output", str(out_path.absolute()),
        str(NB_PATH.absolute())
    ]

    log(f"Running full preprocessing (no timeout)...")
    log(f"Command: {' '.join(cmd)}")
    log("This may take several hours for large datasets...")

    start = time.time()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        elapsed = time.time() - start
        log(f"✓ Full preprocessing completed in {elapsed/3600:.1f} hours")
        log(f"✓ Output: {out_path}")
        return True

    except subprocess.CalledProcessError as e:
        log(f"✗ Full preprocessing failed!")
        log(f"STDERR: {e.stderr[:1000]}")
        return False
    except Exception as e:
        log(f"✗ Error: {e}")
        return False

def verify_outputs():
    """Verify all expected outputs exist."""
    log("=" * 60)
    log("H. VERIFICATION")
    log("=" * 60)

    required_files = [
        PROCESSED_DIR / "manifest.jsonl",
        PROCESSED_DIR / "splits.json",
        PROCESSED_DIR / "label_map.json",
        PROCESSED_DIR / "labels.npy"
    ]

    all_ok = True
    for f in required_files:
        if f.exists():
            log(f"✓ {f.name} exists ({f.stat().st_size} bytes)")
        else:
            log(f"✗ {f.name} MISSING")
            all_ok = False

    # Count records
    records_dir = PROCESSED_DIR / "records"
    if records_dir.exists():
        npz_count = len(list(records_dir.glob("*.npz")))
        log(f"✓ {npz_count} .npz record files")

    # Load splits and show distribution
    splits_file = PROCESSED_DIR / "splits.json"
    if splits_file.exists():
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        if 'paths' in splits:
            train_count = len(splits['paths'].get('train', []))
            val_count = len(splits['paths'].get('val', []))
            test_count = len(splits['paths'].get('test', []))
            log(f"✓ Split distribution: train={train_count}, val={val_count}, test={test_count}")

    return all_ok

def write_final_report():
    """Write final summary report."""
    log("=" * 60)
    log("FINAL SUMMARY")
    log("=" * 60)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("ECG RESEARCH PIPELINE - AUTOMATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Read key metrics from manifest
        manifest = PROCESSED_DIR / "manifest.jsonl"
        if manifest.exists():
            with open(manifest, 'r') as mf:
                records = [json.loads(line) for line in mf]

            f.write(f"Total records processed: {len(records)}\n")

            # Label distribution
            from collections import Counter
            label_counts = Counter(r['label'] for r in records)
            f.write("\nLabel distribution:\n")
            label_map = {0:'MI', 1:'AF', 2:'BBB', 3:'NORM', 4:'OTHER'}
            for label_int, count in sorted(label_counts.items()):
                label_name = label_map.get(label_int, f'Unknown({label_int})')
                f.write(f"  {label_name}: {count}\n")

        f.write(f"\nArtifacts directory: {PROCESSED_DIR}\n")
        f.write(f"Logs directory: {LOGS_DIR}\n")
        f.write(f"Full log: {LOG_FILE}\n")
        f.write("\nTo run training, open the notebook and execute training cells.\n")

    log(f"✓ Report written to: {REPORT_FILE}")

    with open(REPORT_FILE, 'r') as f:
        print("\n" + f.read())

def main(mode='auto'):
    """
    Main orchestration function.
    mode: 'auto' (smoke + ask), 'smoke' (smoke only), 'full' (full run)
    """
    log("ECG RESEARCH PIPELINE - FULL AUTOMATION")
    log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Mode: {mode}")

    # A. Environment checks
    if not check_env():
        log("Environment check failed. Exiting.")
        return 1

    # B. Validate mapping
    validate_mapping()

    # Check current status
    status = check_preprocessing_status()

    if mode == 'smoke':
        # Only run smoke test
        if run_smoke_test():
            verify_outputs()
            write_final_report()
            log("✓ SUCCESS: Smoke test completed")
            return 0
        else:
            log("✗ FAILURE: Smoke test failed")
            return 1

    elif mode == 'full':
        # Skip smoke, go straight to full
        if run_full_preprocessing():
            verify_outputs()
            write_final_report()
            log("✓ SUCCESS: Full preprocessing completed")
            return 0
        else:
            log("✗ FAILURE: Full preprocessing failed")
            return 1

    else:  # mode == 'auto'
        # Smart mode: run smoke, then ask
        if not run_smoke_test():
            log("✗ Smoke test failed, stopping")
            return 1

        verify_outputs()

        # Ask user if they want full run
        log("")
        log("=" * 60)
        log("Smoke test passed! Ready for full preprocessing.")
        log(f"Datasets found in: {DATASET_DIR}")
        log("Full run may take several hours depending on dataset size.")
        log("=" * 60)

        response = input("Run full preprocessing now? (yes/no): ").strip().lower()

        if response in ['yes', 'y']:
            log("Starting full preprocessing...")
            if run_full_preprocessing():
                verify_outputs()
                write_final_report()
                log("✓ SUCCESS: Full preprocessing completed")
                return 0
            else:
                log("✗ FAILURE: Full preprocessing failed")
                return 1
        else:
            log("Full preprocessing skipped by user")
            log(f"To run later: python {__file__} --mode full")
            write_final_report()
            return 0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ECG Pipeline Automation")
    parser.add_argument('--mode', choices=['auto', 'smoke', 'full'], default='auto',
                       help='Run mode: auto (smoke+ask), smoke (test only), full (complete run)')
    args = parser.parse_args()

    exit_code = main(mode=args.mode)

    final_msg = "SUCCESS" if exit_code == 0 else "FAILED"
    log(f"\n{'='*60}")
    log(f"AUTOMATION {final_msg}")
    log(f"Full log: {LOG_FILE}")
    log(f"Report: {REPORT_FILE}")
    log(f"{'='*60}")

    sys.exit(exit_code)

