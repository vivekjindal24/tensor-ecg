"""
Run smoke test on the master pipeline notebook.
Executes preprocessing with a small limit to verify the pipeline works.
"""
import os
import sys
import subprocess
from pathlib import Path

# Set environment variables for smoke test
os.environ['ECG_PREPROCESS_LIMIT'] = '100'  # Process only 100 records
os.environ['ECG_EPOCHS'] = '1'
os.environ['ECG_BATCH_SIZE'] = '4'

ROOT = Path(__file__).parent.parent
NB_PATH = ROOT / "notebooks" / "master_pipeline.ipynb"
OUT_PATH = ROOT / "logs" / "smoke_test_run.ipynb"

print(f"Running smoke test on: {NB_PATH}")
print(f"Output will be saved to: {OUT_PATH}")
print(f"Limit: {os.environ['ECG_PREPROCESS_LIMIT']} records")
print("-" * 60)

# Ensure output directory exists
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Run nbconvert
cmd = [
    sys.executable, "-m", "jupyter", "nbconvert",
    "--to", "notebook",
    "--execute",
    "--ExecutePreprocessor.timeout=600",  # 10 minutes for smoke test
    "--output", str(OUT_PATH.absolute()),
    str(NB_PATH.absolute())
]

try:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print("✓ Smoke test completed successfully!")
    print(f"✓ Output notebook: {OUT_PATH}")
    
    # Check if key artifacts were created
    artifacts_dir = ROOT / "artifacts" / "processed"
    manifest = artifacts_dir / "manifest.jsonl"
    splits = artifacts_dir / "splits.json"
    
    if manifest.exists():
        with open(manifest, 'r') as f:
            count = sum(1 for _ in f)
        print(f"✓ Manifest created with {count} records")
    
    if splits.exists():
        print(f"✓ Splits file created")
    
    sys.exit(0)
    
except subprocess.CalledProcessError as e:
    print("✗ Smoke test failed!")
    print("STDOUT:", e.stdout)
    print("STDERR:", e.stderr)
    sys.exit(1)
except Exception as e:
    print(f"✗ Error running smoke test: {e}")
    sys.exit(1)

