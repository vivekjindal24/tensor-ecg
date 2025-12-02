"""
Quick verification that the notebook will now find the Dataset folder.
Run this to confirm the fix works.
"""
import os
import sys
from pathlib import Path

print("="*60)
print("DATASET DETECTION VERIFICATION")
print("="*60)

# Test 1: From project root
print("\n1. Testing from project root (D:\\ecg-research):")
os.chdir('D:/ecg-research')
ROOT = Path.cwd().resolve()
if ROOT.name == 'notebooks' and (ROOT.parent / 'Dataset').exists():
    ROOT = ROOT.parent
elif not (ROOT / 'Dataset').exists() and (ROOT.parent / 'Dataset').exists():
    ROOT = ROOT.parent
print(f"   ROOT: {ROOT}")
print(f"   Dataset exists: {(ROOT / 'Dataset').exists()}")
if (ROOT / 'Dataset').exists():
    datasets = [d.name for d in (ROOT / 'Dataset').iterdir() if d.is_dir()]
    print(f"   Datasets found: {datasets}")

# Test 2: From notebooks directory
print("\n2. Testing from notebooks directory (D:\\ecg-research\\notebooks):")
os.chdir('D:/ecg-research/notebooks')
ROOT = Path.cwd().resolve()
print(f"   Initial ROOT: {ROOT}")
if ROOT.name == 'notebooks' and (ROOT.parent / 'Dataset').exists():
    ROOT = ROOT.parent
    print(f"   Adjusted ROOT: {ROOT}")
elif not (ROOT / 'Dataset').exists() and (ROOT.parent / 'Dataset').exists():
    ROOT = ROOT.parent
    print(f"   Adjusted ROOT: {ROOT}")
print(f"   Dataset exists: {(ROOT / 'Dataset').exists()}")
if (ROOT / 'Dataset').exists():
    datasets = [d.name for d in (ROOT / 'Dataset').iterdir() if d.is_dir()]
    print(f"   Datasets found: {datasets}")

print("\n" + "="*60)
if (ROOT / 'Dataset').exists() and datasets:
    print("✓ SUCCESS: Dataset folder is detected correctly!")
    print(f"✓ Found {len(datasets)} dataset(s)")
else:
    print("✗ FAILED: Dataset folder not found")
print("="*60)

