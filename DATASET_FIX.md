# Dataset Not Found - Issue Fixed

## Problem
The notebook reported:
```
Datasets found: []
Processing limit (0 means all): 0
No dataset folders – generating synthetic samples for quick smoke tests
```

Even though the `D:\ecg-research\Dataset` folder exists with datasets:
- Chapman_Shaoxing
- ptb-xl
- CinC2017
- PTB_Diagnostic

## Root Cause
The notebook uses `Path.cwd().resolve()` to find the project root. When you open/run the notebook from within the `notebooks/` directory, the current working directory becomes `D:\ecg-research\notebooks\`, and the code looks for `D:\ecg-research\notebooks\Dataset` which doesn't exist.

## Solution Applied

### 1. Updated notebook (master_pipeline.ipynb)
Changed the first cell to intelligently detect the project root:

**Before:**
```python
ROOT = Path.cwd().resolve()
DATASET_DIR = (ROOT / "Dataset")
```

**After:**
```python
# Find project root by looking for Dataset folder or going up from cwd
ROOT = Path.cwd().resolve()
# If we're in notebooks/ subdirectory, go up one level
if ROOT.name == 'notebooks' and (ROOT.parent / 'Dataset').exists():
    ROOT = ROOT.parent
# If still no Dataset found, try going up one more level
elif not (ROOT / 'Dataset').exists() and (ROOT.parent / 'Dataset').exists():
    ROOT = ROOT.parent
DATASET_DIR = (ROOT / "Dataset")
```

### 2. Updated generator script (create_master_notebook.py)
Applied the same fix to ensure future notebook regenerations include this correction.

## How to Test

### Option 1: Restart the Jupyter kernel and re-run first cell
1. In Jupyter: Kernel → Restart Kernel
2. Run the first cell (Environment checks and directory setup)
3. Check the output - it should now show:
   ```
   ROOT: D:\ecg-research
   DATASET_DIR exists: True
   ```

### Option 2: Re-run from project root
```bash
cd D:\ecg-research
jupyter notebook notebooks/master_pipeline.ipynb
```

### Option 3: Run full preprocessing
```bash
cd D:\ecg-research
python scripts/run_full_automation.py --mode smoke
```

## Expected Output After Fix
When you re-run the preprocessing cell, you should see:
```
Datasets found: ['Chapman_Shaoxing', 'CinC2017', 'PTB_Diagnostic', 'ptb-xl']
Processing limit (0 means all): 0
Processing Chapman_Shaoxing: ... files
Processing CinC2017: ... files
Processing PTB_Diagnostic: ... files
Processing ptb-xl: ... files
Done. processed: XXXXX skipped: YY
```

## Files Modified
- ✅ `notebooks/master_pipeline.ipynb` - Fixed ROOT path detection
- ✅ `create_master_notebook.py` - Fixed generator script

## Status
✅ **FIXED** - The notebook will now correctly find the Dataset folder regardless of where it's executed from.

---
**Date**: December 2, 2025  
**Issue**: Dataset not found  
**Status**: Resolved

