# Unified Label Mapping Generator

## Overview

The `generate_unified_mapping.py` script scans ECG datasets and creates a unified label mapping CSV file that maps diverse dataset-specific labels to a common set of target labels: **MI, AF, BBB, NORM, OTHER**.

## Usage

```powershell
cd D:\ecg-research
python scripts\generate_unified_mapping.py
```

## Output

The script generates: `logs/unified_label_mapping.candidate.csv`

### CSV Structure

| Column | Description |
|--------|-------------|
| `dataset` | Dataset name (e.g., ptb-xl, CinC2017) |
| `record_id` | Unique record identifier within dataset |
| `original_label_text` | Original label text from dataset |
| `mapped_label` | Mapped target label (MI/AF/BBB/NORM or empty) |

Records with empty `mapped_label` will be assigned to **OTHER** during preprocessing.

## Supported Datasets

### 1. PTB-XL
- **Source**: `Dataset/ptb-xl/ptbxl_database.csv`
- **Extracts**: filename_lr, scp_codes, report
- **Record ID format**: `00000/00001_lr`

### 2. CinC2017
- **Source**: `Dataset/CinC2017/training/REFERENCE*.csv` and similar
- **Labels**: 
  - N → NORM (Normal)
  - A → AF (Atrial Fibrillation)
  - O → (unmapped, will become OTHER)
  - ~ → (noisy, will become OTHER)
- **Record ID format**: `A00/A00003`

### 3. PTB Diagnostic
- **Source**: WFDB records listed in `Dataset/PTB_Diagnostic/RECORDS`
- **Extracts**: Diagnostic info from .hea file comments when available
- **Record ID format**: `patient001/s0016lre`

### 4. Chapman-Shaoxing
- **Source**: WFDB records in `Dataset/Chapman_Shaoxing/WFDBRecords/`
- **Extracts**: Diagnostic info from .hea file comments when available
- **Record ID format**: `WFDBRecords/01/012/JS00001`

## Label Mapping Heuristics

The script uses keyword pattern matching to automatically map labels:

### MI (Myocardial Infarction)
- Patterns: MI, IMI, AMI, STEMI, NSTEMI, INFARCT, Q WAVE, etc.

### AF (Atrial Fibrillation)
- Patterns: AF, AFIB, ATRIAL FIB, AFLT (flutter), etc.

### BBB (Bundle Branch Block)
- Patterns: BBB, LBBB, RBBB, IRBBB (incomplete), LAFB, LPFB (fascicular blocks), IVCD, etc.

### NORM (Normal)
- Patterns: NORM, NORMAL ECG, SINUS RHYTHM NORMAL
- **Safety check**: Will NOT map to NORM if pathological keywords are also present

### OTHER
- All records with empty `mapped_label` are treated as OTHER during preprocessing

## Workflow

1. **Run the script**:
   ```powershell
   python scripts\generate_unified_mapping.py
   ```

2. **Review the output**:
   - Check `logs/unified_label_mapping.candidate.csv`
   - Verify the summary statistics printed to console
   - Review unmapped records if needed

3. **Refine mappings (optional)**:
   - Manually edit the CSV to fix any misclassifications
   - Add custom mappings for domain-specific labels

4. **Finalize**:
   - Copy or rename to `logs/unified_label_mapping.csv`
   - This file will be used by the preprocessing pipeline

## Example Output

```
============================================================
SUMMARY STATISTICS
============================================================

Records per dataset:
  Chapman_Shaoxing         : 45,152
  CinC2017                 : 17,056
  PTB_Diagnostic           :    549
  ptb-xl                   : 21,799

Records per mapped label:
  MI        :  3,941
  AF        :  2,771
  BBB       :  2,580
  NORM      : 19,286
  OTHER     :      0
  (unmapped): 55,978

Unmapped: 66.2%
(Unmapped records will be assigned to OTHER during preprocessing)
```

## Notes

- The script is idempotent - you can run it multiple times safely
- Deduplication is performed based on `(dataset, record_id)` tuples
- Missing datasets are gracefully skipped with warning messages
- The heuristic mapping is conservative - when in doubt, it leaves `mapped_label` empty
- You can extend the heuristic patterns by editing the `map_label_heuristic()` function

## Troubleshooting

### No records found
- Verify dataset paths are correct
- Check that CSV/RECORDS files exist in expected locations

### Low mapping rate
- Review unmapped records and consider enhancing heuristic patterns
- Manually map domain-specific labels in the output CSV

### Unexpected mappings
- Review the heuristic patterns in `map_label_heuristic()`
- Add safety checks or exclusion patterns as needed
- Manually correct entries in the output CSV

