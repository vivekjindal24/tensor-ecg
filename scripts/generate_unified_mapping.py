"""
Generate unified label mapping for ECG datasets.

This script scans dataset/ folder for known ECG datasets and creates a unified
label mapping CSV with columns: dataset, record_id, original_label_text, mapped_label.

Supported datasets:
- ptb-xl: Parses ptbxl_database.csv
- CinC2017: Parses REFERENCE*.csv files
- PTB_Diagnostic: Lists WFDB records
- Chapman_Shaoxing: Lists WFDB records

Output: logs/unified_label_mapping.candidate.csv
"""

import re
import csv
import sys
from pathlib import Path
from collections import Counter
from typing import List, Dict

# Define project root - adjust if running from different locations
ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "Dataset"
LOGS_DIR = ROOT / "logs"
OUTPUT_CSV = LOGS_DIR / "unified_label_mapping.candidate.csv"

# Target label categories (consistent with pipeline)
TARGET_LABELS = ["MI", "AF", "BBB", "NORM", "OTHER"]


def map_label_heuristic(original_text: str) -> str:
    """
    Apply simple heuristic mapping based on keyword patterns.
    Returns one of: MI, AF, BBB, NORM, OTHER, or empty string if ambiguous.

    Args:
        original_text: Original label text from dataset

    Returns:
        Mapped label or empty string if uncertain
    """
    if not original_text:
        return ""

    text_upper = original_text.upper()

    # Myocardial Infarction patterns
    mi_patterns = [
        r'\bMI\b', r'MYOCARDIAL\s+INFARC', r'\bIMI\b', r'\bAMI\b',
        r'\bSTEMI\b', r'\bNSTEMI\b', r'\bPMI\b', r'\bLMI\b',
        r'INFARCT', r'Q\s*WAVE', r'PATHOLOGICAL\s+Q'
    ]
    if any(re.search(pat, text_upper) for pat in mi_patterns):
        return "MI"

    # Atrial Fibrillation patterns (including flutter)
    af_patterns = [
        r'\bAF\b', r'ATRIAL\s+FIB', r'A[\s-]?FIB', r'\bAFIB\b',
        r'AFIB', r'A\.FIB', r'\bAFLT\b', r'ATRIAL\s+FLUTTER',
        r'A[\s-]?FLUTTER'
    ]
    if any(re.search(pat, text_upper) for pat in af_patterns):
        return "AF"

    # Bundle Branch Block patterns (including fascicular blocks and incomplete blocks)
    bbb_patterns = [
        r'\bBBB\b', r'BUNDLE\s+BRANCH\s+BLOCK',
        r'\bLBBB\b', r'LEFT\s+BUNDLE\s+BRANCH',
        r'\bRBBB\b', r'RIGHT\s+BUNDLE\s+BRANCH',
        r'\bIRBBB\b', r'INCOMPLETE.*BUNDLE',
        r'\bILBBB\b',
        r'IVCD', r'INTRAVENTRICULAR\s+CONDUCTION',
        r'\bLAFB\b', r'\bLPFB\b', r'FASCICULAR\s+BLOCK',
        r'LEFT\s+ANTERIOR\s+FASCICULAR', r'LEFT\s+POSTERIOR\s+FASCICULAR'
    ]
    if any(re.search(pat, text_upper) for pat in bbb_patterns):
        return "BBB"

    # Normal patterns (be careful - only if explicitly stated)
    norm_patterns = [
        r'\bNORM\b', r'NORMAL\s+ECG', r'SINUSRHYTHMUS\s+NORMALES\s+EKG',
        r'^N$', r'^NORMAL$', r'NO\s+ABNORMAL', r'SINUS\s+RHYTHM.*NORMAL'
    ]
    if any(re.search(pat, text_upper) for pat in norm_patterns):
        # Additional safety: if text also contains pathological terms, don't map to NORM
        pathological_terms = ['INFARCT', 'ISCHEMI', 'HYPERTROPHY', 'BLOCK',
                             'FIBRILLATION', 'FLUTTER', 'TACHYCARDIA', 'BRADYCARDIA']
        if not any(term in text_upper for term in pathological_terms):
            return "NORM"

    # If we can't confidently map, return empty (will be OTHER in preprocessing)
    return ""


def parse_ptbxl(dataset_path: Path) -> List[Dict[str, str]]:
    """
    Parse PTB-XL dataset from ptbxl_database.csv.

    Expected columns: filename_lr (or similar), report, scp_codes

    Returns:
        List of dicts with keys: dataset, record_id, original_label_text, mapped_label
    """
    csv_path = dataset_path / "ptbxl_database.csv"
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found. Skipping PTB-XL.")
        return []

    records = []
    try:
        with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Try multiple possible column names
                filename_lr = row.get('filename_lr') or row.get('filename') or row.get('ecg_id')
                if not filename_lr:
                    continue

                # Clean up filename - remove 'records100/' or 'records500/' prefix
                record_id = filename_lr.replace('records100/', '').replace('records500/', '')
                record_id = record_id.strip()

                # Get label information (prefer scp_codes, fall back to report)
                scp_codes = row.get('scp_codes', '')
                report = row.get('report', '')
                original_label = scp_codes if scp_codes else report

                # Map label
                mapped = map_label_heuristic(original_label)

                records.append({
                    'dataset': 'ptb-xl',
                    'record_id': record_id,
                    'original_label_text': original_label.strip() if original_label else '',
                    'mapped_label': mapped
                })

        print(f"✓ PTB-XL: Parsed {len(records)} records from {csv_path.name}")
    except Exception as e:
        print(f"Error parsing PTB-XL: {e}")

    return records


def parse_cinc2017(dataset_path: Path) -> List[Dict[str, str]]:
    """
    Parse CinC2017 dataset from REFERENCE*.csv files in training/validation subdirs.

    CinC2017 labels: N (normal), A (AF), O (other), ~ (noisy)

    Returns:
        List of dicts with keys: dataset, record_id, original_label_text, mapped_label
    """
    records = []

    # CinC2017 label mapping
    cinc_label_map = {
        'N': 'NORM',
        'A': 'AF',
        'O': '',  # Other - leave empty for heuristic or default to OTHER
        '~': ''   # Noisy - leave empty
    }

    # Look in training, validation, and test subdirectories
    subdirs = ['training', 'validation', 'test']
    reference_files = []

    for subdir in subdirs:
        subdir_path = dataset_path / subdir
        if subdir_path.exists():
            # Find REFERENCE*.csv files (prefer REFERENCE.csv, then latest version)
            ref_files = sorted(subdir_path.glob("REFERENCE*.csv"))
            if ref_files:
                # Use the first one (typically REFERENCE.csv or highest version)
                reference_files.append((subdir, ref_files[0]))

    # Also check root directory for REFERENCE files
    root_ref_files = sorted(dataset_path.glob("REFERENCE*.csv"))
    if root_ref_files:
        reference_files.append(('root', root_ref_files[-1]))  # Use latest version

    if not reference_files:
        print(f"Warning: No REFERENCE*.csv files found in {dataset_path}. Skipping CinC2017.")
        return []

    seen_ids = set()
    for subdir, ref_file in reference_files:
        try:
            with open(ref_file, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Format: record_id,label (e.g., "A00/A00003,N")
                    parts = line.split(',')
                    if len(parts) != 2:
                        continue

                    record_id_raw, label = parts[0].strip(), parts[1].strip()

                    # Normalize record_id: use forward slashes
                    record_id = record_id_raw.replace('\\', '/')

                    # Skip duplicates across files
                    if record_id in seen_ids:
                        continue
                    seen_ids.add(record_id)

                    # Map label
                    mapped = cinc_label_map.get(label, '')

                    records.append({
                        'dataset': 'CinC2017',
                        'record_id': record_id,
                        'original_label_text': label,
                        'mapped_label': mapped
                    })
        except Exception as e:
            print(f"Error reading {ref_file.name}: {e}")

    print(f"✓ CinC2017: Parsed {len(records)} records from {len(reference_files)} reference file(s)")
    return records


def parse_wfdb_fallback(dataset_path: Path, dataset_name: str) -> List[Dict[str, str]]:
    """
    Fallback parser for WFDB-format datasets (PTB_Diagnostic, Chapman_Shaoxing).
    Lists all .hea files and uses relative path as record_id.

    For these datasets, we don't have direct label info, so original_label_text
    will be empty and mapped_label will be empty (will become OTHER in preprocessing).

    Returns:
        List of dicts with keys: dataset, record_id, original_label_text, mapped_label
    """
    records = []

    # Check for RECORDS file first
    records_file = dataset_path / "RECORDS"
    if records_file.exists():
        try:
            with open(records_file, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    # RECORDS file lists paths relative to dataset root
                    record_id = line.replace('\\', '/')

                    # For Chapman, RECORDS lists directories; we need to find .hea files
                    if dataset_name == 'Chapman_Shaoxing':
                        record_path = dataset_path / line
                        if record_path.is_dir():
                            # Find .hea files in this directory
                            hea_files = list(record_path.glob("*.hea"))
                            for hea in hea_files:
                                rel_path = hea.relative_to(dataset_path).as_posix()
                                # Remove .hea extension for record_id
                                rel_id = rel_path[:-4] if rel_path.endswith('.hea') else rel_path

                                # Try to infer label from header if available
                                original_label = extract_label_from_wfdb_header(hea)
                                mapped = map_label_heuristic(original_label)

                                records.append({
                                    'dataset': dataset_name,
                                    'record_id': rel_id,
                                    'original_label_text': original_label,
                                    'mapped_label': mapped
                                })
                        else:
                            # Single record entry
                            original_label = ""
                            hea_path = dataset_path / f"{line}.hea"
                            if hea_path.exists():
                                original_label = extract_label_from_wfdb_header(hea_path)

                            mapped = map_label_heuristic(original_label)
                            records.append({
                                'dataset': dataset_name,
                                'record_id': record_id,
                                'original_label_text': original_label,
                                'mapped_label': mapped
                            })
                    else:
                        # PTB_Diagnostic: RECORDS lists actual record paths
                        original_label = ""
                        hea_path = dataset_path / f"{line}.hea"
                        if hea_path.exists():
                            original_label = extract_label_from_wfdb_header(hea_path)

                        mapped = map_label_heuristic(original_label)
                        records.append({
                            'dataset': dataset_name,
                            'record_id': record_id,
                            'original_label_text': original_label,
                            'mapped_label': mapped
                        })
        except Exception as e:
            print(f"Error reading RECORDS file for {dataset_name}: {e}")

    # If no RECORDS file or it failed, fall back to globbing
    if not records:
        print(f"Scanning {dataset_name} for .hea files (no RECORDS file)...")
        hea_files = list(dataset_path.rglob("*.hea"))
        for hea in hea_files:
            try:
                rel_path = hea.relative_to(dataset_path).as_posix()
                # Remove .hea extension
                rel_id = rel_path[:-4] if rel_path.endswith('.hea') else rel_path

                original_label = extract_label_from_wfdb_header(hea)
                mapped = map_label_heuristic(original_label)

                records.append({
                    'dataset': dataset_name,
                    'record_id': rel_id,
                    'original_label_text': original_label,
                    'mapped_label': mapped
                })
            except Exception as e:
                print(f"Error processing {hea.name}: {e}")

    print(f"✓ {dataset_name}: Found {len(records)} records")
    return records


def extract_label_from_wfdb_header(hea_path: Path) -> str:
    """
    Attempt to extract diagnostic label from WFDB .hea file comments.
    Many WFDB headers contain diagnostic info in comment lines.

    Returns:
        Extracted label text or empty string
    """
    try:
        with open(hea_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        # Look for comment lines (start with #)
        comments = []
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                # Remove leading # and whitespace
                comment = line.lstrip('#').strip()
                comments.append(comment)

        # Combine comments
        full_comment = ' '.join(comments)

        # Look for common diagnostic keywords
        if any(kw in full_comment.upper() for kw in ['MYOCARDIAL', 'INFARCT', 'MI', 'HEALTHY',
                                                       'NORMAL', 'BUNDLE', 'FIBRILLATION']):
            return full_comment

        return ""
    except Exception:
        return ""


def deduplicate_records(records: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Remove duplicate entries based on (dataset, record_id) tuple.
    Keep the first occurrence.

    Args:
        records: List of record dictionaries

    Returns:
        Deduplicated list
    """
    seen = set()
    unique = []

    for rec in records:
        key = (rec['dataset'], rec['record_id'])
        if key not in seen:
            seen.add(key)
            unique.append(rec)

    duplicates = len(records) - len(unique)
    if duplicates > 0:
        print(f"Removed {duplicates} duplicate entries")

    return unique


def write_unified_csv(records: List[Dict[str, str]], output_path: Path):
    """
    Write unified label mapping to CSV file.

    Args:
        records: List of record dictionaries
        output_path: Output CSV file path
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV
    fieldnames = ['dataset', 'record_id', 'original_label_text', 'mapped_label']

    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        print(f"\n✓ Wrote {len(records)} records to {output_path}")
    except Exception as e:
        print(f"Error writing CSV: {e}")
        sys.exit(1)


def print_summary(records: List[Dict[str, str]]):
    """
    Print summary statistics of the generated mapping.

    Args:
        records: List of record dictionaries
    """
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    # Count by dataset
    by_dataset = Counter(r['dataset'] for r in records)
    print("\nRecords per dataset:")
    for ds, count in sorted(by_dataset.items()):
        print(f"  {ds:25s}: {count:6,d}")

    # Count by mapped label
    by_label = Counter(r['mapped_label'] for r in records)
    print("\nRecords per mapped label:")
    for label in TARGET_LABELS:
        count = by_label.get(label, 0)
        print(f"  {label:10s}: {count:6,d}")

    unmapped_count = by_label.get('', 0)
    print(f"  {'(unmapped)':10s}: {unmapped_count:6,d}")

    # Unmapped percentage
    if records:
        unmapped_pct = (unmapped_count / len(records)) * 100
        print(f"\nUnmapped: {unmapped_pct:.1f}%")
        print("(Unmapped records will be assigned to OTHER during preprocessing)")

    print("\n" + "="*60)


def main():
    """Main execution function."""
    print("="*60)
    print("ECG Unified Label Mapping Generator")
    print("="*60)
    print(f"Project root: {ROOT}")
    print(f"Dataset directory: {DATASET_DIR}")
    print(f"Output: {OUTPUT_CSV}")
    print()

    # Check if dataset directory exists
    if not DATASET_DIR.exists():
        print(f"Error: Dataset directory not found: {DATASET_DIR}")
        sys.exit(1)

    all_records = []

    # Parse PTB-XL
    ptbxl_path = DATASET_DIR / "ptb-xl"
    if ptbxl_path.exists():
        print(f"\nProcessing PTB-XL at {ptbxl_path}...")
        all_records.extend(parse_ptbxl(ptbxl_path))
    else:
        print(f"\nSkipping PTB-XL (not found at {ptbxl_path})")

    # Parse CinC2017
    cinc_path = DATASET_DIR / "CinC2017"
    if cinc_path.exists():
        print(f"\nProcessing CinC2017 at {cinc_path}...")
        all_records.extend(parse_cinc2017(cinc_path))
    else:
        print(f"\nSkipping CinC2017 (not found at {cinc_path})")

    # Parse PTB_Diagnostic
    ptb_diag_path = DATASET_DIR / "PTB_Diagnostic"
    if ptb_diag_path.exists():
        print(f"\nProcessing PTB_Diagnostic at {ptb_diag_path}...")
        all_records.extend(parse_wfdb_fallback(ptb_diag_path, "PTB_Diagnostic"))
    else:
        print(f"\nSkipping PTB_Diagnostic (not found at {ptb_diag_path})")

    # Parse Chapman_Shaoxing
    chapman_path = DATASET_DIR / "Chapman_Shaoxing"
    if chapman_path.exists():
        print(f"\nProcessing Chapman_Shaoxing at {chapman_path}...")
        all_records.extend(parse_wfdb_fallback(chapman_path, "Chapman_Shaoxing"))
    else:
        print(f"\nSkipping Chapman_Shaoxing (not found at {chapman_path})")

    # Check if we found any records
    if not all_records:
        print("\nError: No records found in any dataset!")
        sys.exit(1)

    print(f"\nTotal records collected: {len(all_records)}")

    # Deduplicate
    print("\nDeduplicating records...")
    unique_records = deduplicate_records(all_records)

    # Write output
    print(f"\nWriting output to {OUTPUT_CSV}...")
    write_unified_csv(unique_records, OUTPUT_CSV)

    # Print summary
    print_summary(unique_records)

    print("\n✓ Done!")
    print(f"\nNext steps:")
    print(f"1. Review the generated file: {OUTPUT_CSV}")
    print(f"2. Manually refine any ambiguous mappings if needed")
    print(f"3. Copy/rename to logs/unified_label_mapping.csv when ready")


if __name__ == "__main__":
    main()

