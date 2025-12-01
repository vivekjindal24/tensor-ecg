"""Validate and summarize unified label mapping CSV"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
LOGS_DIR = ROOT / "logs"
mapping_file = LOGS_DIR / "unified_label_mapping.csv"

if not mapping_file.exists():
    print(f"Error: {mapping_file} not found")
    exit(1)

df = pd.read_csv(mapping_file, dtype=str).fillna('')

print("="*80)
print("UNIFIED LABEL MAPPING VALIDATION")
print("="*80)
print(f"Total rows: {len(df):,}")
print(f"\nColumns: {list(df.columns)}")

# Dataset distribution
print(f"\nDataset distribution:")
for dataset, count in df['dataset'].value_counts().items():
    print(f"  {dataset}: {count:,}")

# Label distribution
print(f"\nLabel distribution:")
label_counts = df['mapped_label'].value_counts()
for label, count in label_counts.items():
    pct = count / len(df) * 100
    print(f"  {label if label else '(unmapped)'}: {count:,} ({pct:.1f}%)")

# Count unmapped
unmapped = (df['mapped_label'] == '').sum()
print(f"\nUnmapped (blank) records: {unmapped:,} ({unmapped/len(df)*100:.1f}%)")

# If high unmapped rate, create sample file
if unmapped > 0:
    unmapped_df = df[df['mapped_label'] == ''].copy()
    sample_size = min(200, len(unmapped_df))
    sample = unmapped_df.sample(n=sample_size, random_state=42)
    sample_file = LOGS_DIR / "unmapped_sample.csv"
    sample.to_csv(sample_file, index=False)
    print(f"\nSaved {sample_size} unmapped samples to: {sample_file}")

print("\nValidation complete!")

