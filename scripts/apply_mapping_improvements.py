"""
Apply mapping improvements to create an updated unified_label_mapping.csv
"""
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
LOGS_DIR = ROOT / "logs"

# Load original mapping
mapping_file = LOGS_DIR / "unified_label_mapping.csv"
improvements_file = LOGS_DIR / "mapping_improvements_suggested.csv"

if not improvements_file.exists():
    print("Error: No improvements file found. Run improve_mapping.py first.")
    exit(1)

# Backup original
backup_file = LOGS_DIR / f"unified_label_mapping.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
print(f"Creating backup: {backup_file.name}")

df = pd.read_csv(mapping_file, dtype=str).fillna("")
df.to_csv(backup_file, index=False)

# Load improvements
improvements_df = pd.read_csv(improvements_file, dtype=str)
print(f"\nLoaded {len(improvements_df)} improvements")

# Apply improvements
applied = 0
for _, imp in improvements_df.iterrows():
    idx = int(imp['index'])
    suggested_label = imp['suggested_label']

    if idx < len(df):
        df.at[idx, 'mapped_label'] = suggested_label
        applied += 1

print(f"Applied {applied} improvements")

# Save updated mapping
df.to_csv(mapping_file, index=False)
print(f"\n✓ Updated mapping saved to: {mapping_file}")

# Print new distribution
print(f"\nUpdated label distribution:")
label_counts = df['mapped_label'].value_counts()
for label, count in label_counts.items():
    pct = count / len(df) * 100
    label_name = label if label else "(unmapped)"
    print(f"  {label_name}: {count:,} ({pct:.1f}%)")

unmapped = (df['mapped_label'] == '').sum()
print(f"\nRemaining unmapped: {unmapped:,} ({unmapped/len(df)*100:.1f}%)")

print("\n✓ Mapping update complete!")
print("\nYou can now run preprocessing with the improved mapping.")

