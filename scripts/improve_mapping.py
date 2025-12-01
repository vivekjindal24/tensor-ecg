"""
Improved label mapping using enhanced heuristics.
Analyzes unmapped records and attempts to assign labels based on pattern matching.
"""
import pandas as pd
import re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent
LOGS_DIR = ROOT / "logs"

# Load current mapping
mapping_file = LOGS_DIR / "unified_label_mapping.csv"
df = pd.read_csv(mapping_file, dtype=str).fillna("")

print("="*80)
print("LABEL MAPPING IMPROVEMENT")
print("="*80)

# Identify unmapped
unmapped = df[df['mapped_label'] == ''].copy()
print(f"\nUnmapped records: {len(unmapped):,}")

# Enhanced heuristics
def enhanced_label_heuristic(original_text: str, dataset: str) -> str:
    """
    Apply enhanced heuristics to guess labels.
    Returns label or empty string if uncertain.
    """
    if not original_text:
        return ""

    text_lower = str(original_text).lower()

    # MI patterns (more comprehensive)
    mi_patterns = [
        r'\bmi\b', r'myocardial\s*infarction', r'\bstemi\b', r'\bnstemi\b',
        r'anterior\s*infarct', r'inferior\s*infarct', r'lateral\s*infarct',
        r'old\s*infarct', r'recent\s*infarct', r'\bami\b', r'\bpmi\b',
        r'q\s*wave', r'pathologic.*q', r'\bimis?\b'
    ]

    # AF patterns
    af_patterns = [
        r'\baf\b', r'atrial\s*fib', r'a[\s-]*fib', r'\bafib\b', r'\baflu?\b',
        r'atrial\s*flutter', r'a[\s-]*flutter', r'\bpaf\b'
    ]

    # BBB patterns
    bbb_patterns = [
        r'\bbbb\b', r'bundle\s*branch\s*block', r'\blbbb\b', r'\brbbb\b',
        r'left\s*bundle', r'right\s*bundle', r'bifascicular',
        r'incomplete.*bundle', r'complete.*bundle'
    ]

    # NORM patterns
    norm_patterns = [
        r'\bnorm(al)?\b', r'sinus\s*rhythm', r'\bsr\b', r'regular\s*rhythm',
        r'no\s*abnormal', r'within\s*normal', r'unremarkable',
        r'^normal$', r'healthy', r'control'
    ]

    # Check patterns (order matters - specific before general)
    for pattern in mi_patterns:
        if re.search(pattern, text_lower):
            return "MI"

    for pattern in af_patterns:
        if re.search(pattern, text_lower):
            return "AF"

    for pattern in bbb_patterns:
        if re.search(pattern, text_lower):
            return "BBB"

    for pattern in norm_patterns:
        if re.search(pattern, text_lower):
            return "NORM"

    # Dataset-specific rules
    if dataset == "PTB_Diagnostic":
        # PTB often has "Healthy control" or specific condition names
        if "healthy" in text_lower or "control" in text_lower:
            return "NORM"
        if "myocardial infarction" in text_lower:
            return "MI"

    if dataset == "ptb-xl":
        # PTBXL uses SCP codes, check for common patterns
        if "norm" in text_lower:
            return "NORM"

    return ""

# Apply enhanced heuristics to unmapped records
print("\nApplying enhanced heuristics...")
improvements = []

for idx, row in unmapped.iterrows():
    original = str(row.get('original_label_text', '')).strip()
    dataset = str(row.get('dataset', '')).strip()

    new_label = enhanced_label_heuristic(original, dataset)

    if new_label:
        improvements.append({
            'index': idx,
            'dataset': dataset,
            'record_id': row.get('record_id', ''),
            'original_text': original,
            'suggested_label': new_label
        })

print(f"Found {len(improvements):,} potential improvements")

# Show distribution of suggested labels
if improvements:
    suggested_counts = Counter(imp['suggested_label'] for imp in improvements)
    print(f"\nSuggested label distribution:")
    for label, count in suggested_counts.most_common():
        pct = count / len(improvements) * 100
        print(f"  {label}: {count:,} ({pct:.1f}%)")

    # Save suggestions to file
    improvements_df = pd.DataFrame(improvements)
    output_file = LOGS_DIR / "mapping_improvements_suggested.csv"
    improvements_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved suggestions to: {output_file}")

    # Show samples from each label
    print(f"\nExample suggestions (5 per label):")
    for label in ['MI', 'AF', 'BBB', 'NORM']:
        label_samples = improvements_df[improvements_df['suggested_label'] == label].head(5)
        if not label_samples.empty:
            print(f"\n  {label}:")
            for _, row in label_samples.iterrows():
                print(f"    - [{row['dataset']}] {row['original_text'][:60]}")

    # Option to apply improvements
    print("\n" + "="*80)
    print("TO APPLY THESE IMPROVEMENTS:")
    print("="*80)
    print("Review the suggestions in: logs/mapping_improvements_suggested.csv")
    print("Then run: python scripts/apply_mapping_improvements.py")
    print("\nThis will create an updated unified_label_mapping.csv file.")
else:
    print("\nNo improvements found with current heuristics.")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

