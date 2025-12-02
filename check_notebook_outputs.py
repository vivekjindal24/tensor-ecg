import json
from pathlib import Path

nb_path = Path("D:/ecg-research/notebooks/master_pipeline.ipynb")
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("=" * 60)
print("CHECKING NOTEBOOK CELL OUTPUTS")
print("=" * 60)

for i, cell in enumerate(nb['cells']):
    source = ''.join(cell.get('source', []))

    # Check first cell (Environment checks)
    if 'Environment checks and directory setup' in source:
        print(f"\nCell {i}: Environment Setup")
        print("-" * 60)
        outputs = cell.get('outputs', [])
        if outputs:
            for output in outputs:
                if output.get('output_type') == 'stream':
                    text = output.get('text', [])
                    if isinstance(text, list):
                        print(''.join(text))
                    else:
                        print(text)
        else:
            print("(No output - cell not executed yet)")

    # Check preprocessing cell
    if 'Datasets found:' in source:
        print(f"\nCell {i}: Preprocessing (Dataset Detection)")
        print("-" * 60)
        outputs = cell.get('outputs', [])
        if outputs:
            for output in outputs:
                if output.get('output_type') == 'stream':
                    text = output.get('text', [])
                    if isinstance(text, list):
                        text_str = ''.join(text)
                    else:
                        text_str = text
                    # Print first 500 chars
                    lines = text_str.split('\n')[:10]
                    print('\n'.join(lines))
        else:
            print("(No output - cell not executed yet)")

print("\n" + "=" * 60)

