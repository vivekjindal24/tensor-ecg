from pathlib import Path
import os

# Simulate running from notebooks directory
os.chdir('D:/ecg-research/notebooks')

ROOT = Path.cwd().resolve()
print('Initial ROOT:', ROOT)
print('ROOT.name:', ROOT.name)

# Check if we're in notebooks subdirectory
if ROOT.name == 'notebooks' and (ROOT.parent / 'Dataset').exists():
    ROOT = ROOT.parent
    print('Adjusted ROOT (from notebooks):', ROOT)
elif not (ROOT / 'Dataset').exists() and (ROOT.parent / 'Dataset').exists():
    ROOT = ROOT.parent
    print('Adjusted ROOT (Dataset in parent):', ROOT)

print('Final ROOT:', ROOT)
print('Dataset path:', ROOT / 'Dataset')
print('Dataset exists:', (ROOT / 'Dataset').exists())

if (ROOT / 'Dataset').exists():
    datasets = list((ROOT / 'Dataset').iterdir())
    print('Datasets found:', [d.name for d in datasets if d.is_dir()])

