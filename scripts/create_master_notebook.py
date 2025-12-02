#!/usr/bin/env python3
"""
scripts/create_master_notebook.py
Builds notebooks/master_pipeline.ipynb by inlining .py files and creating runnable cells.
"""
from pathlib import Path
import json
import textwrap

ROOT = Path.cwd()
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
NOTEBOOKS = ROOT / "notebooks"
TARGET = NOTEBOOKS / "master_pipeline.ipynb"
NOTEBOOKS.mkdir(parents=True, exist_ok=True)

def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def code_cell(source: str):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }

def md_cell(text: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": (text + "\n").splitlines(keepends=True),
    }

cells = []

# 0. Title + run-headless note
cells.append(md_cell(
    "# Master ECG Pipeline\n\n"
    "This notebook combines all project scripts and modules into one single runnable file.\n\n"
    "**Usage:** run cells top-to-bottom. For headless execution on Windows use:\n\n"
    "```python\n"
    "import asyncio, sys\n"
    "if sys.platform == 'win32':\n"
    "    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())\n"
    "```\n"
))

# 1. Environment & imports (idempotent)
env_code = textwrap.dedent("""
# Environment & imports - idempotent
import os, sys, json, time, math, asyncio
from pathlib import Path
import numpy as np
import random
import torch
print('Python:', sys.executable)
print('Torch:', getattr(torch, '__version__', 'n/a'))
# Windows asyncio fix for nbconvert headless runs
import platform
if platform.system() == 'Windows':
    try:
        import asyncio, sys
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# Project root detection
ROOT = Path(os.environ.get('ECG_ROOT', Path.cwd().resolve()))
DATASET_DIR = ROOT / 'Dataset'
ARTIFACTS_DIR = ROOT / 'artifacts'
PROCESSED_DIR = ARTIFACTS_DIR / 'processed'
LOGS_DIR = ROOT / 'logs'
NOTEBOOKS_DIR = ROOT / 'notebooks'
for p in (ARTIFACTS_DIR, PROCESSED_DIR, PROCESSED_DIR/'records', LOGS_DIR):
    p.mkdir(parents=True, exist_ok=True)

# seeds for reproducibility
SEED = int(os.environ.get('ECG_SEED', '42'))
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('DEVICE', DEVICE)
""")
cells.append(code_cell(env_code))

# 2. Inline src modules (each file -> markdown header + code cell)
def inline_dir(path: Path, title: str):
    py_files = sorted([p for p in path.glob('*.py') if p.is_file()])
    if not py_files:
        return
    cells.append(md_cell(f"## Inlined: {title}\n\nFiles: {', '.join(p.name for p in py_files)}"))
    for p in py_files:
        source = read_file(p)
        header = f"# --- {p.name} (inlined) ---\n"
        # wrap in a try/except to prevent name collisions stopping the notebook
        wrapped = (
            header +
            "try:\n" +
            "\n".join("    " + line for line in source.splitlines()) +
            "\nexcept Exception as _e:\n    print('Warning: inlined module', '{}', 'raised', _e)\n".format(p.name)
        )
        cells.append(code_cell(wrapped))

# Inline src and scripts
inline_dir(SRC, "Source modules (src/)")
inline_dir(SCRIPTS, "Utility scripts (scripts/)")

# 3. Operational cells: run mapping generation, optional improvement, preprocessing stub
cells.append(md_cell("## Quick: Generate/Load unified mapping (run this cell)"))
cells.append(code_cell(textwrap.dedent("""
# Generate unified mapping (if you have script)
candidate = Path('logs/unified_label_mapping.candidate.csv')
prod = Path('logs/unified_label_mapping.csv')
if (Path('scripts/generate_unified_mapping.py')).exists() and not candidate.exists():
    print('Generating candidate mapping...')
    os.system(f'python \"{str(Path("scripts/generate_unified_mapping.py"))}\"')
else:
    print('Candidate mapping exists:', candidate.exists(), 'Prod file exists:', prod.exists())
# If you have a candidate and want to promote it, uncomment:
# if candidate.exists(): candidate.replace(prod)
""")))

cells.append(md_cell("## Preprocessing (streaming, memory-safe)"))
cells.append(code_cell(textwrap.dedent("""
# Run streaming preprocessing from scripts/preprocess_streaming.py if present
proc_script = Path('scripts/preprocess_streaming.py')
if proc_script.exists():
    print('Launching streaming preprocessing (script)...')
    # recommend using environment var ECG_PREPROCESS_LIMIT to test
    os.system(f'python \"{proc_script}\"')
else:
    print('No preprocess_streaming.py found. Implement preprocessing in this notebook or inline alternate script.')
""")))

cells.append(md_cell("## Training (run this after preprocessing finishes)"))
cells.append(code_cell(textwrap.dedent("""
# Launch training script if present
train_script = Path('scripts/train_pipeline.py')  # optional
if train_script.exists():
    print('Running training script...')
    os.system(f'python \"{train_script}\"')
else:
    print('No training script detected. Use in-notebook training cells or create scripts/training.py and link it.')
""")))

cells.append(md_cell("## Evaluation and Visuals"))
cells.append(code_cell(textwrap.dedent("""
# Run evaluation if script exists
eval_script = Path('scripts/evaluate.py')
if eval_script.exists():
    os.system(f'python \"{eval_script}\"')
else:
    print('No evaluate.py. Use notebook cells to visualize artifacts/figures/')
""")))

cells.append(md_cell("## Smoke tests and quick validation"))
cells.append(code_cell(textwrap.dedent("""
# Run smoke tests
smoke = Path('scripts/verify_smoke_test.py')
if smoke.exists():
    os.system(f'python \"{smoke}\"')
else:
    print('No smoke-test script. Manual checks:')
    print(' - Count files:', len(list((PROCESSED_DIR/'records').glob('*.npz'))))
    print(' - Check splits:', (PROCESSED_DIR/'splits.json').exists())
""")))

cells.append(md_cell("## Final: Notebook control\nYou can now run cells in order. Long-running steps are executed as external scripts to avoid kernel timeouts."))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

TARGET.write_text(json.dumps(nb, indent=2), encoding="utf-8")
print("Wrote notebook to", TARGET)
