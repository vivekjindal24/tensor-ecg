# create_master_notebook.py
"""
Create a single self-contained notebook for the ECG pipeline.
Writes notebooks/master_pipeline.ipynb
"""
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path

NB_DIR = Path("notebooks")
NB_DIR.mkdir(parents=True, exist_ok=True)
TARGET = NB_DIR / "master_pipeline.ipynb"

cells = []

# --- Title + quick instructions ---
cells.append(new_markdown_cell(
"# ECG Master Pipeline\n\n"
"Single notebook with preprocessing, training, evaluation and smoke tests.\n\n"
"How to run:\n\n"
"1. Open interactively: `jupyter notebook notebooks/master_pipeline.ipynb`\n"
"2. Or run headless (recommended for full preprocessing):\n"
"   `jupyter nbconvert --to notebook --execute notebooks/master_pipeline.ipynb --output logs/preprocess_run.ipynb`\n\n"
"Notes:\n- If you are on Windows and see asyncio warnings, add at top of kernel: "
"`import asyncio, sys; asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())`.\n"
"\n---\n"
"Cells are grouped: Environment, Config, Utilities, Mapping Load, Preprocessing, Dataset, Model, Training, Evaluation, Smoke Tests, Orchestrator."
))

# --- Environment checks ---
cells.append(new_code_cell(
"""# Environment checks and directory setup
import os, sys, asyncio
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

from pathlib import Path
# Find project root by looking for Dataset folder or going up from cwd
ROOT = Path.cwd().resolve()
initial_root = ROOT
# If we're in notebooks/ subdirectory, go up one level
if ROOT.name == 'notebooks' and (ROOT.parent / 'Dataset').exists():
    ROOT = ROOT.parent
    print(f'[Adjusted] ROOT from {initial_root} -> {ROOT}')
# If still no Dataset found, try going up one more level
elif not (ROOT / 'Dataset').exists() and (ROOT.parent / 'Dataset').exists():
    ROOT = ROOT.parent
    print(f'[Adjusted] ROOT from {initial_root} -> {ROOT}')
DATASET_DIR = (ROOT / "Dataset")
ARTIFACTS_DIR = (ROOT / "artifacts")
PROCESSED_DIR = ARTIFACTS_DIR / "processed"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
LOGS_DIR = ROOT / "logs"
for p in [ARTIFACTS_DIR, PROCESSED_DIR, PROCESSED_DIR / "records", FIGURES_DIR, LOGS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

print('ROOT:', ROOT)
print('DATASET_DIR:', DATASET_DIR)
print('DATASET_DIR exists:', DATASET_DIR.exists())
if DATASET_DIR.exists():
    subdirs = [d.name for d in DATASET_DIR.iterdir() if d.is_dir()]
    print(f'Dataset subdirectories: {subdirs}')
print('ARTIFACTS_DIR:', ARTIFACTS_DIR)
print('PROCESSED_DIR:', PROCESSED_DIR)
"""
))

# --- Imports and config ---
cells.append(new_code_cell(
"""# Imports, device, seeds
import os, random, json, time, math
import numpy as np, pandas as pd
import torch
from collections import Counter, defaultdict
from pathlib import Path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device ->', DEVICE)

SEED = int(os.environ.get('ECG_SEED', 42))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed_all(SEED)
"""
))

cells.append(new_code_cell(
"""# Configuration constants
TARGET_FS = 500
TARGET_SAMPLES = 5000   # 10s @ 500Hz
LABEL_ORDER = ['MI','AF','BBB','NORM','OTHER']
LABEL_TO_INT = {l:i for i,l in enumerate(LABEL_ORDER)}
BATCH_SIZE = 8 if DEVICE.type=='cpu' else int(os.environ.get('ECG_BATCH_SIZE', 32))
EPOCHS = int(os.environ.get('ECG_EPOCHS', 2))
LR = float(os.environ.get('ECG_LR', 1e-3))
USE_AMP = torch.cuda.is_available()
print('BATCH_SIZE', BATCH_SIZE, 'EPOCHS', EPOCHS, 'AMP', USE_AMP)
"""
))

# --- Utility functions ---
cells.append(new_code_cell(
"""# Utilities: IO, normalization, resample, safe save/load
import json, gzip
import numpy as np
from scipy import signal
from pathlib import Path

def zscore_norm(x, eps=1e-6):
    x = np.asarray(x, dtype=np.float32)
    m = x.mean(axis=-1, keepdims=True)
    s = x.std(axis=-1, keepdims=True)
    s[s < eps] = 1.0
    return (x - m) / s

def pad_or_truncate(x, target_len):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        if x.shape[0] >= target_len:
            return x[:target_len]
        else:
            pad = target_len - x.shape[0]
            return np.pad(x, (0, pad), mode='constant')
    elif x.ndim == 2:
        # assume shape (leads, samples)
        if x.shape[1] >= target_len:
            return x[:, :target_len]
        else:
            pad = target_len - x.shape[1]
            return np.pad(x, ((0,0),(0,pad)), mode='constant')
    else:
        raise ValueError('Unexpected signal shape')

def safe_save_npz(path: Path, signal_array, label:int, metadata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if metadata is None:
        metadata = {}
    np.savez_compressed(path, signal=signal_array.astype(np.float32), label=int(label), metadata=json.dumps(metadata))

def load_npz(path:Path):
    with np.load(path, allow_pickle=True) as d:
        sig = d['signal'].astype(np.float32)
        lbl = int(d['label'])
        meta = json.loads(d['metadata'].tolist() if hasattr(d['metadata'],'tolist') else d['metadata'])
    return sig, lbl, meta
"""
))

# --- Mapping loader ---
cells.append(new_code_cell(
"""# Load unified mapping if present; else load candidate, else fallback
from collections import Counter
UNIFIED_CSV = LOGS_DIR / "unified_label_mapping.csv"
CANDIDATE_CSV = LOGS_DIR / "unified_label_mapping.candidate.csv"

mapping_index = {}
if UNIFIED_CSV.exists() and UNIFIED_CSV.stat().st_size>0:
    df_map = pd.read_csv(UNIFIED_CSV, dtype=str).fillna('')
    print('Loaded unified mapping:', UNIFIED_CSV, len(df_map))
else:
    if CANDIDATE_CSV.exists() and CANDIDATE_CSV.stat().st_size>0:
        df_map = pd.read_csv(CANDIDATE_CSV, dtype=str).fillna('')
        print('Loaded candidate mapping:', CANDIDATE_CSV, len(df_map))
    else:
        df_map = pd.DataFrame(columns=['dataset','record_id','mapped_label'])
        print('No mapping CSV found; will default to OTHER')

# Build mapping index (dataset -> key -> label)
for _, row in df_map.iterrows():
    ds = str(row.get('dataset','')).strip()
    rid = str(row.get('record_id','')).strip().replace('\\\\','/').strip('/')
    lab = str(row.get('mapped_label','')).strip().upper()
    if not ds or not rid:
        continue
    mapping_index.setdefault(ds, {})[rid] = lab

print('Datasets in mapping:', list(mapping_index.keys())[:10])
"""
))

cells.append(new_code_cell(
"""# label lookup utility used during preprocessing
def lookup_mapped_label(dataset_name, record_id):
    idx = mapping_index.get(dataset_name, {})
    if record_id in idx:
        lab = idx[record_id].upper()
        return lab if lab in LABEL_TO_INT else 'OTHER'
    # try basename
    base = record_id.split('/')[-1]
    if base in idx:
        lab = idx[base].upper()
        return lab if lab in LABEL_TO_INT else 'OTHER'
    return 'OTHER'
"""
))

# --- Preprocessing cell ---
cells.append(new_markdown_cell("## Preprocessing (streaming). This cell scans supported datasets and writes per-record .npz files into artifacts/processed/records. It is I/O-heavy and may take hours for full dataset."))

cells.append(new_code_cell(
"""# Preprocessing: very conservative memory-safe loop
import wfdb
import scipy.io
from pathlib import Path
from tqdm import tqdm
import traceback

RECORDS_DIR = PROCESSED_DIR / "records"
RECORDS_DIR.mkdir(parents=True, exist_ok=True)

# helper to read recordings (WFDB .hea/.dat or .mat)
def read_record_generic(full_path: Path):
    # returns (signal (n_leads, n_samples), fs, meta_dict)
    try:
        if full_path.suffix.lower() == '.mat':
            data = scipy.io.loadmat(str(full_path))
            # try several common keys
            for k in ['val','data','sig','ecg']:
                if k in data:
                    arr = data[k]
                    arr = np.asarray(arr, dtype=np.float32)
                    if arr.ndim==2 and arr.shape[0] > arr.shape[1]:
                        # ensure shape (leads, samples)
                        return arr, int(data.get('fs', TARGET_FS)), {'source':'mat','path':str(full_path)}
            # fallback - find first numeric
            arr = None
            for v in data.values():
                if isinstance(v, np.ndarray) and v.ndim==2:
                    arr = v.astype(np.float32)
                    break
            if arr is None:
                raise RuntimeError('No 2D array found in mat')
            return arr, int(data.get('fs', TARGET_FS)), {'source':'mat','path':str(full_path)}
        else:
            # WFDB read using record name without .hea
            rec_dir = full_path.parent
            rec_name = full_path.stem
            record = wfdb.rdrecord(str(full_path.with_suffix('')))
            sig = np.asarray(record.p_signal.T, dtype=np.float32)  # shape (leads, samples)
            fs = int(getattr(record, 'fs', TARGET_FS))
            return sig, fs, {'source':'wfdb','path':str(full_path)}
    except Exception as e:
        # bubble up
        raise

# iterate datasets (supported minimal set)
candidates = []
if DATASET_DIR.exists():
    for ds in sorted(DATASET_DIR.iterdir()):
        if ds.is_dir():
            candidates.append(ds)
print('Datasets found:', [p.name for p in candidates])

# We'll process with a limit if provided
LIMIT = int(os.environ.get('ECG_PREPROCESS_LIMIT', 0))
print('Processing limit (0 means all):', LIMIT)

manifest = []
skipped = 0
processed = 0

# For speed and safety, define file patterns per dataset (common)
patterns = {
    'ptb-xl': ['**/*.dat','**/*.hea','**/*_hr.mat','**/*_lr.mat'],
    'CinC2017': ['**/*.mat','**/*.hea','**/*.atr','training/*.mat'],
    'PTB_Diagnostic': ['**/*.dat','**/*.hea'],
    'Chapman_Shaoxing': ['**/*.dat','**/*.hea','**/*.mat']
}

# If wfdb package missing, fallback to synthetic creation
if not candidates:
    print('No dataset folders – generating synthetic samples for quick smoke tests')
    t = np.linspace(0, 10, TARGET_SAMPLES, dtype=np.float32)
    for i in range(200):
        s = np.sin(2*np.pi*(1+i*0.1)*t).astype(np.float32)
        out = RECORDS_DIR / f"SYNTH_{i:05d}.npz"
        safe_save_npz(out, s, i%len(LABEL_ORDER), {'dataset':'SYNTH'})
        manifest.append({'path': f"records/{out.name}", 'label': int(i%len(LABEL_ORDER))})
    processed = len(manifest)
else:
    # iterate dataset folders and patterns
    for ds in candidates:
        ds_name = ds.name
        pat_list = patterns.get(ds_name, ['**/*.hea','**/*.mat','**/*.dat'])
        files = []
        for pat in pat_list:
            files.extend(list(ds.rglob(pat)))
        # prefer .hea as index entries: convert to unique set
        files = sorted(set(files))
        if LIMIT and processed >= LIMIT:
            break
        for fpath in tqdm(files, desc=f"Processing {ds_name}", unit='file'):
            try:
                # simple TRY: read using wfdb or mat loader; if fails, skip
                try:
                    sig, fs, meta = read_record_generic(fpath)
                except Exception:
                    # if WFDB read fails try reading .hea by name
                    try:
                        rec = wfdb.rdrecord(str(fpath.with_suffix('')))
                        sig = np.asarray(rec.p_signal.T, dtype=np.float32)
                        fs = int(getattr(rec, 'fs', TARGET_FS))
                        meta = {'source':'wfdb'}
                    except Exception as e:
                        skipped += 1
                        continue

                # resample if needed
                if fs != TARGET_FS:
                    # resample each lead
                    num = int(round(sig.shape[1] * (TARGET_FS / float(fs))))
                    sig = signal.resample(sig, num, axis=1).astype(np.float32)
                    fs = TARGET_FS

                # normalize and pad/truncate
                if sig.ndim == 1:
                    sig = np.expand_dims(sig, 0)
                sig = zscore_norm(sig)
                sig = pad_or_truncate(sig, TARGET_SAMPLES)

                # build record id relative to dataset root
                try:
                    rel = fpath.relative_to(DATASET_DIR).as_posix()
                except Exception:
                    rel = fpath.name
                # lookup mapped label
                mapped = lookup_mapped_label(ds_name, rel)
                label_int = LABEL_TO_INT.get(mapped, LABEL_TO_INT['OTHER'])

                out_file = RECORDS_DIR / f"{ds_name}__{rel.replace('/','__').replace('.','_')}.npz"
                safe_save_npz(out_file, sig, label_int, {'dataset': ds_name, 'src': rel})
                manifest.append({'path': f"records/{out_file.name}", 'label': label_int})
                processed += 1

                if LIMIT and processed >= LIMIT:
                    break
            except Exception as e:
                skipped += 1
                # write short log entry
                with open(LOGS_DIR / "preprocess_errors.log", "a", encoding="utf-8") as fh:
                    fh.write(f"{fpath} -> {repr(e)}\\n")
                continue

print('Done. processed:', processed, 'skipped:', skipped)
# persist manifest and splits
import json
with open(PROCESSED_DIR / "manifest.jsonl", "w", encoding="utf-8") as fh:
    for rec in manifest:
        fh.write(json.dumps(rec) + "\\n")

# build simple stratified splits
from sklearn.model_selection import train_test_split
paths = [m['path'] for m in manifest]
labels = [m['label'] for m in manifest]
if paths:
    train_p, test_p, y_train, y_test = train_test_split(paths, labels, test_size=0.2, stratify=labels, random_state=SEED)
    val_p, test_p, y_val, y_test = train_test_split(test_p, y_test, test_size=0.5, stratify=y_test, random_state=SEED)
    splits = {'paths': {'train': train_p, 'val': val_p, 'test': test_p}}
    with open(PROCESSED_DIR / "splits.json", "w", encoding="utf-8") as fh:
        json.dump(splits, fh, indent=2)
    print('Splits saved. Train:', len(train_p), 'Val:', len(val_p), 'Test:', len(test_p))
else:
    print('No manifest entries – nothing to split.')
"""
))

# --- Dataset & DataLoader ---
cells.append(new_markdown_cell("## Dataset & DataLoader (lazy loading)"))

cells.append(new_code_cell(
"""# PyTorch Dataset reading .npz files lazily
import torch
from torch.utils.data import Dataset, DataLoader

class ECGDataset(Dataset):
    def __init__(self, entries, base_dir):
        self.entries = entries
        self.base_dir = Path(base_dir)
    def __len__(self):
        return len(self.entries)
    def __getitem__(self, idx):
        p = self.entries[idx]
        sig, label, meta = load_npz(self.base_dir / p.split('records/')[-1])
        # ensure shape (1, samples)
        if sig.ndim == 2:
            # use mean across leads for single-lead baseline
            sig = sig.mean(axis=0, keepdims=True)
        tensor = torch.from_numpy(sig).float()
        return tensor, torch.tensor(label, dtype=torch.long)

# quick loader constructor
def build_loaders(limit=None):
    import json
    with open(PROCESSED_DIR / 'splits.json','r') as fh:
        splits = json.load(fh)
    train_list = splits['paths']['train']
    val_list = splits['paths']['val']
    test_list = splits['paths']['test']
    if limit:
        train_list = train_list[:limit]
        val_list = val_list[:int(limit*0.2)]
        test_list = test_list[:int(limit*0.2)]
    train_ds = ECGDataset(train_list, PROCESSED_DIR / 'records')
    val_ds = ECGDataset(val_list, PROCESSED_DIR / 'records')
    test_ds = ECGDataset(test_list, PROCESSED_DIR / 'records')
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader

# show example batch if available
try:
    tr, va, te = build_loaders(limit=16)
    xb, yb = next(iter(tr))
    print('example batch:', xb.shape, yb.shape)
except Exception as e:
    print('build_loaders failed:', e)
"""
))

# --- Model definition ---
cells.append(new_markdown_cell("## Model (compact 1D ResNet-like). GPU intensive: forward/backward, mixed precision."))

cells.append(new_code_cell(
"""# Simple 1D CNN with residual blocks
import torch.nn as nn
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=7, s=2):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=k//2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
    def forward(self,x):
        return self.act(self.bn(self.conv(x)))

class SmallResNet1D(nn.Module):
    def __init__(self, in_ch=1, num_classes=len(LABEL_ORDER)):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock(in_ch, 16, k=11, s=2),
            ConvBlock(16, 32, k=9, s=2),
        )
        self.res1 = nn.Sequential(
            ConvBlock(32, 32, k=7, s=1),
            ConvBlock(32, 32, k=5, s=1),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
    def forward(self,x):
        x = self.stem(x)
        r = self.res1(x)
        x = x + r
        return self.head(x)

model = SmallResNet1D().to(DEVICE)
print(model)
print('num params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
"""
))

# --- Training loop ---
cells.append(new_markdown_cell("## Training loop (uses AMP when available). Logs metrics and saves checkpoints."))

cells.append(new_code_cell(
"""# Training loop with evaluation function
import torch
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
import json, os

def evaluate(model, loader):
    model.eval()
    ys, ypreds = [], []
    with torch.no_grad():
        for xb,yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().tolist()
            ys.extend(yb.cpu().tolist())
            ypreds.extend(preds)
    report = {
        'acc': accuracy_score(ys, ypreds),
        'f1_macro': f1_score(ys, ypreds, average='macro'),
        'confusion': confusion_matrix(ys, ypreds).tolist()
    }
    return report

def train(model, train_loader, val_loader, epochs=EPOCHS, lr=LR):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    criterion = torch.nn.CrossEntropyLoss()
    best_val = -1.0
    history = {'train_loss':[], 'val_f1':[]}
    for ep in range(epochs):
        model.train()
        losses = []
        for xb,yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        avg_loss = float(np.mean(losses)) if losses else 0.0
        val_report = evaluate(model, val_loader) if val_loader is not None else {}
        val_f1 = val_report.get('f1_macro', 0.0)
        history['train_loss'].append(avg_loss)
        history['val_f1'].append(val_f1)
        print(f"Epoch {ep+1}/{epochs} loss={avg_loss:.4f} val_f1={val_f1:.4f}")
        # save checkpoint
        ckpt = PROCESSED_DIR / f"checkpoint_ep{ep+1}.pth"
        torch.save({'model_state': model.state_dict(), 'opt_state': opt.state_dict(), 'epoch': ep+1, 'history': history}, ckpt)
        if val_f1 > best_val:
            best_val = val_f1
            torch.save({'model_state': model.state_dict(), 'opt_state': opt.state_dict(), 'epoch': ep+1}, PROCESSED_DIR / "best_model.pth")
    # final history
    with open(PROCESSED_DIR / "training_history.json","w",encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    return history
"""
))

# --- Evaluation & plots ---
cells.append(new_markdown_cell("## Evaluation & Plots"))

cells.append(new_code_cell(
"""# Plot training curves and confusion matrix if evaluation available
import matplotlib.pyplot as plt
def plot_history(history, savepath=FIGURES_DIR/'training_curves.png'):
    plt.figure(figsize=(8,4))
    plt.plot(history.get('train_loss',[]), label='train_loss')
    plt.plot(history.get('val_f1',[]), label='val_f1')
    plt.legend()
    plt.title('Training history')
    plt.savefig(savepath)
    plt.close()
    print('Saved', savepath)

# Confusion matrix plotting helper
def plot_confusion(cm, labels=LABEL_ORDER, savepath=FIGURES_DIR/'confusion.png'):
    import seaborn as sns
    plt.figure(figsize=(6,5))
    sns.heatmap(np.array(cm), annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.savefig(savepath)
    plt.close()
    print('Saved', savepath)
"""
))

# --- Smoke tests ---
cells.append(new_markdown_cell("## Smoke tests — quick checks to ensure pipeline integrity"))

cells.append(new_code_cell(
"""# Basic smoke tests: manifest existence, ability to load one record, model forward pass
errors = []
if not (PROCESSED_DIR / 'manifest.jsonl').exists():
    errors.append('manifest.jsonl missing')
else:
    # try to load first manifest entry
    import json
    with open(PROCESSED_DIR / 'manifest.jsonl','r',encoding='utf-8') as fh:
        first = fh.readline().strip()
    if not first:
        errors.append('manifest empty')
    else:
        rec = json.loads(first)
        path = PROCESSED_DIR / 'records' / Path(rec['path']).name
        try:
            sig, lbl, meta = load_npz(path)
            print('Loaded sample shape', sig.shape, 'label', lbl)
        except Exception as e:
            errors.append(f'load_npz failed: {e}')

# model forward test
try:
    m = model.to(DEVICE)
    m.eval()
    dummy = torch.randn(2,1,TARGET_SAMPLES).to(DEVICE)
    with torch.no_grad():
        out = m(dummy)
    print('Model forward ok, out shape', out.shape)
except Exception as e:
    errors.append(f'model forward failed: {e}')

if errors:
    print('SMOKE TESTS FOUND ISSUES:')
    for e in errors:
        print('-', e)
else:
    print('SMOKE TESTS PASSED')
"""
))

# --- Orchestrator cell ---
cells.append(new_markdown_cell("## Orchestrator: run preprocessing -> build loaders -> train -> evaluate\nUse this cell to run the full pipeline (careful: preprocessing may take long)"))

cells.append(new_code_cell(
"""# Orchestrator. Set env var ECG_PREPROCESS_LIMIT to test quickly.
import os
def run_full(limit=None, do_preprocess=True, do_train=True):
    if do_preprocess:
        print('Run preprocessing cell above or set ECG_PREPROCESS_LIMIT and re-run notebook headless')
        # We included a full preprocessing cell earlier; simply re-run the cell if needed.
    # Build loaders
    try:
        tr, va, te = build_loaders(limit=limit)
        print('Loaders ready. sizes:', len(tr.dataset), len(va.dataset), len(te.dataset))
    except Exception as e:
        print('Failed to build loaders:', e)
        return
    if do_train:
        hist = train(model, tr, va, epochs=EPOCHS)
        plot_history(hist)
    # final eval
    rep = evaluate(model, te)
    print('Test eval:', rep)
    # confusion matrix plot
    if 'confusion' in rep:
        plot_confusion(rep['confusion'])
    return rep

# Example usage:
# run_full(limit=200, do_preprocess=False, do_train=False)
print('Orchestrator ready. To run: run_full(limit=500, do_preprocess=False, do_train=True)')
"""
))

# --- Final notes cell ---
cells.append(new_markdown_cell(
"## Final notes\n\n"
"- For a quick smoke run set `ECG_PREPROCESS_LIMIT=5000` in your environment and run the preprocessing cell.\n"
"- For full production, run headless overnight: `jupyter nbconvert --to notebook --execute notebooks/master_pipeline.ipynb --output logs/preprocess_run.ipynb`\n"
"- If you want me to generate a variant that uses TFRecords, ONNX export, MLflow logging, or multi-label training — say which and I'll produce it."
))

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"name":"python3", "display_name":"Python 3"},
    "language_info": {"name":"python"}
})

TARGET.write_text(nbformat.writes(nb), encoding="utf-8")
print("Wrote notebook to", TARGET)
