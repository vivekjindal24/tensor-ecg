# ECG Research Project

**Status**: ✅ Operational | **Last Updated**: December 2, 2025

A unified multi-dataset ECG preprocessing, training, and serving pipeline for cardiovascular disease classification.

## 🚀 Quick Start

```bash
# Run smoke test (1 minute)
python scripts\run_full_automation.py --mode smoke

# Run full pipeline (several hours)
python scripts\run_full_automation.py --mode full

# Open interactive notebook
jupyter notebook notebooks\master_pipeline.ipynb
```

📖 **See [QUICKSTART.md](QUICKSTART.md) for detailed instructions**

---

## 📊 Project Overview

This project implements a complete machine learning pipeline for ECG (electrocardiogram) signal analysis, integrating multiple public datasets to train robust deep learning models for cardiac condition classification.

### Key Features

- **Self-Contained Notebook**: Complete pipeline in `notebooks/master_pipeline.ipynb`
- **Multi-Dataset Integration**: Unified preprocessing for PTB-XL, PTB Diagnostic, CinC2017, and Chapman-Shaoxing datasets
- **Standardized Label Mapping**: 84,556 records mapped to MI, AF, BBB, NORM, OTHER
- **Memory-Safe Streaming**: Lazy loading, per-record processing, resumable
- **Deep Learning Pipeline**: PyTorch-based training with 1D CNN and mixed precision
- **GPU Support**: Automatic CUDA detection with AMP for faster training
- **Production Ready**: Comprehensive logging, checkpointing, and validation

## 📁 Project Structure

```
D:\ecg-research\
├── dataset\                          # Raw ECG datasets (not tracked in git)
│   ├── ptb-xl\                      # PTB-XL Database
│   ├── PTB_Diagnostic\              # PTB Diagnostic ECG Database
│   ├── CinC2017\                    # CinC Challenge 2017
│   └── Chapman_Shaoxing\            # Chapman-Shaoxing Database
│
├── artifacts\                        # Generated data and models (not tracked)
│   ├── processed\
│   │   └── records\                 # Preprocessed .npz records
│   ├── models\                      # Trained model checkpoints
│   └── mlflow\                      # MLflow tracking data
│
├── figures\                          # Plots and visualizations (not tracked)
│
├── logs\                             # Logs and mappings
│   ├── unified_label_mapping.csv    # Label mapping reference (tracked)
│   └── preprocess_run.log           # Preprocessing logs
│
├── notebooks\
│   └── ecg_tensor_pipeline.ipynb    # Main analysis notebook
│
├── src\                              # Source code
│   ├── __init__.py
│   ├── utils.py                     # Helper I/O functions
│   ├── preprocessing.py             # Signal preprocessing
│   ├── training.py                  # Model training
│   └── serving.py                   # Model inference
│
├── scripts\                          # Setup and utility scripts
│   ├── bootstrap.ps1                # Project initialization
│   ├── download_check.ps1           # Dataset verification
│   └── clean_other.ps1              # Label quality analysis
│
├── requirements.txt                  # Python dependencies
├── # ECG Research Pipeline

A complete end-to-end machine learning pipeline for ECG signal classification using deep learning.

## Overview

This project provides a production-ready pipeline for:
- Multi-dataset ECG signal preprocessing and harmonization
- Unified label mapping across heterogeneous ECG databases
- Deep learning model training with PyTorch
- Comprehensive evaluation and visualization

## Quick Start

### 1. Generate Unified Label Mapping
```powershell
python scripts\generate_unified_mapping.py
```

This scans all datasets and creates `logs/unified_label_mapping.candidate.csv` with 84,556+ records mapped to 5 target labels.

### 2. Run Complete Pipeline
```powershell
jupyter notebook notebooks\ecg_tensor_pipeline.ipynb
```

Or execute headless:
```powershell
jupyter nbconvert --to notebook --execute notebooks\ecg_tensor_pipeline.ipynb
```

## Target Labels

1. **MI** - Myocardial Infarction
2. **AF** - Atrial Fibrillation (including flutter)
3. **BBB** - Bundle Branch Blocks
4. **NORM** - Normal ECG
5. **OTHER** - All other conditions

## Project Structure

```
ecg-research/
├── Dataset/                    # Raw datasets (not in git)
├── artifacts/                  # Generated outputs
│   ├── processed/             # Preprocessed signals
│   ├── figures/               # Plots and visualizations
│   └── models/                # Trained models
├── notebooks/
│   └── ecg_tensor_pipeline.ipynb  # Complete pipeline
├── scripts/
│   ├── generate_unified_mapping.py  # Label harmonization
│   ├── create_notebook.py          # Notebook builder
│   └── README_generate_mapping.md  # Documentation
├── logs/
│   └── unified_label_mapping.csv   # Label mappings
└── requirements.txt
```

## Features

✅ **Unified Label Mapping**: Automatic harmonization of labels across 4 major ECG datasets  
✅ **Streaming Preprocessing**: Memory-efficient per-record processing  
✅ **Lazy Loading**: PyTorch Dataset with on-demand file loading  
✅ **GPU Acceleration**: Mixed precision training with CUDA  
✅ **Comprehensive Evaluation**: Confusion matrix, ROC, PR curves, per-class metrics  
✅ **Reproducible**: Fixed seeds and deterministic operations  
✅ **Production-Ready**: Type hints, error handling, logging  

## Supported Datasets

- **PTB-XL** (21,799 records)
- **CinC2017** (17,056 records)
- **PTB Diagnostic** (549 records)
- **Chapman-Shaoxing** (45,152 records)

## Requirements

```
Python 3.10+
numpy, scipy, pandas, wfdb
torch, torchvision
scikit-learn, matplotlib, tqdm
jupyter, nbformat
```

Install all dependencies:
```powershell
pip install -r requirements.txt
```

## Pipeline Workflow

1. **Raw Datasets** → Multiple ECG databases in various formats
2. **Label Mapping** → Unified mapping to 5 target labels
3. **Preprocessing** → Resample (500Hz), normalize, pad to 5000 samples
4. **Dataset Creation** → Stratified 80/10/10 train/val/test splits
5. **Training** → 1D CNN with mixed precision and data augmentation
6. **Evaluation** → Full metrics suite and visualizations

## Model Architecture

- **Type**: 1D Convolutional Neural Network
- **Input**: (1, 5000) - Single-lead ECG, 10 seconds at 500 Hz
- **Output**: 5-class classification (MI, AF, BBB, NORM, OTHER)
- **Features**: BatchNorm, Dropout, Residual connections, Global pooling

## Configuration

Edit constants in notebook Cell 3:
```python
TARGET_FS = 500          # Sampling frequency (Hz)
TARGET_SAMPLES = 5000    # Signal length
BATCH_SIZE = 32          # Training batch size
EPOCHS = 10              # Training epochs
LR = 1e-3                # Learning rate
```

## Output Artifacts

- **Preprocessed Data**: `artifacts/processed/records/*.npz`
- **Model Checkpoints**: `artifacts/processed/checkpoints/best_model.pth`
- **Evaluation Results**: `artifacts/processed/evaluation_results.json`
- **Training History**: `artifacts/processed/training_history.json`
- **Visualizations**: `artifacts/figures/*.png`

## Usage Examples

### Generate Label Mapping
```powershell
cd D:\ecg-research
python scripts\generate_unified_mapping.py

# Review output
notepad logs\unified_label_mapping.candidate.csv

# Finalize
cp logs\unified_label_mapping.candidate.csv logs\unified_label_mapping.csv
```

### Run Pipeline
```powershell
# Interactive
jupyter notebook notebooks\ecg_tensor_pipeline.ipynb

# Headless execution
jupyter nbconvert --to notebook --execute notebooks\ecg_tensor_pipeline.ipynb
```

### Inference Example
```python
import torch
from pathlib import Path
import numpy as np

# Load model
model = ECGNet1D(n_classes=5)
checkpoint = torch.load('artifacts/processed/checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load signal
signal = np.load('artifacts/processed/records/example.npz')['signal']
signal_tensor = torch.from_numpy(signal).float().unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(signal_tensor)
    predicted_class = output.argmax(dim=1).item()
    
print(f"Predicted: {LABEL_ORDER[predicted_class]}")
```

## Performance

**On NVIDIA L4 GPU**:
- Preprocessing: ~50-100 records/second
- Training: ~200-500 samples/second
- Inference: ~2000-5000 samples/second

## Documentation

- **Complete Setup Guide**: `COMPLETE_SETUP_SUMMARY.md`
- **Label Mapping Guide**: `scripts/README_generate_mapping.md`
- **Implementation Guide**: `IMPLEMENTATION_GUIDE.md`
- **Pipeline Status**: `PIPELINE_STATUS.md`

## Troubleshooting

### Windows Event Loop Error
Add at the top of your script:
```python
import asyncio, sys
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
```

### CUDA Out of Memory
Reduce batch size:
```python
BATCH_SIZE = 16  # or 8
```

### Missing wfdb Package
```powershell
pip install wfdb
```

## Contributing

This is a research project. Feel free to:
- Add new datasets
- Improve label mapping heuristics
- Implement new model architectures
- Add data augmentation techniques

## License

Research and educational use. Please cite original dataset sources.

## Citation

If using this pipeline, please cite the original ECG databases:
- PTB-XL: Wagner et al., Scientific Data, 2020
- CinC2017: Clifford et al., Computing in Cardiology, 2017
- PTB Diagnostic: Bousseljot et al., IEEE EMB Magazine, 1995

## Contact

For questions or issues, check the documentation or review the notebook's smoke tests.

---

**Status**: ✅ Ready for production use  
**Last Updated**: December 1, 2025  
**Version**: 1.0.0                         # This file
└── .gitignore                        # Git ignore rules
```

## 🚀 Getting Started

### Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: Windows (PowerShell scripts provided)
- **GPU**: Optional but recommended (CUDA 11.8+ for PyTorch)
- **Storage**: ~50GB for datasets and artifacts

### Installation

1. **Clone or navigate to the project directory**:
   ```powershell
   cd D:\ecg-research
   ```

2. **Run the bootstrap script to set up the project structure**:
   ```powershell
   .\scripts\bootstrap.ps1
   ```

3. **Create and activate a Python virtual environment**:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

4. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

5. **Verify dataset presence**:
   ```powershell
   .\scripts\download_check.ps1
   ```

### Dataset Sources

If datasets are missing, download them from PhysioNet:

- **PTB-XL**: https://physionet.org/content/ptb-xl/
- **PTB Diagnostic**: https://physionet.org/content/ptbdb/
- **CinC2017**: https://physionet.org/content/challenge-2017/
- **Chapman-Shaoxing**: https://physionet.org/content/ecg-arrhythmia/

Extract all datasets to `D:\ecg-research\dataset\` maintaining the folder names as specified.

## 🔬 Usage

### 1. Preprocessing

Open and run the Jupyter notebook:

```powershell
jupyter lab notebooks\ecg_tensor_pipeline.ipynb
```

The preprocessing pipeline:
- Loads raw WFDB format ECG records
- Applies bandpass filtering (0.5-40 Hz)
- Resamples to 500 Hz
- Normalizes signals (z-score)
- Pads/truncates to fixed length (5000 samples = 10 seconds)
- Maps labels to unified categories
- Saves as compressed .npz files

### 2. Training

Run training from the notebook or use the training module:

```python
from src.training import ECGResNet1D, train_epoch, evaluate
import mlflow

# Initialize MLflow
mlflow.set_tracking_uri("file:///D:/ecg-research/artifacts/mlflow")
mlflow.set_experiment("ecg-classification")

# Load data, train model, log metrics...
```

### 3. Model Evaluation

Evaluation metrics include:
- ROC curves (saved to `figures/`)
- Precision-Recall curves
- Confusion matrices
- Per-class F1 scores

### 4. Model Serving

Load and use trained models for inference:

```python
from src.serving import ECGPredictor

# Load model
predictor = ECGPredictor(model_path="artifacts/models/best_model.pth")

# Make prediction
probabilities = predictor.predict(ecg_signal)
```

### 5. MLflow UI

Track experiments and compare models:

```powershell
mlflow ui --backend-store-uri file:///D:/ecg-research/artifacts/mlflow
```

Then open: http://localhost:5000

## 📋 Label Mapping

The project uses a unified label mapping system defined in `logs/unified_label_mapping.csv`:

- **Normal**: Normal sinus rhythm
- **Atrial Fibrillation**: AF, AFIB variants
- **Myocardial Infarction**: MI, AMI, IMI
- **ST-T Changes**: ST segment and T-wave abnormalities
- **Other**: Miscellaneous conditions

To analyze label quality and identify excessive "OTHER" labels:

```powershell
.\scripts\clean_other.ps1
```

## 🛠️ Maintenance Scripts

### Bootstrap Script
Creates the complete project structure and all placeholder files:
```powershell
.\scripts\bootstrap.ps1
```

### Dataset Verification
Checks dataset existence, file counts, and sizes:
```powershell
.\scripts\download_check.ps1
```

### Label Quality Analysis
Identifies problematic label mappings:
```powershell
.\scripts\clean_other.ps1
```

## 📊 Datasets Used

| Dataset | Records | Classes | Sampling Rate | Source |
|---------|---------|---------|---------------|--------|
| PTB-XL | 21,837 | 71 | 100/500 Hz | PhysioNet |
| PTB Diagnostic | 549 | 16 | 1000 Hz | PhysioNet |
| CinC2017 | 8,528 | 4 | 300 Hz | PhysioNet |
| Chapman-Shaoxing | 10,646 | 11 | 500 Hz | PhysioNet |

## 🔧 Technology Stack

- **Data Processing**: NumPy, Pandas, SciPy, WFDB
- **Deep Learning**: PyTorch
- **Experiment Tracking**: MLflow
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Notebooks**: Jupyter Lab

## 📝 License

This project uses publicly available datasets from PhysioNet. Please cite the original dataset papers when using this code:

- PTB-XL: Wagner et al., "PTB-XL, a large publicly available electrocardiography dataset" (2020)
- PTB Diagnostic: Bousseljot et al., "Nutzung der EKG-Signaldatenbank CARDIODAT der PTB" (1995)
- CinC2017: Clifford et al., "AF Classification from a Short Single Lead ECG Recording" (2017)
- Chapman-Shaoxing: Zheng et al., "A 12-lead electrocardiogram database for arrhythmia research" (2020)

## 🤝 Contributing

This is a research project. For questions or collaboration opportunities, please refer to the project maintainer.

## 📧 Contact

For questions or issues, please open an issue in the project repository or contact the project maintainer.

---

**Last Updated**: November 29, 2025  
**Version**: 0.1.0

