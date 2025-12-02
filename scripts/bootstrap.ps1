# ============================================================================
# ECG Research Project - Bootstrap Script
# ============================================================================
# This script sets up the complete directory structure and creates
# all necessary placeholder files for the ECG research pipeline.
# ============================================================================

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  ECG Research Project - Bootstrap Setup" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$ErrorActionPreference = "Stop"
$ProjectRoot = "D:\ecg-research"

# Navigate to project root
Set-Location $ProjectRoot

# ============================================================================
# 1. Create Directory Structure
# ============================================================================
Write-Host "[1/6] Creating directory structure..." -ForegroundColor Yellow

$directories = @(
    "dataset",
    "artifacts\processed\records",
    "artifacts\models",
    "artifacts\mlflow",
    "figures",
    "logs",
    "notebooks",
    "src",
    "scripts"
)

foreach ($dir in $directories) {
    if (-Not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  ✓ Created: $dir" -ForegroundColor Green
    } else {
        Write-Host "  ✓ Exists:  $dir" -ForegroundColor Gray
    }
}

# ============================================================================
# 2. Verify Dataset Folders
# ============================================================================
Write-Host ""
Write-Host "[2/6] Verifying raw datasets..." -ForegroundColor Yellow

$datasets = @("ptb-xl", "PTB_Diagnostic", "CinC2017", "Chapman_Shaoxing")
$allDatasetsExist = $true

foreach ($ds in $datasets) {
    $dsPath = Join-Path "dataset" $ds
    if (Test-Path $dsPath) {
        Write-Host "  ✓ Found: $ds" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Missing: $ds" -ForegroundColor Red
        $allDatasetsExist = $false
    }
}

if (-Not $allDatasetsExist) {
    Write-Host ""
    Write-Host "  ⚠ Warning: Some datasets are missing!" -ForegroundColor Yellow
    Write-Host "  Please ensure all datasets are downloaded to dataset\" -ForegroundColor Yellow
}

# ============================================================================
# 3. Create Python Source Files
# ============================================================================
Write-Host ""
Write-Host "[3/6] Creating Python source files..." -ForegroundColor Yellow

# src/__init__.py
$initContent = @"
"""
ECG Research Pipeline
A unified multi-dataset ECG preprocessing, training, and serving pipeline.
"""

__version__ = "0.1.0"
__author__ = "ECG Research Team"
"@

if (-Not (Test-Path "src\__init__.py")) {
    Set-Content -Path "src\__init__.py" -Value $initContent
    Write-Host "  ✓ Created: src\__init__.py" -ForegroundColor Green
} else {
    Write-Host "  ✓ Exists:  src\__init__.py" -ForegroundColor Gray
}

# src/utils.py
$utilsContent = @"
"""
Utility functions for ECG data I/O and common operations.
"""

import os
import json
import numpy as np
from pathlib import Path


def load_record(record_path):
    """Load a preprocessed ECG record from .npz file."""
    data = np.load(record_path)
    return {
        'signals': data['signals'],
        'labels': data['labels'],
        'metadata': data['metadata'].item() if 'metadata' in data else {}
    }


def save_record(record_path, signals, labels, metadata=None):
    """Save preprocessed ECG record to .npz file."""
    os.makedirs(os.path.dirname(record_path), exist_ok=True)
    if metadata is None:
        metadata = {}
    np.savez_compressed(record_path, signals=signals, labels=labels, metadata=metadata)


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


def load_label_mapping(mapping_path=None):
    """Load unified label mapping from CSV."""
    import pandas as pd
    if mapping_path is None:
        mapping_path = get_project_root() / "logs" / "unified_label_mapping.csv"
    return pd.read_csv(mapping_path)


def setup_logging(log_file="preprocess_run.log"):
    """Setup logging configuration."""
    import logging
    log_path = get_project_root() / "logs" / log_file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)
"@

if (-Not (Test-Path "src\utils.py")) {
    Set-Content -Path "src\utils.py" -Value $utilsContent
    Write-Host "  ✓ Created: src\utils.py" -ForegroundColor Green
} else {
    Write-Host "  ✓ Exists:  src\utils.py" -ForegroundColor Gray
}

# src/preprocessing.py
$preprocessingContent = @"
"""
ECG Preprocessing Module
Handles loading, filtering, resampling, and normalization of ECG signals.
"""

import numpy as np
from scipy import signal
import wfdb


def load_wfdb_record(record_path):
    """Load WFDB format ECG record."""
    record = wfdb.rdrecord(record_path)
    return record.p_signal, record.fs, record.__dict__


def bandpass_filter(ecg_signal, lowcut=0.5, highcut=40, fs=500, order=4):
    """Apply Butterworth bandpass filter to ECG signal."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, ecg_signal, axis=0)


def resample_signal(ecg_signal, original_fs, target_fs=500):
    """Resample ECG signal to target frequency."""
    if original_fs == target_fs:
        return ecg_signal
    num_samples = int(len(ecg_signal) * target_fs / original_fs)
    return signal.resample(ecg_signal, num_samples, axis=0)


def normalize_signal(ecg_signal, method='z-score'):
    """Normalize ECG signal using specified method."""
    if method == 'z-score':
        mean = np.mean(ecg_signal, axis=0)
        std = np.std(ecg_signal, axis=0)
        return (ecg_signal - mean) / (std + 1e-8)
    elif method == 'min-max':
        min_val = np.min(ecg_signal, axis=0)
        max_val = np.max(ecg_signal, axis=0)
        return (ecg_signal - min_val) / (max_val - min_val + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def preprocess_record(record_path, target_fs=500, target_length=5000):
    """Complete preprocessing pipeline for a single ECG record."""
    # Load record
    ecg_data, fs, metadata = load_wfdb_record(record_path)

    # Resample
    ecg_data = resample_signal(ecg_data, fs, target_fs)

    # Filter
    ecg_data = bandpass_filter(ecg_data, fs=target_fs)

    # Normalize
    ecg_data = normalize_signal(ecg_data, method='z-score')

    # Pad or truncate to target length
    current_length = len(ecg_data)
    if current_length < target_length:
        pad_width = ((0, target_length - current_length), (0, 0))
        ecg_data = np.pad(ecg_data, pad_width, mode='constant')
    else:
        ecg_data = ecg_data[:target_length]

    return ecg_data, metadata
"@

if (-Not (Test-Path "src\preprocessing.py")) {
    Set-Content -Path "src\preprocessing.py" -Value $preprocessingContent
    Write-Host "  ✓ Created: src\preprocessing.py" -ForegroundColor Green
} else {
    Write-Host "  ✓ Exists:  src\preprocessing.py" -ForegroundColor Gray
}

# src/training.py
$trainingContent = @"
"""
ECG Model Training Module
Defines models, training loops, and evaluation metrics.
"""

import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
from torch.utils.data import Dataset, DataLoader


class ECGDataset(Dataset):
    """PyTorch Dataset for preprocessed ECG records."""

    def __init__(self, record_paths, transform=None):
        self.record_paths = record_paths
        self.transform = transform

    def __len__(self):
        return len(self.record_paths)

    def __getitem__(self, idx):
        import numpy as np
        data = np.load(self.record_paths[idx])
        signals = torch.FloatTensor(data['signals'])
        labels = torch.FloatTensor(data['labels'])

        if self.transform:
            signals = self.transform(signals)

        return signals, labels


class ECGResNet1D(nn.Module):
    """1D ResNet architecture for ECG classification."""

    def __init__(self, num_classes, input_channels=12):
        super(ECGResNet1D, self).__init__()

        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Add residual blocks here
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1))
        layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(1, blocks):
            layers.append(nn.Conv1d(out_channels, out_channels, kernel_size=3,
                                   stride=1, padding=1))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0

    for signals, labels in dataloader:
        signals = signals.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for signals, labels in dataloader:
            signals = signals.to(device)
            labels = labels.to(device)

            outputs = model(signals)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

    return running_loss / len(dataloader)
"@

if (-Not (Test-Path "src\training.py")) {
    Set-Content -Path "src\training.py" -Value $trainingContent
    Write-Host "  ✓ Created: src\training.py" -ForegroundColor Green
} else {
    Write-Host "  ✓ Exists:  src\training.py" -ForegroundColor Gray
}

# src/serving.py
$servingContent = @"
"""
ECG Model Serving Module
Handles model loading and inference for production use.
"""

import torch
import mlflow
import numpy as np
from pathlib import Path


class ECGPredictor:
    """
    ECG inference class for loading trained models and making predictions.
    """

    def __init__(self, model_path=None, model_uri=None):
        """
        Initialize predictor with either a local model path or MLflow model URI.

        Args:
            model_path: Path to local .pth model file
            model_uri: MLflow model URI (e.g., 'models:/ECGClassifier/Production')
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model_uri:
            self.model = mlflow.pytorch.load_model(model_uri)
        elif model_path:
            self.model = torch.load(model_path, map_location=self.device)
        else:
            raise ValueError("Must provide either model_path or model_uri")

        self.model.to(self.device)
        self.model.eval()

    def predict(self, ecg_signal):
        """
        Make prediction on a single ECG signal.

        Args:
            ecg_signal: numpy array of shape (length, channels)

        Returns:
            predictions: numpy array of class probabilities
        """
        # Prepare input
        if isinstance(ecg_signal, np.ndarray):
            ecg_tensor = torch.FloatTensor(ecg_signal).unsqueeze(0)
        else:
            ecg_tensor = ecg_signal.unsqueeze(0)

        ecg_tensor = ecg_tensor.to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(ecg_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        return probabilities.cpu().numpy()[0]

    def predict_batch(self, ecg_signals):
        """
        Make predictions on a batch of ECG signals.

        Args:
            ecg_signals: numpy array of shape (batch_size, length, channels)

        Returns:
            predictions: numpy array of shape (batch_size, num_classes)
        """
        ecg_tensor = torch.FloatTensor(ecg_signals).to(self.device)

        with torch.no_grad():
            outputs = self.model(ecg_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        return probabilities.cpu().numpy()


def load_mlflow_model(run_id=None, model_name=None, stage='Production'):
    """
    Load model from MLflow.

    Args:
        run_id: MLflow run ID
        model_name: Registered model name
        stage: Model stage (Production, Staging, etc.)

    Returns:
        predictor: ECGPredictor instance
    """
    if model_name:
        model_uri = f"models:/{model_name}/{stage}"
    elif run_id:
        model_uri = f"runs:/{run_id}/model"
    else:
        raise ValueError("Must provide either run_id or model_name")

    return ECGPredictor(model_uri=model_uri)
"@

if (-Not (Test-Path "src\serving.py")) {
    Set-Content -Path "src\serving.py" -Value $servingContent
    Write-Host "  ✓ Created: src\serving.py" -ForegroundColor Green
} else {
    Write-Host "  ✓ Exists:  src\serving.py" -ForegroundColor Gray
}

# ============================================================================
# 4. Create Placeholder Log Files
# ============================================================================
Write-Host ""
Write-Host "[4/6] Creating placeholder log files..." -ForegroundColor Yellow

# logs/unified_label_mapping.csv
$labelMappingContent = @"
original_label,unified_label,dataset,snomed_code,description
NORM,Normal,ptb-xl,426783006,Normal sinus rhythm
SR,Normal,PTB_Diagnostic,426783006,Sinus rhythm
N,Normal,CinC2017,426783006,Normal
Normal,Normal,Chapman_Shaoxing,426783006,Normal ECG
AFIB,Atrial Fibrillation,ptb-xl,164889003,Atrial fibrillation
AF,Atrial Fibrillation,PTB_Diagnostic,164889003,Atrial fibrillation
A,Atrial Fibrillation,CinC2017,164889003,Atrial fibrillation
MI,Myocardial Infarction,ptb-xl,164865005,Myocardial infarction
IMI,Myocardial Infarction,PTB_Diagnostic,164865005,Inferior MI
AMI,Myocardial Infarction,PTB_Diagnostic,164865005,Anterior MI
STTC,ST-T Changes,ptb-xl,698252002,ST-T changes
OTHER,Other,ptb-xl,,,Miscellaneous abnormalities
"@

if (-Not (Test-Path "logs\unified_label_mapping.csv")) {
    Set-Content -Path "logs\unified_label_mapping.csv" -Value $labelMappingContent
    Write-Host "  ✓ Created: logs\unified_label_mapping.csv" -ForegroundColor Green
} else {
    Write-Host "  ✓ Exists:  logs\unified_label_mapping.csv" -ForegroundColor Gray
}

# logs/preprocess_run.log
$preprocessLogContent = @"
# ECG Preprocessing Run Log
# This file will be populated during preprocessing runs
# Timestamp: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
"@

if (-Not (Test-Path "logs\preprocess_run.log")) {
    Set-Content -Path "logs\preprocess_run.log" -Value $preprocessLogContent
    Write-Host "  ✓ Created: logs\preprocess_run.log" -ForegroundColor Green
} else {
    Write-Host "  ✓ Exists:  logs\preprocess_run.log" -ForegroundColor Gray
}

# ============================================================================
# 5. Create Placeholder Notebook
# ============================================================================
Write-Host ""
Write-Host "[5/6] Creating placeholder notebook..." -ForegroundColor Yellow

$notebookContent = @"
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# ECG Tensor Pipeline\n",
    "\n",
    "This notebook demonstrates the complete ECG preprocessing, training, and evaluation pipeline.\n",
    "\n",
    "## Steps:\n",
    "1. Load raw datasets\n",
    "2. Preprocess and unify labels\n",
    "3. Generate tensor records\n",
    "4. Train models\n",
    "5. Evaluate and visualize results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.append('../src')\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from pathlib import Path\n",
    "\n",
    "from utils import get_project_root, load_label_mapping\n",
    "from preprocessing import preprocess_record\n",
    "\n",
    "print('Setup complete!')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Dataset Overview"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# List available datasets\n",
    "project_root = get_project_root()\n",
    "dataset_dir = project_root / 'dataset'\n",
    "\n",
    "datasets = ['ptb-xl', 'PTB_Diagnostic', 'CinC2017', 'Chapman_Shaoxing']\n",
    "for ds in datasets:\n",
    "    ds_path = dataset_dir / ds\n",
    "    if ds_path.exists():\n",
    "        print(f'✓ {ds}: Found')\n",
    "    else:\n",
    "        print(f'✗ {ds}: Missing')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Load Label Mapping"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load unified label mapping\n",
    "label_mapping = load_label_mapping()\n",
    "print(f'Loaded {len(label_mapping)} label mappings')\n",
    "label_mapping.head(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Preprocessing Pipeline\n",
    "\n",
    "Add your preprocessing code here..."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# TODO: Implement preprocessing logic\n",
    "pass"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Training Pipeline\n",
    "\n",
    "Add your training code here..."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# TODO: Implement training logic\n",
    "pass"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Evaluation and Visualization\n",
    "\n",
    "Add your evaluation code here..."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# TODO: Implement evaluation and visualization\n",
    "pass"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
"@

if (-Not (Test-Path "notebooks\ecg_tensor_pipeline.ipynb")) {
    Set-Content -Path "notebooks\ecg_tensor_pipeline.ipynb" -Value $notebookContent
    Write-Host "  ✓ Created: notebooks\ecg_tensor_pipeline.ipynb" -ForegroundColor Green
} else {
    Write-Host "  ✓ Exists:  notebooks\ecg_tensor_pipeline.ipynb" -ForegroundColor Gray
}

# ============================================================================
# 6. Summary
# ============================================================================
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  ✓ Setup Complete!" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "Project structure is ready at: $ProjectRoot" -ForegroundColor White
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Run: .\scripts\download_check.ps1 (verify datasets)" -ForegroundColor White
Write-Host "  2. Install Python dependencies: pip install -r requirements.txt" -ForegroundColor White
Write-Host "  3. Open notebooks\ecg_tensor_pipeline.ipynb to start processing" -ForegroundColor White
Write-Host "  4. Run MLflow UI: mlflow ui --backend-store-uri file:///D:/ecg-research/artifacts/mlflow" -ForegroundColor White
Write-Host ""

