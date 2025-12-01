"""Model smoke test - verify shapes and forward pass"""
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Simple 1D CNN for testing
class ECGNet1D(nn.Module):
    def __init__(self, n_classes=5, input_channels=1, base_channels=32, dropout=0.3):
        super(ECGNet1D, self).__init__()
        
        self.conv1 = nn.Conv1d(input_channels, base_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(base_channels)
        
        self.conv2 = nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(base_channels * 2)
        
        self.conv3 = nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(base_channels * 4)
        
        self.conv4 = nn.Conv1d(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(base_channels * 8)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base_channels * 8, n_classes)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


ROOT = Path(__file__).parent.parent
PROCESSED_DIR = ROOT / "artifacts" / "processed"
RECORDS_DIR = PROCESSED_DIR / "records"
CHECKPOINTS_DIR = PROCESSED_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(exist_ok=True)

print("="*80)
print("MODEL SMOKE TEST")
print("="*80)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# Load splits
splits_file = PROCESSED_DIR / "splits.json"
with open(splits_file, 'r') as f:
    splits = json.load(f)

# Load 3 samples from each label (if available)
print("\nLoading sample records from each label...")
by_label = {}
for entry in splits['train'][:100]:  # Check first 100
    label = entry['label']
    if label not in by_label:
        by_label[label] = []
    if len(by_label[label]) < 3:
        by_label[label].append(entry)

print(f"Loaded samples for {len(by_label)} labels")

# Load signals into a batch
signals = []
labels = []
for label, entries in by_label.items():
    for entry in entries:
        npy_file = PROCESSED_DIR / entry['path']
        signal = np.load(npy_file, allow_pickle=False)
        signals.append(signal)
        labels.append(label)

batch_signals = torch.from_numpy(np.stack(signals)).float()
batch_labels = torch.tensor(labels, dtype=torch.long)

print(f"\nBatch shape: {batch_signals.shape}")
print(f"Labels shape: {batch_labels.shape}")
print(f"Label values: {batch_labels.tolist()}")

# Create model
model = ECGNet1D(n_classes=5, base_channels=32, dropout=0.3)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {total_params:,}")

# Forward pass
batch_signals = batch_signals.to(device)
batch_labels = batch_labels.to(device)

model.eval()
with torch.no_grad():
    outputs = model(batch_signals)
    predictions = outputs.argmax(dim=1)

print(f"\nForward pass successful!")
print(f"Output shape: {outputs.shape}")
print(f"Predictions: {predictions.cpu().tolist()}")
print(f"True labels: {batch_labels.cpu().tolist()}")

# Save skeleton checkpoint
checkpoint_path = CHECKPOINTS_DIR / "checkpoint_skel.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'n_classes': 5,
    'base_channels': 32,
    'dropout': 0.3
}, checkpoint_path)

print(f"\n✓ Saved skeleton checkpoint to {checkpoint_path}")

print("\n" + "="*80)
print("MODEL SMOKE TEST PASSED")
print("="*80)

