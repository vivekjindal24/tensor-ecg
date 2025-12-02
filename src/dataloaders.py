"""
ECG Dataloaders Module
Implements PyTorch Dataset and DataLoader creation with lazy loading.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .utils import read_json

logger = logging.getLogger(__name__)


class ECGDataset(Dataset):
    """
    PyTorch Dataset for lazy loading of preprocessed ECG records.
    Reads individual compressed .npz files on demand.
    """

    def __init__(
        self,
        manifest: list,
        processed_dir: Path,
        augment: bool = False,
        noise_std: float = 0.01
    ):
        """
        Initialize ECG Dataset.

        Args:
            manifest: List of dicts with 'path' and 'label' keys
            processed_dir: Root directory containing processed records
            augment: Whether to apply augmentation
            noise_std: Standard deviation for gaussian noise augmentation
        """
        self.manifest = manifest
        self.processed_dir = Path(processed_dir)
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return a single record.

        Args:
            idx: Index of record

        Returns:
            Tuple of (signal_tensor, label_tensor)
        """
        item = self.manifest[idx]
        record_path = self.processed_dir / item['path']

        try:
            data = np.load(record_path)
            signal = data['signal'].astype(np.float32)
            label = int(data['label'])
        except Exception as e:
            logger.error(f"Failed to load {record_path}: {e}")
            # Return zero signal and label
            signal = np.zeros(5000, dtype=np.float32)
            label = 0

        # Optional augmentation (on CPU for reproducibility)
        if self.augment:
            # Add small gaussian noise
            if self.noise_std > 0:
                noise = np.random.normal(0, self.noise_std, signal.shape).astype(np.float32)
                signal = signal + noise

            # Random crop (if longer than needed, take random segment)
            # Here we assume signal is already fixed length, so this is optional placeholder

        # Convert to tensor
        signal_tensor = torch.from_numpy(signal).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        return signal_tensor, label_tensor


def collate_fn(batch: list) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function to reshape signals to (batch, 1, samples).

    Args:
        batch: List of (signal, label) tuples

    Returns:
        Tuple of (signals_batch, labels_batch)
    """
    signals = []
    labels = []

    for signal, label in batch:
        signals.append(signal.unsqueeze(0))  # Add channel dimension
        labels.append(label)

    signals_batch = torch.stack(signals, dim=0)  # Shape: (batch, 1, samples)
    labels_batch = torch.stack(labels, dim=0)

    return signals_batch, labels_batch


def create_dataloaders(
    splits_json: Path,
    processed_dir: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    augment_train: bool = False
) -> Dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders from splits.json.

    Args:
        splits_json: Path to splits.json file
        processed_dir: Root directory containing processed records
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        augment_train: Whether to augment training data

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    splits = read_json(splits_json)

    train_manifest = splits['train']
    val_manifest = splits.get('val', [])
    test_manifest = splits.get('test', [])

    logger.info(f"Creating dataloaders: train={len(train_manifest)}, "
                f"val={len(val_manifest)}, test={len(test_manifest)}")

    # Create datasets
    train_dataset = ECGDataset(train_manifest, processed_dir, augment=augment_train)
    val_dataset = ECGDataset(val_manifest, processed_dir, augment=False)
    test_dataset = ECGDataset(test_manifest, processed_dir, augment=False)

    # Optional: Create weighted sampler for imbalanced classes
    train_labels = [item['label'] for item in train_manifest]
    label_counts = np.bincount(train_labels)
    weights = 1.0 / (label_counts + 1e-6)
    sample_weights = weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    ) if len(val_manifest) > 0 else None

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    ) if len(test_manifest) > 0 else None

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

