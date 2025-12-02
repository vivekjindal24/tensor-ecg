"""
ECG Saver Module
Utilities for saving models, results, and artifacts.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from .utils import safe_makedirs, write_json

logger = logging.getLogger(__name__)


def save_model_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    save_path: Path,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> None:
    """
    Save complete model checkpoint with optimizer and scheduler state.

    Args:
        model: Model instance
        optimizer: Optimizer instance
        epoch: Current epoch number
        metrics: Dictionary of metrics
        save_path: Path to save checkpoint
        scheduler: Optional learning rate scheduler
    """
    safe_makedirs(save_path.parent)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(checkpoint, save_path)
    logger.info(f"Saved checkpoint to {save_path}")


def save_predictions(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
    save_path: Path,
    label_names: Optional[list] = None
) -> None:
    """
    Save predictions, probabilities, and labels to compressed file.

    Args:
        predictions: Predicted class indices
        probabilities: Class probabilities
        labels: True labels
        save_path: Path to save file
        label_names: Optional list of label names
    """
    safe_makedirs(save_path.parent)

    save_dict = {
        'predictions': predictions,
        'probabilities': probabilities,
        'labels': labels
    }

    if label_names is not None:
        save_dict['label_names'] = np.array(label_names, dtype=object)

    np.savez_compressed(save_path, **save_dict)
    logger.info(f"Saved predictions to {save_path}")


def save_training_history(
    history: list,
    save_path: Path,
    additional_info: Optional[Dict] = None
) -> None:
    """
    Save training history to JSON file.

    Args:
        history: List of epoch dictionaries
        save_path: Path to save file
        additional_info: Optional additional metadata
    """
    output = {'history': history}

    if additional_info is not None:
        output.update(additional_info)

    write_json(save_path, output)
    logger.info(f"Saved training history to {save_path}")


def export_model_onnx(
    model: nn.Module,
    save_path: Path,
    input_shape: tuple = (1, 1, 5000),
    device: str = 'cpu'
) -> None:
    """
    Export model to ONNX format for deployment.

    Args:
        model: Trained model
        save_path: Path to save ONNX file
        input_shape: Example input shape
        device: Device to use for export
    """
    safe_makedirs(save_path.parent)

    model.eval()
    model = model.to(device)

    dummy_input = torch.randn(*input_shape).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    logger.info(f"Exported model to ONNX: {save_path}")


def save_evaluation_report(
    metrics: Dict,
    save_path: Path
) -> None:
    """
    Save evaluation metrics report to JSON.

    Args:
        metrics: Dictionary of evaluation metrics
        save_path: Path to save file
    """
    write_json(save_path, metrics)
    logger.info(f"Saved evaluation report to {save_path}")

