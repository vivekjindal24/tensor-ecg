"""
ECG Evaluation Module
Implements evaluation metrics and visualization for trained models.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from tqdm import tqdm

from .utils import safe_makedirs, write_json

logger = logging.getLogger(__name__)


def predict_on_split(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    use_amp: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate predictions and probabilities for a data split.

    Args:
        model: Trained model
        dataloader: DataLoader for the split
        device: Device string
        use_amp: Whether to use automatic mixed precision

    Returns:
        Tuple of (predictions, probabilities, true_labels)
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for signals, labels in tqdm(dataloader, desc="Predicting"):
            signals = signals.to(device, non_blocking=True)

            if use_amp and device == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = model(signals)
            else:
                outputs = model(signals)

            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    return (
        np.array(all_preds),
        np.array(all_probs),
        np.array(all_labels)
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Optional[List[str]] = None
) -> Dict[str, any]:
    """
    Compute classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: List of label names

    Returns:
        Dictionary of metrics
    """
    if label_names is None:
        label_names = [f"Class_{i}" for i in range(max(y_true.max(), y_pred.max()) + 1)]

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Macro and micro averages
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Per-class results
    per_class = []
    for i, label in enumerate(label_names):
        per_class.append({
            'label': label,
            'precision': float(precision[i]) if i < len(precision) else 0.0,
            'recall': float(recall[i]) if i < len(recall) else 0.0,
            'f1': float(f1[i]) if i < len(f1) else 0.0,
            'support': int(support[i]) if i < len(support) else 0
        })

    metrics = {
        'confusion_matrix': cm.tolist(),
        'per_class': per_class,
        'f1_macro': float(f1_macro),
        'f1_micro': float(f1_micro),
        'f1_weighted': float(f1_weighted),
        'accuracy': float((y_true == y_pred).mean())
    }

    # Classification report
    report = classification_report(
        y_true, y_pred, target_names=label_names, zero_division=0, output_dict=True
    )
    metrics['classification_report'] = report

    return metrics


def plot_class_f1(
    metrics: Dict[str, any],
    output_path: Path,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot per-class F1 scores as bar chart.

    Args:
        metrics: Metrics dictionary from compute_metrics
        output_path: Path to save figure
        figsize: Figure size tuple
    """
    per_class = metrics['per_class']
    labels = [item['label'] for item in per_class]
    f1_scores = [item['f1'] for item in per_class]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(range(len(labels)), f1_scores, color='steelblue', alpha=0.8)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Per-Class F1 Scores', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    safe_makedirs(output_path.parent)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved F1 bar plot to {output_path}")


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    label_names: List[str],
    output_dir: Path,
    split_name: str = 'test',
    use_amp: bool = True
) -> Dict[str, any]:
    """
    Complete evaluation pipeline: predict, compute metrics, save results and plots.

    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device string
        label_names: List of label names
        output_dir: Directory to save outputs
        split_name: Name of split being evaluated
        use_amp: Whether to use mixed precision

    Returns:
        Metrics dictionary
    """
    logger.info(f"Evaluating model on {split_name} split")

    # Get predictions
    preds, probs, labels = predict_on_split(model, dataloader, device, use_amp)

    # Compute metrics
    metrics = compute_metrics(labels, preds, label_names)

    # Add split name
    metrics['split'] = split_name

    # Save metrics
    metrics_path = output_dir / f'{split_name}_metrics.json'
    write_json(metrics_path, metrics)
    logger.info(f"Saved metrics to {metrics_path}")

    # Plot F1 scores
    figures_dir = output_dir.parent.parent / 'figures'
    plot_path = figures_dir / f'{split_name}_f1_scores.png'
    plot_class_f1(metrics, plot_path)

    # Log summary
    logger.info(f"{split_name.capitalize()} Results:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  F1 Macro: {metrics['f1_macro']:.4f}")
    logger.info(f"  F1 Micro: {metrics['f1_micro']:.4f}")
    logger.info(f"  F1 Weighted: {metrics['f1_weighted']:.4f}")

    return metrics

