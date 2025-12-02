"""
ECG Training Module
Implements training loop with mixed precision, checkpointing, and metrics logging.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .utils import write_json

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer class for ECG classification with mixed precision support.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader],
        device: str,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 50,
        checkpoint_dir: Path = Path('artifacts/models'),
        use_amp: bool = True,
        grad_accum_steps: int = 1
    ):
        """
        Initialize Trainer.

        Args:
            model: PyTorch model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            device: Device string ('cuda' or 'cpu')
            learning_rate: Initial learning rate
            weight_decay: Weight decay for optimizer
            epochs: Number of training epochs
            checkpoint_dir: Directory to save checkpoints
            use_amp: Whether to use automatic mixed precision
            grad_accum_steps: Gradient accumulation steps
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.grad_accum_steps = grad_accum_steps

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=1e-6
        )

        # Mixed precision
        self.use_amp = use_amp and device == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None

        # Tracking
        self.best_val_f1 = 0.0
        self.history = []

        logger.info(f"Initialized Trainer on {device}, AMP={self.use_amp}")

    def train_step(self) -> Dict[str, float]:
        """
        Execute one training epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch_idx, (signals, labels) in enumerate(pbar):
            signals = signals.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Forward pass with optional mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(signals)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.grad_accum_steps

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(signals)
                loss = self.criterion(outputs, labels)
                loss = loss / self.grad_accum_steps
                loss.backward()

                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Track metrics
            total_loss += loss.item() * self.grad_accum_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total

        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }

    def val_step(self) -> Dict[str, float]:
        """
        Execute one validation epoch.

        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation", leave=False)
            for signals, labels in pbar:
                signals = signals.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if self.use_amp:
                    with autocast():
                        outputs = self.model(signals)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(signals)
                    loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total

        # Calculate F1 macro (simple approximation)
        # For full metrics, use eval.py functions
        from sklearn.metrics import f1_score
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1_macro': f1_macro
        }

    def fit(self) -> Dict[str, list]:
        """
        Train model for specified number of epochs.

        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {self.epochs} epochs")

        for epoch in range(self.epochs):
            start_time = time.time()

            # Train
            train_metrics = self.train_step()

            # Validate
            val_metrics = self.val_step()

            # Step scheduler
            self.scheduler.step()

            # Log metrics
            epoch_time = time.time() - start_time
            lr = self.optimizer.param_groups[0]['lr']

            log_str = f"Epoch {epoch+1}/{self.epochs} ({epoch_time:.1f}s) - "
            log_str += f"LR: {lr:.6f} - "
            log_str += f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%"

            if val_metrics:
                log_str += f" - Val Loss: {val_metrics['loss']:.4f}, "
                log_str += f"Acc: {val_metrics['accuracy']:.2f}%, "
                log_str += f"F1: {val_metrics['f1_macro']:.4f}"

            logger.info(log_str)

            # Save history
            epoch_record = {
                'epoch': epoch + 1,
                'train': train_metrics,
                'val': val_metrics,
                'lr': lr,
                'time': epoch_time
            }
            self.history.append(epoch_record)

            # Save checkpoints
            val_f1 = val_metrics.get('f1_macro', 0.0)

            # Save latest
            latest_path = self.checkpoint_dir / 'latest.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'metrics': val_metrics
            }, latest_path)

            # Save best
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                best_path = self.checkpoint_dir / 'best.pth'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'metrics': val_metrics
                }, best_path)
                logger.info(f"Saved new best model with F1={val_f1:.4f}")

        # Save training log
        log_path = self.checkpoint_dir.parent / 'training_log.json'
        write_json(log_path, {'history': self.history, 'best_val_f1': self.best_val_f1})
        logger.info(f"Saved training log to {log_path}")

        return self.history

