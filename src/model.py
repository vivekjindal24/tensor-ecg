"""
ECG Model Module
Defines 1D CNN architecture for ECG classification with residual blocks.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ResidualBlock1D(nn.Module):
    """
    1D Residual block with batch normalization and skip connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.1
    ):
        """
        Initialize residual block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            dropout: Dropout probability
        """
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=1, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)

        # Skip connection with projection if dimensions change
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block."""
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.activation(out)

        return out


class ECGResNet1D(nn.Module):
    """
    1D ResNet for ECG classification.
    Suitable for single-lead or multi-lead flattened ECG signals.
    """

    def __init__(
        self,
        input_channels: int = 1,
        n_classes: int = 5,
        base_channels: int = 64,
        dropout: float = 0.2
    ):
        """
        Initialize ECG ResNet model.

        Args:
            input_channels: Number of input channels (1 for single-lead)
            n_classes: Number of output classes
            base_channels: Base number of channels (will be scaled in deeper layers)
            dropout: Dropout probability
        """
        super().__init__()

        self.input_channels = input_channels
        self.n_classes = n_classes

        # Initial convolution
        self.conv1 = nn.Conv1d(input_channels, base_channels, kernel_size=15,
                               stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.activation = nn.GELU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = self._make_layer(base_channels, base_channels, 2, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(base_channels, base_channels*2, 2, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(base_channels*2, base_channels*4, 2, stride=2, dropout=dropout)
        self.layer4 = self._make_layer(base_channels*4, base_channels*8, 2, stride=2, dropout=dropout)

        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base_channels*8, n_classes)

        self._initialize_weights()

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
        dropout: float
    ) -> nn.Sequential:
        """Create a layer with multiple residual blocks."""
        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, stride=stride, dropout=dropout))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels, stride=1, dropout=dropout))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, samples)

        Returns:
            Logits of shape (batch, n_classes)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: str = 'cpu'
) -> nn.Module:
    """
    Load model from checkpoint.

    Args:
        model: Model instance
        checkpoint_path: Path to checkpoint file
        device: Device to load model to

    Returns:
        Model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    return model


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    checkpoint_path: Path
) -> None:
    """
    Save model checkpoint.

    Args:
        model: Model instance
        optimizer: Optimizer instance
        epoch: Current epoch
        metrics: Dictionary of metrics
        checkpoint_path: Path to save checkpoint
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")

