"""
ECG Research Pipeline
A unified multi-dataset ECG preprocessing, training, and serving pipeline.
"""

__version__ = "0.1.0"
__author__ = "ECG Research Team"

from . import dataloaders, eval, model, preprocessing, saver, training, utils

__all__ = ['utils', 'preprocessing', 'dataloaders', 'model', 'training', 'eval', 'saver']

