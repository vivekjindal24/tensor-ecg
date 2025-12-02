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

