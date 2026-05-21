"""
BloodTwin Fusion Model

Combines LSTM and XGBoost predictions by averaging,
as described in the paper (Section 2.1.3, Fig. 3).
"""

import numpy as np
import torch
from pathlib import Path
import logging

from bloodTwin.models.lstm import BloodTwinLSTM
from bloodTwin.models.xgboost_model import BloodTwinXGBoost

logger = logging.getLogger(__name__)


class FusionPredictor:
    """
    Loads both trained models and fuses their predictions.
    
    LSTM receives:   (batch, lookback, n_features) sequence tensor
    XGBoost receives: (batch, 8) statistical features array
    Output:          (batch, horizon) averaged predictions
    """

    def __init__(
        self,
        lstm_model: BloodTwinLSTM,
        xgb_model: BloodTwinXGBoost,
        scaler           # the same scaler used during training
    ):
        self.lstm = lstm_model
        self.xgb = xgb_model
        self.scaler = scaler

        self.lstm.eval()

    def predict(
        self,
        sequence: torch.Tensor,
        stat_features: np.ndarray
    ) -> np.ndarray:
        """
        Args:
            sequence:      (batch, lookback, n_features) — LSTM input
            stat_features: (batch, 8) — XGBoost input
        Returns:
            fused: (batch, horizon) — averaged predictions in original scale
        """
        # LSTM prediction
        with torch.no_grad():
            lstm_pred = self.lstm(sequence).numpy()   # (batch, horizon)

        # XGBoost prediction
        xgb_pred = self.xgb.predict(stat_features)    # (batch, horizon)

        # Normalize both to same scale before averaging (paper Fig. 3)
        lstm_norm = self._normalize(lstm_pred)
        xgb_norm = self._normalize(xgb_pred)

        # Simple average — equation from Section 2.1.3
        fused_norm = (lstm_norm + xgb_norm) / 2

        # Inverse transform back to mg/dL
        fused = self.scaler.inverse_transform(fused_norm)

        return fused

    def _normalize(self, predictions: np.ndarray) -> np.ndarray:
        """Min-max normalize predictions before fusion."""
        min_val = predictions.min()
        max_val = predictions.max()
        if max_val == min_val:
            return predictions
        return (predictions - min_val) / (max_val - min_val)

    @classmethod
    def from_artifacts(
        cls,
        artifacts_dir: Path,
        lstm_checkpoint: str,
        horizon: int,
        xgb_params: dict,
        scaler
    ) -> "FusionPredictor":
        """
        Convenience loader — reconstructs both models from saved artifacts.
        Use this at inference time.
        """
        lstm_model = BloodTwinLSTM.load_from_checkpoint(lstm_checkpoint)

        xgb_model = BloodTwinXGBoost.load(
            dir_path=artifacts_dir / 'xgb_models',
            horizon=horizon,
            params=xgb_params
        )

        return cls(lstm_model, xgb_model, scaler)