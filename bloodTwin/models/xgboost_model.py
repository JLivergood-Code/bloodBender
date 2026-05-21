"""
BloodTwin XGBoost Model

Trained on statistical features extracted from CGM data.
Designed to be fused with the LSTM model at inference time.
"""

import numpy as np
import xgboost as xgb
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BloodTwinXGBoost:
    """
    One XGBRegressor per horizon step.
    
    Input:  statistical features (n_samples, 8)
            [min, max, mean, std, median, skewness, kurtosis, peak_to_peak]
    Output: predicted glucose values (n_samples, horizon)
    """

    def __init__(self, horizon: int, params: dict):
        self.horizon = horizon
        self.params = params
        # One model per timestep ahead
        self.models = [
            xgb.XGBRegressor(**params)
            for _ in range(horizon)
        ]
        self.is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ):
        """
        Args:
            X_train: (n_samples, n_features) statistical features
            y_train: (n_samples, horizon) target glucose values
            X_val:   optional validation features for early stopping
            y_val:   optional validation targets
        """
        for t in range(self.horizon):
            logger.info(f"Training XGBoost for horizon step {t+1}/{self.horizon}")

            eval_set = [(X_val, y_val[:, t])] if X_val is not None else None

            self.models[t].fit(
                X_train, y_train[:, t],
                eval_set=eval_set,
                verbose=False
            )

        self.is_fitted = True
        logger.info("XGBoost training complete")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: (n_samples, n_features)
        Returns:
            predictions: (n_samples, horizon)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting")

        return np.stack(
            [self.models[t].predict(X) for t in range(self.horizon)],
            axis=1
        )

    def save(self, dir_path: Path):
        """Save one .json file per horizon step."""
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        for t, model in enumerate(self.models):
            model.save_model(dir_path / f"xgb_horizon_{t}.json")

        logger.info(f"Saved XGBoost models to {dir_path}")

    @classmethod
    def load(cls, dir_path: Path, horizon: int, params: dict) -> "BloodTwinXGBoost":
        """Load a previously saved BloodTwinXGBoost."""
        instance = cls(horizon=horizon, params=params)

        for t in range(horizon):
            instance.models[t].load_model(
                Path(dir_path) / f"xgb_horizon_{t}.json"
            )

        instance.is_fitted = True
        logger.info(f"Loaded XGBoost models from {dir_path}")
        return instance