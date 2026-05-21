"""
BloodTwin Evaluation Script

Runs fusion prediction over validation or test set and reports
RMSE, MAE, MAPE, and R² — matching the metrics from the paper (Section 2.2).
"""

import numpy as np
import yaml
import pickle
import torch
import logging
import argparse
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from bloodTwin.models.fusion import FusionPredictor
from bloodTwin.models.xgboost_model import BloodTwinXGBoost
from bloodTwin.models.lstm import BloodTwinLSTM
from bloodTwin.data.dataset import create_dataloaders, extract_stat_features_from_loader, extract_statistical_features
from bloodTwin import ARTIFACTS_DIR, CONFIGS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Metric functions (equations 9–12 from the paper) ──────────────────────────

def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return np.sqrt(mean_squared_error(actual, predicted))

def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return mean_absolute_error(actual, predicted)

def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    # Avoid division by zero
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

def r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    return r2_score(actual, predicted)

def compute_all_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """
    Compute all four metrics the paper reports.
    Both arrays shape: (n_samples, horizon)
    Returns a dict of metrics averaged across all samples.
    """
    return {
        'RMSE': rmse(actual.flatten(), predicted.flatten()),
        'MAE':  mae(actual.flatten(), predicted.flatten()),
        'MAPE': mape(actual.flatten(), predicted.flatten()),
        'R2':   r2(actual.flatten(), predicted.flatten())
    }


# ── Per-model evaluation ───────────────────────────────────────────────────────

def evaluate_lstm(
    model: BloodTwinLSTM,
    dataloader,
    scaler
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run LSTM over a full dataloader.
    Returns (actuals, predictions) both in original mg/dL scale.
    """
    model.eval()
    all_preds, all_actuals = [], []

    with torch.no_grad():
        for sequences, targets in dataloader:
            preds = model(sequences).numpy()       # (batch, horizon)
            all_preds.append(preds)
            all_actuals.append(targets.numpy())

    preds   = np.concatenate(all_preds,   axis=0)  # (n_samples, horizon)
    actuals = np.concatenate(all_actuals, axis=0)

    # Inverse transform back to mg/dL
    preds   = scaler.inverse_transform(preds)
    actuals = scaler.inverse_transform(actuals)

    return actuals, preds


def evaluate_xgboost(
    xgb_model: BloodTwinXGBoost,
    dataloader,
    scaler,
    horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run XGBoost over a full dataloader.
    Returns (actuals, predictions) both in original mg/dL scale.
    """
    X, y_scaled = extract_stat_features_from_loader(dataloader, horizon)

    preds   = xgb_model.predict(X)                 # (n_samples, horizon)
    actuals = scaler.inverse_transform(y_scaled)
    preds   = scaler.inverse_transform(preds)

    return actuals, preds


def evaluate_fusion(
    fusion: FusionPredictor,
    dataloader,
    horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run fusion predictor over a full dataloader.
    Returns (actuals, predictions) both in original mg/dL scale.
    """
    all_preds, all_actuals = [], []

    for sequences, targets in dataloader:
        # Stat features from glucose channel (index 0)
        glucose = sequences[:, :, 0].numpy()
        stat_features = np.array([
            extract_statistical_features(glucose[i])
            for i in range(glucose.shape[0])
        ])

        preds = fusion.predict(sequences, stat_features)  # (batch, horizon)
        all_preds.append(preds)
        all_actuals.append(targets.numpy())

    preds   = np.concatenate(all_preds,   axis=0)
    actuals = np.concatenate(all_actuals, axis=0)

    # Fusion predict already inverse transforms, so no scaler call needed here
    return actuals, preds


# ── Results formatting ─────────────────────────────────────────────────────────

def print_results_table(results: dict):
    """
    Prints a comparison table matching the paper's Tables 5–8 format.
    
    results: {
        'LSTM':                         {'RMSE': ..., 'MAE': ..., ...},
        'LSTM with Additional features': {'RMSE': ..., 'MAE': ..., ...},
        'Fusion':                        {'RMSE': ..., 'MAE': ..., ...}
    }
    """
    col_width = 12
    methods = list(results.keys())
    metrics = ['RMSE', 'MAE', 'MAPE', 'R2']

    # Header
    print("\n" + "─" * 70)
    print(f"{'Method':<35}" + "".join(f"{m:>{col_width}}" for m in metrics))
    print("─" * 70)

    # Rows
    for method, scores in results.items():
        row = f"{method:<35}"
        for m in metrics:
            row += f"{scores[m]:>{col_width}.4f}"
        print(row)

    print("─" * 70 + "\n")


def save_results(results: dict, horizon: int, output_path: Path):
    """Save results to YAML for later comparison."""
    import yaml
    output = {
        'horizon_minutes': horizon,
        'results': {
            method: {k: float(v) for k, v in scores.items()}
            for method, scores in results.items()
        }
    }
    with open(output_path, 'w') as f:
        yaml.dump(output, f, default_flow_style=False)
    logger.info(f"Saved results to {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def evaluate(config_path: Path, split: str = 'val'):
    """
    Args:
        config_path: path to lstm.yaml
        split:       'val' or 'test'
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    artifacts_dir = ARTIFACTS_DIR / config['model']['name']
    horizon       = config['data']['horizon']
    data_dir      = Path(config['data']['train_dir'])

    # Load scaler
    with open(artifacts_dir / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Create dataloaders — we only need the one matching `split`
    logger.info(f"Loading {split} data...")
    train_loader, val_loader, test_loader, _ = create_dataloaders(
        data_dir=data_dir,
        features=config['data']['features'],
        target=config['data']['target'],
        lookback=config['data']['lookback'],
        horizon=horizon,
        stride=config['data']['stride'],
        batch_size=config['training']['batch_size'],
        scaler_path=artifacts_dir / 'scaler.pkl'
    )
    loader = val_loader if split == 'val' else test_loader

    # Load LSTM
    logger.info("Loading LSTM...")
    lstm_model = BloodTwinLSTM.load_from_checkpoint(
        str(artifacts_dir / 'checkpoints' / 'best_model.ckpt')
    )
    lstm_model.eval()

    # Load XGBoost
    logger.info("Loading XGBoost...")
    xgb_params = config['model'].get('xgboost', {
        'n_estimators': 400,
        'max_depth': 10,
        'learning_rate': 0.01,
        'random_state': 42
    })
    xgb_model = BloodTwinXGBoost.load(
        dir_path=artifacts_dir / 'xgb_models',
        horizon=horizon,
        params=xgb_params
    )

    # Build fusion predictor
    fusion = FusionPredictor(lstm_model, xgb_model, scaler)

    # ── Run evaluation for each model ────────────────────────────────────────
    logger.info("Evaluating LSTM alone...")
    actuals, lstm_preds = evaluate_lstm(lstm_model, loader, scaler)
    lstm_metrics = compute_all_metrics(actuals, lstm_preds)

    logger.info("Evaluating XGBoost alone...")
    _, xgb_preds = evaluate_xgboost(xgb_model, loader, scaler, horizon)
    xgb_metrics = compute_all_metrics(actuals, xgb_preds)

    logger.info("Evaluating Fusion...")
    _, fusion_preds = evaluate_fusion(fusion, loader, horizon)
    fusion_metrics = compute_all_metrics(actuals, fusion_preds)

    # ── Print and save ────────────────────────────────────────────────────────
    results = {
        'LSTM':                          lstm_metrics,
        'XGBoost':                       xgb_metrics,
        'LSTM + XGBoost (Fusion)':       fusion_metrics
    }

    horizon_minutes = horizon * 15  # assuming 15-min CGM intervals
    print(f"\nResults on {split} set — Prediction Horizon: {horizon_minutes} min")
    print_results_table(results)

    save_results(
        results,
        horizon_minutes,
        artifacts_dir / f'eval_{split}_results.yaml'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate BloodTwin models")
    parser.add_argument(
        '--config',
        type=Path,
        default=CONFIGS_DIR / 'lstm.yaml'
    )
    parser.add_argument(
        '--split',
        choices=['val', 'test'],
        default='val',
        help="Which data split to evaluate on"
    )
    args = parser.parse_args()
    evaluate(args.config, args.split)