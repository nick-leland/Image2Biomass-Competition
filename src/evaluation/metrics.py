"""
Evaluation metrics for multi-task biomass regression.
"""

import numpy as np
import torch
from typing import Dict
from src.config import TARGET_NAMES

# Competition weights for weighted R² score
KAGGLE_WEIGHTS = {
    'Dry_Total_g': 0.50,
    'GDM_g': 0.20,
    'Dry_Green_g': 0.10,
    'Dry_Dead_g': 0.10,
    'Dry_Clover_g': 0.10,
}


def compute_metrics(predictions: Dict[str, torch.Tensor],
                   targets: Dict[str, torch.Tensor],
                   denormalize_fn=None) -> Dict[str, float]:
    """
    Compute comprehensive metrics for multi-target regression.

    Args:
        predictions: Dict of prediction tensors [n_samples] for each target
        targets: Dict of target tensors [n_samples] for each target
        denormalize_fn: Optional function to denormalize values back to original scale

    Returns:
        metrics: Dict with per-target and overall metrics
    """
    metrics = {}

    # Convert tensors to numpy
    pred_dict = {name: pred.cpu().numpy() for name, pred in predictions.items()}
    true_dict = {name: true.cpu().numpy() for name, true in targets.items()}

    # Denormalize if function provided
    if denormalize_fn is not None:
        pred_dict = denormalize_fn(pred_dict)
        true_dict = denormalize_fn(true_dict)

    # Per-target metrics
    for target_name in TARGET_NAMES:
        pred = pred_dict[target_name]
        true = true_dict[target_name]

        # Mean Absolute Error
        mae = float(np.mean(np.abs(pred - true)))

        # Root Mean Squared Error
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))

        # R-squared
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 1e-8 else 0.0

        # Mean Absolute Percentage Error (avoid division by zero)
        mape = float(np.mean(np.abs((true - pred) / (np.abs(true) + 1e-8))) * 100)

        metrics[f'{target_name}_MAE'] = mae
        metrics[f'{target_name}_RMSE'] = rmse
        metrics[f'{target_name}_R2'] = r2
        metrics[f'{target_name}_MAPE'] = mape

    # Overall metrics (average across targets)
    metrics['overall_MAE'] = np.mean([metrics[f'{t}_MAE'] for t in TARGET_NAMES])
    metrics['overall_RMSE'] = np.mean([metrics[f'{t}_RMSE'] for t in TARGET_NAMES])
    metrics['overall_R2'] = np.mean([metrics[f'{t}_R2'] for t in TARGET_NAMES])
    metrics['overall_MAPE'] = np.mean([metrics[f'{t}_MAPE'] for t in TARGET_NAMES])

    # Kaggle weighted R² (competition metric)
    # Uses log-stabilizing transformation: log(1 + x)
    weighted_r2 = 0.0
    for target_name in TARGET_NAMES:
        pred = pred_dict[target_name]
        true = true_dict[target_name]

        # Log-stabilizing transformation
        pred_log = np.log1p(np.maximum(pred, 0))
        true_log = np.log1p(np.maximum(true, 0))

        # R² on log-transformed values
        ss_res = np.sum((true_log - pred_log) ** 2)
        ss_tot = np.sum((true_log - np.mean(true_log)) ** 2)
        r2_log = float(1 - (ss_res / ss_tot)) if ss_tot > 1e-8 else 0.0

        weighted_r2 += KAGGLE_WEIGHTS[target_name] * r2_log
        metrics[f'{target_name}_R2_log'] = r2_log

    metrics['weighted_R2'] = weighted_r2

    # Constraint violation metrics
    pred_total = pred_dict['Dry_Total_g']
    pred_sum = (pred_dict['Dry_Clover_g'] +
                pred_dict['Dry_Dead_g'] +
                pred_dict['Dry_Green_g'])

    constraint_violation = np.abs(pred_total - pred_sum)
    metrics['constraint_MAE'] = float(np.mean(constraint_violation))
    metrics['constraint_RMSE'] = float(np.sqrt(np.mean(constraint_violation ** 2)))
    metrics['constraint_max'] = float(np.max(constraint_violation))
    metrics['constraint_violation_rate_0.1'] = float(np.mean(constraint_violation > 0.1))
    metrics['constraint_violation_rate_1.0'] = float(np.mean(constraint_violation > 1.0))

    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    Print metrics in a formatted way.

    Args:
        metrics: Dict of metrics
        prefix: Prefix for printing (e.g., "Train" or "Val")
    """
    print(f"\n{prefix} Metrics:")
    print("=" * 70)

    # Overall metrics
    print(f"{'Overall MAE:':<30} {metrics['overall_MAE']:>8.4f}")
    print(f"{'Overall RMSE:':<30} {metrics['overall_RMSE']:>8.4f}")
    print(f"{'Overall R2:':<30} {metrics['overall_R2']:>8.4f}")
    print(f"{'Weighted R2 (Kaggle):':<30} {metrics['weighted_R2']:>8.4f}")
    print(f"{'Overall MAPE:':<30} {metrics['overall_MAPE']:>8.2f}%")

    print("\nPer-Target Metrics:")
    print("-" * 70)

    for target_name in TARGET_NAMES:
        mae = metrics[f'{target_name}_MAE']
        rmse = metrics[f'{target_name}_RMSE']
        r2 = metrics[f'{target_name}_R2']
        print(f"{target_name:<20} MAE: {mae:>7.4f}  RMSE: {rmse:>7.4f}  R2: {r2:>6.4f}")

    print("\nConstraint Violation:")
    print("-" * 70)
    print(f"{'MAE:':<30} {metrics['constraint_MAE']:>8.4f}")
    print(f"{'RMSE:':<30} {metrics['constraint_RMSE']:>8.4f}")
    print(f"{'Max:':<30} {metrics['constraint_max']:>8.4f}")
    print(f"{'Rate > 0.1g:':<30} {metrics['constraint_violation_rate_0.1']:>8.2%}")
    print(f"{'Rate > 1.0g:':<30} {metrics['constraint_violation_rate_1.0']:>8.2%}")
    print("=" * 70)


def compute_kaggle_score(predictions: Dict[str, torch.Tensor],
                        targets: Dict[str, torch.Tensor],
                        denormalize_fn=None) -> float:
    """
    Compute Kaggle competition score (weighted R² with log transform).

    Competition uses weighted R² with weights:
    - Dry_Total_g: 0.50
    - GDM_g: 0.20
    - Dry_Green_g, Dry_Dead_g, Dry_Clover_g: 0.10 each

    Args:
        predictions: Dict of prediction tensors
        targets: Dict of target tensors
        denormalize_fn: Optional function to denormalize values

    Returns:
        score: Weighted R² (higher is better)
    """
    metrics = compute_metrics(predictions, targets, denormalize_fn)
    return metrics['weighted_R2']
