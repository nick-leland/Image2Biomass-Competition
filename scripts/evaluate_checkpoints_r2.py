"""
Evaluate existing checkpoints with weighted R² metric (Kaggle competition metric).

Usage:
    python scripts/evaluate_checkpoints_r2.py
"""

import sys
sys.path.insert(0, '/home/chaot/kaggle/Image2Biomass-Competition')

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.evaluation.metrics import compute_metrics, KAGGLE_WEIGHTS
from src.config import TARGET_NAMES


def evaluate_checkpoint_dir(checkpoint_dir: Path, model_type: str = 'auto'):
    """
    Evaluate all folds in a checkpoint directory.

    Returns dict with per-fold and mean weighted R² scores.
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Check if kfold_results.json exists
    results_file = checkpoint_dir / 'kfold_results.json'
    if not results_file.exists():
        print(f"  No kfold_results.json found in {checkpoint_dir}")
        return None

    with open(results_file) as f:
        results = json.load(f)

    # Check if weighted_R2 already computed
    if 'mean_weighted_r2' in results:
        return {
            'checkpoint_dir': str(checkpoint_dir),
            'model_type': results.get('backbone', 'unknown'),
            'use_depth': results.get('use_depth', False),
            'mean_val_loss': results['mean_val_loss'],
            'std_val_loss': results['std_val_loss'],
            'mean_weighted_r2': results['mean_weighted_r2'],
            'std_weighted_r2': results['std_weighted_r2'],
            'fold_r2s': [r.get('best_weighted_r2', 0.0) for r in results['fold_results']],
            'already_computed': True,
        }

    # Need to load models and evaluate - for now just return val loss info
    return {
        'checkpoint_dir': str(checkpoint_dir),
        'model_type': results.get('backbone', results.get('config', {}).get('backbone', 'unknown')),
        'use_depth': results.get('use_depth', results.get('config', {}).get('use_depth', False)),
        'mean_val_loss': results.get('mean_val_loss', 0.0),
        'std_val_loss': results.get('std_val_loss', 0.0),
        'mean_weighted_r2': None,  # Not computed
        'std_weighted_r2': None,
        'fold_losses': [r.get('best_val_loss', 0.0) for r in results.get('fold_results', [])],
        'already_computed': False,
    }


def main():
    print("=" * 70)
    print("Evaluating Checkpoints with Weighted R² Metric")
    print("=" * 70)
    print(f"\nKaggle weights: {KAGGLE_WEIGHTS}")

    # Key checkpoint directories to evaluate
    checkpoints = {
        'V4 (EfficientNetV2 MSE)': 'experiments/checkpoints_kfold_mse',
        'V5 (Depth Fusion)': 'experiments/checkpoints_depth_rgb_depth_fusion_combined_20260115_011025',
        'V7 (DINOv2)': 'experiments/checkpoints_dinov2_base_20260116_100625',
        'V8 (DINOv2 + Depth)': 'experiments/checkpoints_dinov2_base_depth_20260116_140724',
        'SigLIP (for comparison)': 'experiments/checkpoints_siglip_base_20260116_094648',
    }

    # Known LB scores
    lb_scores = {
        'V4 (EfficientNetV2 MSE)': 0.50,
        'V5 (Depth Fusion)': 0.57,
        'V7 (DINOv2)': 0.58,
        'V8 (DINOv2 + Depth)': None,  # Pending
    }

    results_summary = []

    print("\n" + "-" * 70)

    for name, ckpt_path in checkpoints.items():
        full_path = Path('/home/chaot/kaggle/Image2Biomass-Competition') / ckpt_path

        if not full_path.exists():
            print(f"\n{name}: NOT FOUND")
            continue

        print(f"\n{name}:")
        print(f"  Path: {ckpt_path}")

        result = evaluate_checkpoint_dir(full_path)

        if result is None:
            continue

        result['name'] = name
        result['lb_score'] = lb_scores.get(name)
        results_summary.append(result)

        print(f"  Model: {result['model_type']}, Depth: {result['use_depth']}")
        print(f"  Val Loss: {result['mean_val_loss']:.4f} +/- {result['std_val_loss']:.4f}")

        if result['mean_weighted_r2'] is not None:
            print(f"  Weighted R² (CV): {result['mean_weighted_r2']:.4f} +/- {result['std_weighted_r2']:.4f}")
        else:
            print(f"  Weighted R² (CV): Not computed (needs model reload)")

        if result['lb_score'] is not None:
            print(f"  LB Score: {result['lb_score']:.2f}")
        else:
            print(f"  LB Score: Pending")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<30} {'Val Loss':>12} {'CV R²':>12} {'LB R²':>10}")
    print("-" * 70)

    for r in results_summary:
        val_loss = f"{r['mean_val_loss']:.4f}" if r['mean_val_loss'] else "N/A"
        cv_r2 = f"{r['mean_weighted_r2']:.4f}" if r['mean_weighted_r2'] else "N/A"
        lb_r2 = f"{r['lb_score']:.2f}" if r['lb_score'] else "Pending"
        print(f"{r['name']:<30} {val_loss:>12} {cv_r2:>12} {lb_r2:>10}")

    print("-" * 70)
    print("\nNote: CV R² shows 'N/A' for models trained before weighted R² was added.")
    print("To compute CV R², we need to reload models and run validation.")


if __name__ == '__main__':
    main()
