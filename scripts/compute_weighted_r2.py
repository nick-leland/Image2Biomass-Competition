"""
Compute weighted R² for existing checkpoints by reloading models.

Usage:
    python scripts/compute_weighted_r2.py --checkpoint experiments/checkpoints_dinov2_base_20260116_100625
"""

import sys
sys.path.insert(0, '/home/chaot/kaggle/Image2Biomass-Competition')

import argparse
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.data.dataset import BiomassDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.data.splitter import get_group_kfold_splits
from src.evaluation.metrics import compute_metrics, KAGGLE_WEIGHTS, TARGET_NAMES
from src.config import DATA_DIR


def load_model_from_checkpoint(checkpoint_path: Path, config: dict, device: str = 'cuda'):
    """Load model based on config."""
    backbone = config.get('backbone', config.get('backbone_model', 'tf_efficientnetv2_m'))
    use_depth = config.get('use_depth', False)

    # Determine model type
    if 'dinov2' in backbone.lower() or 'siglip' in backbone.lower():
        if use_depth:
            from scripts.train_foundation_models import FoundationModelWithDepth
            model = FoundationModelWithDepth(
                backbone_name=config.get('backbone_model', 'vit_base_patch14_dinov2'),
                num_features=config.get('features', 768),
                dropout=config.get('dropout', 0.3),
            )
        else:
            from scripts.train_foundation_models import FoundationModelRegressor
            model = FoundationModelRegressor(
                backbone_name=config.get('backbone_model', 'vit_base_patch14_dinov2'),
                num_features=config.get('features', 768),
                dropout=config.get('dropout', 0.3),
            )
    elif 'depth' in str(checkpoint_path).lower() and 'dinov2' not in backbone.lower():
        # V5-style depth fusion model using model_factory
        from src.models.model_factory import create_model
        model = create_model(config)
    else:
        # Standard backbone model (EfficientNet, ResNet, etc.)
        from src.models.resnet_baseline import MultiTaskResNet
        model = MultiTaskResNet(
            backbone=backbone,
            pretrained=False,
            dropout=config.get('dropout', 0.5),
            head_hidden_dim=config.get('head_hidden_dim', 512),
        )

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Handle depth_estimator keys (they may not be saved)
    model_state = model.state_dict()
    for key in model_state.keys():
        if key in state_dict:
            model_state[key] = state_dict[key]

    model.load_state_dict(model_state, strict=False)
    model.to(device)
    model.eval()

    return model, checkpoint.get('best_val_loss', 0.0)


def evaluate_fold(model, val_loader, denormalize_fn, device='cuda'):
    """Evaluate model on validation set."""
    model.eval()
    all_predictions = {name: [] for name in TARGET_NAMES}
    all_targets = {name: [] for name in TARGET_NAMES}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            images = batch['image'].to(device)
            targets = {k: v.to(device) for k, v in batch['targets'].items()}

            # Handle depth models
            if hasattr(model, 'depth_estimator') or 'depth' in str(type(model)).lower():
                predictions = model(images)
            else:
                predictions = model(images)

            for name in TARGET_NAMES:
                all_predictions[name].append(predictions[name])
                all_targets[name].append(targets[name])

    # Concatenate
    all_predictions = {name: torch.cat(preds) for name, preds in all_predictions.items()}
    all_targets = {name: torch.cat(tgts) for name, tgts in all_targets.items()}

    # Compute metrics
    metrics = compute_metrics(all_predictions, all_targets, denormalize_fn)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Checkpoint directory')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint)
    device = args.device if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print(f"Computing Weighted R² for: {checkpoint_dir.name}")
    print("=" * 70)

    # Load config
    config_file = checkpoint_dir / 'config.json'
    if not config_file.exists():
        # Try kfold_results.json
        results_file = checkpoint_dir / 'kfold_results.json'
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)
            config = results.get('config', {})
        else:
            print("No config found!")
            return
    else:
        with open(config_file) as f:
            config = json.load(f)

    print(f"\nConfig: {json.dumps(config, indent=2)[:500]}...")

    # Load data
    train_csv = DATA_DIR / 'train.csv'
    img_dir = Path(config.get('img_dir', DATA_DIR / 'train'))

    df = pd.read_csv(train_csv)
    image_size = config.get('image_size', 384)

    # Convert to wide format for splitting
    df['image_id'] = df['sample_id'].str.split('__').str[0]
    df_wide = df.pivot_table(
        index=['image_id', 'image_path', 'Sampling_Date', 'State', 'Species',
               'Pre_GSHH_NDVI', 'Height_Ave_cm'],
        columns='target_name',
        values='target',
        aggfunc='first'
    ).reset_index()

    # Create splits (same as training)
    n_folds = config.get('n_folds', 5)

    val_transform = get_val_transforms(image_size=image_size)

    fold_results = []

    for fold_idx, (train_df_wide, val_df_wide) in get_group_kfold_splits(df_wide, n_folds=n_folds):
        fold_dir = checkpoint_dir / f'fold_{fold_idx}'
        best_model_path = fold_dir / 'best_model.pth'

        if not best_model_path.exists():
            print(f"\nFold {fold_idx}: best_model.pth not found")
            continue

        print(f"\nFold {fold_idx + 1}/{n_folds}...")

        # val_df_wide is already in wide format

        # Save temp CSV for BiomassDataset (expects csv_path)
        # Convert wide format back to long format for dataset
        val_long = val_df_wide.melt(
            id_vars=['image_id', 'image_path', 'Sampling_Date', 'State', 'Species',
                     'Pre_GSHH_NDVI', 'Height_Ave_cm'],
            value_vars=TARGET_NAMES,
            var_name='target_name',
            value_name='target'
        )
        val_long['sample_id'] = val_long['image_id'] + '__' + val_long['target_name']

        temp_csv = Path('/tmp/val_fold.csv')
        val_long.to_csv(temp_csv, index=False)

        val_dataset = BiomassDataset(
            csv_path=temp_csv,
            img_dir=img_dir,
            transform=val_transform,
            is_test=False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.get('batch_size', 4),
            shuffle=False,
            num_workers=0,
        )

        # Load model
        try:
            model, saved_val_loss = load_model_from_checkpoint(best_model_path, config, device)
        except Exception as e:
            print(f"  Error loading model: {e}")
            continue

        # Evaluate
        metrics = evaluate_fold(model, val_loader, val_dataset.denormalize_targets, device)

        weighted_r2 = metrics['weighted_R2']
        overall_r2 = metrics['overall_R2']
        overall_mae = metrics['overall_MAE']

        print(f"  Saved Val Loss: {saved_val_loss:.4f}")
        print(f"  Weighted R² (Kaggle): {weighted_r2:.4f}")
        print(f"  Overall R² (unweighted): {overall_r2:.4f}")
        print(f"  Overall MAE: {overall_mae:.4f}")

        fold_results.append({
            'fold': fold_idx,
            'saved_val_loss': saved_val_loss,
            'weighted_r2': weighted_r2,
            'overall_r2': overall_r2,
            'overall_mae': overall_mae,
        })

        # Clear GPU
        del model
        torch.cuda.empty_cache()

    # Summary
    if fold_results:
        weighted_r2s = [r['weighted_r2'] for r in fold_results]
        mean_r2 = np.mean(weighted_r2s)
        std_r2 = np.std(weighted_r2s)

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"\nWeighted R² by fold:")
        for r in fold_results:
            print(f"  Fold {r['fold'] + 1}: {r['weighted_r2']:.4f}")
        print(f"\nMean Weighted R² (CV): {mean_r2:.4f} +/- {std_r2:.4f}")
        print("=" * 70)


if __name__ == '__main__':
    main()
