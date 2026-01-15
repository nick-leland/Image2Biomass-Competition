"""
Generate Kaggle submission from depth fusion K-Fold ensemble models.

Usage:
    python scripts/generate_submission_depth.py
    python scripts/generate_submission_depth.py --checkpoint_dir experiments/checkpoints_depth_rgb_depth_fusion_combined_20260115_011025
"""

import sys
sys.path.insert(0, '/home/chaot/kaggle/Image2Biomass-Competition')

import argparse
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from glob import glob

from src.config import TEST_CSV, TEST_IMG_DIR, TRAIN_CSV, TRAIN_IMG_DIR, TARGET_NAMES, SUBMISSIONS_DIR
from src.utils.seed import set_seed
from src.data.dataset import BiomassDataset
from src.data.transforms import get_val_transforms, get_tta_transforms
from src.models.model_factory import create_model


def main():
    parser = argparse.ArgumentParser(description='Generate depth model submission')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='Checkpoint directory (auto-detects latest if not specified)')
    parser.add_argument('--use_tta', action='store_true', default=True,
                       help='Use test-time augmentation (default: True)')
    parser.add_argument('--no_tta', action='store_true',
                       help='Disable TTA')
    args = parser.parse_args()

    if args.no_tta:
        args.use_tta = False

    print("=" * 70)
    print("Image2Biomass - Depth Fusion Model Submission")
    print("=" * 70)

    # Find checkpoint directory
    if args.checkpoint_dir is None:
        # Auto-detect latest depth checkpoint
        checkpoint_dirs = sorted(glob('experiments/checkpoints_depth_*'))
        if not checkpoint_dirs:
            raise FileNotFoundError("No depth model checkpoints found")
        args.checkpoint_dir = checkpoint_dirs[-1]
        print(f"Auto-detected checkpoint: {args.checkpoint_dir}")

    checkpoint_dir = Path(args.checkpoint_dir)
    results_path = checkpoint_dir / 'kfold_results.json'

    if not results_path.exists():
        raise FileNotFoundError(f"Results not found: {results_path}")

    with open(results_path, 'r') as f:
        kfold_results = json.load(f)

    config = kfold_results['config']
    n_folds = kfold_results['n_folds']

    print(f"\nLoaded {n_folds}-fold results")
    print(f"Model type: {config['model_type']}")
    print(f"Backbone: {config['backbone']}")
    print(f"Depth model: {config['depth_model']}")
    print(f"Mean val loss: {kfold_results['mean_val_loss']:.4f}")
    print(f"Use TTA: {args.use_tta}")

    # Set seed
    set_seed(config['seed'])

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Get target stats from training data
    print("\nLoading training data for normalization stats...")
    train_df = pd.read_csv(TRAIN_CSV)

    temp_dir = Path('experiments/temp')
    temp_dir.mkdir(exist_ok=True, parents=True)
    temp_train_csv = temp_dir / 'train_for_stats_depth.csv'
    train_df.to_csv(temp_train_csv, index=False)

    val_transform = get_val_transforms(image_size=config['image_size'])
    train_dataset = BiomassDataset(
        csv_path=temp_train_csv,
        img_dir=TRAIN_IMG_DIR,
        transform=val_transform,
        is_test=False
    )
    target_stats = train_dataset.get_target_stats()

    # Create test dataset
    print(f"\nLoading test data from {TEST_CSV}...")
    test_dataset = BiomassDataset(
        csv_path=TEST_CSV,
        img_dir=TEST_IMG_DIR,
        transform=val_transform,
        is_test=True,
        target_stats=target_stats
    )
    print(f"Test samples: {len(test_dataset)}")

    # Load all fold models
    print(f"\n{'=' * 70}")
    print(f"Loading {n_folds} fold models...")
    print("=" * 70)

    fold_models = []
    for fold_idx in range(n_folds):
        fold_dir = checkpoint_dir / f'fold_{fold_idx}'
        checkpoint_path = fold_dir / 'best_model.pth'

        if not checkpoint_path.exists():
            print(f"Warning: Fold {fold_idx} checkpoint not found, skipping")
            continue

        print(f"\nLoading Fold {fold_idx + 1}...")
        model = create_model(config)
        model = model.to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print(f"  Val Loss: {checkpoint['best_val_loss']:.4f}")
        fold_models.append(model)

    print(f"\nLoaded {len(fold_models)} models")

    # Generate predictions
    print(f"\n{'=' * 70}")
    print("Generating predictions...")
    print("=" * 70)

    all_fold_predictions = []

    # Get TTA transforms if enabled
    if args.use_tta:
        tta_transforms = get_tta_transforms(
            image_size=config['image_size'],
            include_rotations=True
        )
        print(f"Using {len(tta_transforms)} TTA transforms")
    else:
        tta_transforms = [val_transform]
        print("TTA disabled")

    for fold_idx, model in enumerate(fold_models):
        print(f"\nFold {fold_idx + 1}/{len(fold_models)}...")
        fold_predictions = {name: [] for name in TARGET_NAMES}

        with torch.no_grad():
            for tta_idx, tta_transform in enumerate(tta_transforms):
                # Create dataset with TTA transform
                test_dataset_tta = BiomassDataset(
                    csv_path=TEST_CSV,
                    img_dir=TEST_IMG_DIR,
                    transform=tta_transform,
                    is_test=True,
                    target_stats=target_stats
                )

                test_loader = torch.utils.data.DataLoader(
                    test_dataset_tta,
                    batch_size=config['batch_size'],
                    shuffle=False,
                    num_workers=0,
                    collate_fn=BiomassDataset.collate_fn
                )

                tta_preds = {name: [] for name in TARGET_NAMES}

                for batch in test_loader:
                    images = batch['image'].to(device)
                    predictions = model(images)

                    batch_size = images.size(0)
                    for i in range(batch_size):
                        pred_dict = {name: predictions[name][i].item() for name in TARGET_NAMES}
                        pred_denorm = train_dataset.denormalize_targets(pred_dict)

                        for name in TARGET_NAMES:
                            tta_preds[name].append(pred_denorm[name])

                # Accumulate TTA predictions
                if tta_idx == 0:
                    for name in TARGET_NAMES:
                        fold_predictions[name] = tta_preds[name]
                else:
                    for name in TARGET_NAMES:
                        for i in range(len(tta_preds[name])):
                            fold_predictions[name][i] += tta_preds[name][i]

        # Average TTA predictions
        n_tta = len(tta_transforms)
        for name in TARGET_NAMES:
            fold_predictions[name] = [p / n_tta for p in fold_predictions[name]]

        all_fold_predictions.append(fold_predictions)

    # Average across folds
    print(f"\nAveraging predictions across {len(fold_models)} folds...")
    n_samples = len(all_fold_predictions[0][TARGET_NAMES[0]])
    ensemble_predictions = {name: [] for name in TARGET_NAMES}

    for sample_idx in range(n_samples):
        for name in TARGET_NAMES:
            fold_preds = [fp[name][sample_idx] for fp in all_fold_predictions]
            ensemble_predictions[name].append(np.mean(fold_preds))

    # Apply biological constraints
    print("\nApplying biological constraints...")
    for sample_idx in range(n_samples):
        # Clip negatives
        for name in TARGET_NAMES:
            ensemble_predictions[name][sample_idx] = max(0.0, ensemble_predictions[name][sample_idx])

        clover = ensemble_predictions['Dry_Clover_g'][sample_idx]
        dead = ensemble_predictions['Dry_Dead_g'][sample_idx]
        green = ensemble_predictions['Dry_Green_g'][sample_idx]
        gdm = ensemble_predictions['GDM_g'][sample_idx]
        total = ensemble_predictions['Dry_Total_g'][sample_idx]

        # Enforce GDM = Green + Clover
        gdm_calc = green + clover
        adjusted_gdm = (gdm + gdm_calc) / 2
        if gdm_calc > 0:
            scale = adjusted_gdm / gdm_calc
            ensemble_predictions['Dry_Green_g'][sample_idx] = green * scale
            ensemble_predictions['Dry_Clover_g'][sample_idx] = clover * scale
        ensemble_predictions['GDM_g'][sample_idx] = adjusted_gdm

        # Enforce Total = GDM + Dead
        total_calc = adjusted_gdm + dead
        adjusted_total = (total + total_calc) / 2
        if adjusted_total > adjusted_gdm:
            ensemble_predictions['Dry_Dead_g'][sample_idx] = adjusted_total - adjusted_gdm
        else:
            ensemble_predictions['Dry_Dead_g'][sample_idx] = 0.0
            adjusted_total = adjusted_gdm
        ensemble_predictions['Dry_Total_g'][sample_idx] = adjusted_total

    # Create submission
    print("\nCreating submission...")
    test_df = pd.read_csv(TEST_CSV)
    test_image_ids = test_df['sample_id'].str.split('__').str[0].unique()

    submission_rows = []
    for img_idx, image_id in enumerate(test_image_ids):
        for name in TARGET_NAMES:
            sample_id = f"{image_id}__{name}"
            prediction = ensemble_predictions[name][img_idx]
            submission_rows.append({'sample_id': sample_id, 'target': prediction})

    submission_df = pd.DataFrame(submission_rows)

    # Save
    SUBMISSIONS_DIR.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tta_suffix = '_tta' if args.use_tta else ''
    output_path = SUBMISSIONS_DIR / f'submission_depth_fusion_{n_folds}folds{tta_suffix}_{timestamp}.csv'
    submission_df.to_csv(output_path, index=False)

    print(f"\n{'=' * 70}")
    print("Submission Complete!")
    print(f"Saved to: {output_path}")
    print("=" * 70)

    print("\nPredictions summary:")
    for name in TARGET_NAMES:
        values = ensemble_predictions[name]
        print(f"  {name:<15} mean: {np.mean(values):>8.2f}  min: {np.min(values):>8.2f}  max: {np.max(values):>8.2f}")

    print("\nConstraint check:")
    for sample_idx in range(n_samples):
        gdm = ensemble_predictions['GDM_g'][sample_idx]
        gdm_calc = ensemble_predictions['Dry_Green_g'][sample_idx] + ensemble_predictions['Dry_Clover_g'][sample_idx]
        total = ensemble_predictions['Dry_Total_g'][sample_idx]
        total_calc = ensemble_predictions['GDM_g'][sample_idx] + ensemble_predictions['Dry_Dead_g'][sample_idx]
        print(f"  Sample {sample_idx}: GDM diff={abs(gdm-gdm_calc):.4f}, Total diff={abs(total-total_calc):.4f}")

    return output_path


if __name__ == '__main__':
    main()
