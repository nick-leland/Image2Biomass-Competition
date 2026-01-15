"""
Generate Kaggle submission from K-Fold ensemble models.

This script loads all fold models and averages their predictions for more
robust results.

Usage:
    python scripts/generate_submission_kfold.py
    python scripts/generate_submission_kfold.py --checkpoint_dir experiments/checkpoints_kfold
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

from src.config import TEST_CSV, TEST_IMG_DIR, TRAIN_CSV, TRAIN_IMG_DIR, TARGET_NAMES, SUBMISSIONS_DIR
from src.utils.seed import set_seed, worker_init_fn
from src.data.dataset import BiomassDataset
from src.data.transforms import get_val_transforms
from src.models.model_factory import create_model


def main():
    parser = argparse.ArgumentParser(description='Generate K-Fold ensemble submission')
    parser.add_argument('--checkpoint_dir', type=str,
                       default='experiments/checkpoints_kfold',
                       help='Base directory containing fold checkpoints')
    parser.add_argument('--use_tta', action='store_true',
                       help='Use test-time augmentation')
    args = parser.parse_args()

    print("=" * 70)
    print("Image2Biomass - K-Fold Ensemble Submission Generation")
    print("=" * 70)

    # Load fold results
    checkpoint_dir = Path(args.checkpoint_dir)
    results_path = checkpoint_dir / 'kfold_results.json'

    if not results_path.exists():
        raise FileNotFoundError(f"K-fold results not found: {results_path}")

    with open(results_path, 'r') as f:
        kfold_results = json.load(f)

    config = kfold_results['config']
    n_folds = kfold_results['n_folds']

    print(f"\nLoaded K-fold results: {n_folds} folds")
    print(f"Mean validation loss: {kfold_results['mean_val_loss']:.4f} ± {kfold_results['std_val_loss']:.4f}")
    print(f"Model: {config['backbone']}")
    print(f"Image size: {config['image_size']}")
    print(f"Use TTA: {args.use_tta}")

    # Set seed
    set_seed(config['seed'])

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Get target statistics from training data
    print(f"\nLoading training data to get normalization statistics...")
    df = pd.read_csv(TRAIN_CSV)
    df['image_id'] = df['sample_id'].str.split('__').str[0]

    # Create temporary training dataset for stats
    temp_dir = Path('experiments/temp')
    temp_dir.mkdir(exist_ok=True, parents=True)
    temp_train_csv = temp_dir / 'train_for_stats.csv'
    df.to_csv(temp_train_csv, index=False)

    val_transform = get_val_transforms(image_size=config['image_size'])
    train_dataset = BiomassDataset(
        csv_path=temp_train_csv,
        img_dir=TRAIN_IMG_DIR,
        transform=val_transform,
        is_test=False
    )
    target_stats = train_dataset.get_target_stats()

    print("\nTarget normalization statistics:")
    for target_name in TARGET_NAMES:
        stats = target_stats[target_name]
        print(f"  {target_name:<20} mean: {stats['mean']:>8.2f}  std: {stats['std']:>8.2f}")

    # Create test dataset
    print(f"\nLoading test data from {TEST_CSV}...")
    test_dataset = BiomassDataset(
        csv_path=TEST_CSV,
        img_dir=TEST_IMG_DIR,
        transform=val_transform,
        is_test=True,
        target_stats=target_stats
    )

    print(f"Test dataset: {len(test_dataset)} samples")

    # Create test loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=BiomassDataset.collate_fn,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    # Load all fold models
    print(f"\n" + "=" * 70)
    print(f"Loading {n_folds} fold models...")
    print("=" * 70)

    fold_models = []
    for fold_idx in range(n_folds):
        fold_checkpoint_dir = checkpoint_dir / f'fold_{fold_idx}'
        checkpoint_path = fold_checkpoint_dir / 'best_model.pth'

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"\nLoading Fold {fold_idx + 1} model from {checkpoint_path}...")

        # Create model
        model = create_model(config)
        model = model.to(device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Val Loss: {checkpoint['best_val_loss']:.4f}")

        fold_models.append(model)

    print(f"\nSuccessfully loaded {len(fold_models)} models")

    # Generate predictions from all folds
    print(f"\n" + "=" * 70)
    print("Generating predictions from ensemble...")
    print("=" * 70)

    # Store predictions from each fold
    all_fold_predictions = []

    for fold_idx, model in enumerate(fold_models):
        print(f"\nGenerating predictions from Fold {fold_idx + 1}...")

        fold_predictions = {name: [] for name in TARGET_NAMES}

        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(device)
                predictions = model(images)

                # Denormalize for each sample in batch
                batch_size = images.size(0)
                for i in range(batch_size):
                    pred_dict = {name: predictions[name][i].item() for name in TARGET_NAMES}
                    pred_denorm = train_dataset.denormalize_targets(pred_dict)

                    for name in TARGET_NAMES:
                        fold_predictions[name].append(pred_denorm[name])

        all_fold_predictions.append(fold_predictions)

    # Average predictions across folds
    print(f"\nAveraging predictions across {n_folds} folds...")
    ensemble_predictions = {name: [] for name in TARGET_NAMES}

    # Get number of test samples
    n_test_samples = len(all_fold_predictions[0][TARGET_NAMES[0]])

    for sample_idx in range(n_test_samples):
        for target_name in TARGET_NAMES:
            # Collect predictions from all folds for this sample and target
            fold_preds = [fold_preds[target_name][sample_idx] for fold_preds in all_fold_predictions]
            # Average them
            ensemble_predictions[target_name].append(np.mean(fold_preds))

    # Apply biological constraints post-processing
    print("\nApplying biological constraints post-processing...")
    print("  - Clipping negative values to 0")
    print("  - Enforcing GDM = Green + Clover")
    print("  - Enforcing Total = GDM + Dead")

    for sample_idx in range(n_test_samples):
        # Step 1: Clip all predictions to non-negative
        for target_name in TARGET_NAMES:
            ensemble_predictions[target_name][sample_idx] = max(
                0.0, ensemble_predictions[target_name][sample_idx]
            )

        clover = ensemble_predictions['Dry_Clover_g'][sample_idx]
        dead = ensemble_predictions['Dry_Dead_g'][sample_idx]
        green = ensemble_predictions['Dry_Green_g'][sample_idx]
        gdm = ensemble_predictions['GDM_g'][sample_idx]
        total = ensemble_predictions['Dry_Total_g'][sample_idx]

        # Step 2: Enforce GDM = Green + Clover
        gdm_from_components = green + clover
        adjusted_gdm = (gdm + gdm_from_components) / 2

        if gdm_from_components > 0:
            gdm_scale = adjusted_gdm / gdm_from_components
            ensemble_predictions['Dry_Green_g'][sample_idx] = green * gdm_scale
            ensemble_predictions['Dry_Clover_g'][sample_idx] = clover * gdm_scale
        ensemble_predictions['GDM_g'][sample_idx] = adjusted_gdm

        # Step 3: Enforce Total = GDM + Dead
        total_from_components = adjusted_gdm + dead
        adjusted_total = (total + total_from_components) / 2

        if adjusted_total > adjusted_gdm:
            ensemble_predictions['Dry_Dead_g'][sample_idx] = adjusted_total - adjusted_gdm
        else:
            ensemble_predictions['Dry_Dead_g'][sample_idx] = 0.0
            adjusted_total = adjusted_gdm

        ensemble_predictions['Dry_Total_g'][sample_idx] = adjusted_total

    # Create submission DataFrame
    print("\nCreating submission file...")

    # Get test image IDs from test.csv
    test_df = pd.read_csv(TEST_CSV)
    test_image_ids = test_df['sample_id'].str.split('__').str[0].unique()

    submission_rows = []
    for image_id in test_image_ids:
        for target_name in TARGET_NAMES:
            sample_id = f"{image_id}__{target_name}"
            target_idx = list(TARGET_NAMES).index(target_name)
            prediction = ensemble_predictions[target_name][0]  # Assuming single test image

            submission_rows.append({
                'sample_id': sample_id,
                'target': prediction
            })

    submission_df = pd.DataFrame(submission_rows)

    # Generate output filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f'submission_kfold_ensemble_{n_folds}folds_{timestamp}.csv'
    output_path = SUBMISSIONS_DIR / output_filename

    # Ensure submissions directory exists
    SUBMISSIONS_DIR.mkdir(exist_ok=True, parents=True)

    # Save submission
    submission_df.to_csv(output_path, index=False)

    print("\n" + "=" * 70)
    print("Submission generation complete!")
    print(f"Saved to: {output_path}")
    print("=" * 70)

    # Print predictions summary
    print("\nEnsemble predictions by target:")
    for target_name in TARGET_NAMES:
        values = ensemble_predictions[target_name]
        print(f"  {target_name:<20} mean: {sum(values)/len(values):>8.2f}  "
              f"min: {min(values):>8.2f}  max: {max(values):>8.2f}")

    # Check constraint violations
    print("\nConstraint check:")

    # Check 1: GDM = Green + Clover
    gdm_violations = []
    for sample_idx in range(n_test_samples):
        gdm = ensemble_predictions['GDM_g'][sample_idx]
        gdm_calc = (ensemble_predictions['Dry_Green_g'][sample_idx] +
                   ensemble_predictions['Dry_Clover_g'][sample_idx])
        gdm_violations.append(abs(gdm - gdm_calc))

    print(f"  GDM = Green + Clover:")
    print(f"    Mean violation: {sum(gdm_violations)/len(gdm_violations):.6f}g")
    print(f"    Max violation: {max(gdm_violations):.6f}g")

    # Check 2: Total = GDM + Dead
    total_violations = []
    for sample_idx in range(n_test_samples):
        total = ensemble_predictions['Dry_Total_g'][sample_idx]
        total_calc = (ensemble_predictions['GDM_g'][sample_idx] +
                     ensemble_predictions['Dry_Dead_g'][sample_idx])
        total_violations.append(abs(total - total_calc))

    print(f"  Total = GDM + Dead:")
    print(f"    Mean violation: {sum(total_violations)/len(total_violations):.6f}g")
    print(f"    Max violation: {max(total_violations):.6f}g")

    # Check 3: No negative values
    has_negatives = any(
        ensemble_predictions[t][i] < 0
        for t in TARGET_NAMES
        for i in range(n_test_samples)
    )
    print(f"  Non-negative check: {'PASSED' if not has_negatives else 'FAILED'}")

    # Print individual fold predictions for comparison
    print("\n" + "=" * 70)
    print("Individual fold predictions (for comparison):")
    print("=" * 70)
    for target_name in TARGET_NAMES:
        print(f"\n{target_name}:")
        for fold_idx in range(n_folds):
            pred = all_fold_predictions[fold_idx][target_name][0]
            print(f"  Fold {fold_idx + 1}: {pred:>8.2f}")
        ensemble_pred = ensemble_predictions[target_name][0]
        print(f"  Ensemble: {ensemble_pred:>8.2f}")

    print(f"\n{'=' * 70}")
    print("Ready to submit to Kaggle!")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
