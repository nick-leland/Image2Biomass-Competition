"""
Train K-Fold Cross-Validation models with Optuna-optimized hyperparameters.

This script trains K separate models (one per fold) that can be ensembled
at inference time for more robust predictions.

Usage:
    python scripts/train_kfold.py --n_folds 5
    python scripts/train_kfold.py --n_folds 5 --config path/to/config.json
"""

import sys
sys.path.insert(0, '/home/chaot/kaggle/Image2Biomass-Competition')

import argparse
import json
import torch
import pandas as pd
from pathlib import Path

from src.config import TRAIN_CSV, TRAIN_IMG_DIR, TARGET_NAMES
from src.utils.seed import set_seed, worker_init_fn
from src.data.dataset import BiomassDataset, create_dataloaders
from src.data.splitter import get_kfold_splits
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.model_factory import create_model
from src.models.loss_functions import create_loss_function
from src.training.optimizer_factory import create_optimizer, create_scheduler
from src.training.trainer import BiomassTrainer


def main():
    parser = argparse.ArgumentParser(description='Train K-Fold CV models')
    parser.add_argument('--config', type=str,
                       default='experiments/optuna_studies/advanced_models_20260111_013829/best_config.json',
                       help='Path to config JSON file')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of folds for cross-validation')
    parser.add_argument('--checkpoint_dir', type=str,
                       default='experiments/checkpoints_kfold',
                       help='Base directory to save fold checkpoints')
    parser.add_argument('--start_fold', type=int, default=0,
                       help='Starting fold index (for resuming)')
    args = parser.parse_args()

    print("=" * 70)
    print(f"Image2Biomass - {args.n_folds}-Fold Cross-Validation Training")
    print("=" * 70)

    # Load config
    config_path = Path(args.config)
    print(f"\nLoading config from: {config_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    print("\nOptimized Hyperparameters:")
    print("-" * 70)
    for key, value in config.items():
        if key != 'task_weights':
            print(f"  {key:<25} {value}")
        else:
            print(f"  {key}:")
            for target, weight in value.items():
                print(f"    {target:<20} {weight}")
    print("-" * 70)

    # Set seed
    set_seed(config['seed'])

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print(f"\nLoading data from {TRAIN_CSV}...")
    df = pd.read_csv(TRAIN_CSV)
    df['image_id'] = df['sample_id'].str.split('__').str[0]
    df_wide = df.pivot_table(
        index=['image_id', 'image_path', 'Sampling_Date', 'State', 'Species',
               'Pre_GSHH_NDVI', 'Height_Ave_cm'],
        columns='target_name',
        values='target'
    ).reset_index()

    print(f"Total images: {len(df_wide)}")

    # Create base checkpoint directory
    base_checkpoint_dir = Path(args.checkpoint_dir)
    base_checkpoint_dir.mkdir(exist_ok=True, parents=True)

    # Store all fold results
    fold_results = []

    # K-Fold training loop
    print(f"\n" + "=" * 70)
    print(f"Starting {args.n_folds}-Fold Cross-Validation Training")
    print("=" * 70)

    for fold_idx, (train_df, val_df) in get_kfold_splits(
        df_wide,
        n_folds=args.n_folds,
        stratify_by=config['stratify_by'],
        random_seed=config['seed']
    ):
        if fold_idx < args.start_fold:
            print(f"\nSkipping Fold {fold_idx + 1} (already trained)")
            continue

        print(f"\n{'=' * 70}")
        print(f"FOLD {fold_idx + 1}/{args.n_folds}")
        print(f"{'=' * 70}")

        # Reset seed for each fold for reproducibility
        set_seed(config['seed'] + fold_idx)

        # Get transforms
        train_transform = get_train_transforms(
            level=config['augmentation_level'],
            image_size=config['image_size']
        )
        val_transform = get_val_transforms(image_size=config['image_size'])

        # Convert to long format for datasets
        print(f"\nPreparing Fold {fold_idx + 1} data...")
        train_long = []
        for _, row in train_df.iterrows():
            for target_name in TARGET_NAMES:
                train_long.append({
                    'sample_id': f"{row['image_id']}__{target_name}",
                    'image_id': row['image_id'],
                    'image_path': row['image_path'],
                    'target_name': target_name,
                    'target': row[target_name],
                    'State': row.get('State', ''),
                    'Species': row.get('Species', ''),
                    'Pre_GSHH_NDVI': row.get('Pre_GSHH_NDVI', 0),
                    'Height_Ave_cm': row.get('Height_Ave_cm', 0),
                    'Sampling_Date': row.get('Sampling_Date', ''),
                })
        train_long_df = pd.DataFrame(train_long)

        val_long = []
        for _, row in val_df.iterrows():
            for target_name in TARGET_NAMES:
                val_long.append({
                    'sample_id': f"{row['image_id']}__{target_name}",
                    'image_id': row['image_id'],
                    'image_path': row['image_path'],
                    'target_name': target_name,
                    'target': row[target_name],
                    'State': row.get('State', ''),
                    'Species': row.get('Species', ''),
                    'Pre_GSHH_NDVI': row.get('Pre_GSHH_NDVI', 0),
                    'Height_Ave_cm': row.get('Height_Ave_cm', 0),
                    'Sampling_Date': row.get('Sampling_Date', ''),
                })
        val_long_df = pd.DataFrame(val_long)

        # Save temporary CSVs
        temp_dir = Path('experiments/temp')
        temp_dir.mkdir(exist_ok=True, parents=True)
        train_csv_path = temp_dir / f'train_fold{fold_idx}.csv'
        val_csv_path = temp_dir / f'val_fold{fold_idx}.csv'
        train_long_df.to_csv(train_csv_path, index=False)
        val_long_df.to_csv(val_csv_path, index=False)

        # Create datasets
        train_dataset = BiomassDataset(
            csv_path=train_csv_path,
            img_dir=TRAIN_IMG_DIR,
            transform=train_transform,
            is_test=False
        )

        val_dataset = BiomassDataset(
            csv_path=val_csv_path,
            img_dir=TRAIN_IMG_DIR,
            transform=val_transform,
            is_test=False,
            target_stats=train_dataset.get_target_stats()
        )

        print(f"Train dataset: {len(train_dataset)} samples")
        print(f"Val dataset: {len(val_dataset)} samples")

        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            train_dataset,
            val_dataset,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            worker_init_fn=worker_init_fn
        )

        # Create model
        print(f"\nCreating model: {config['backbone']}...")
        model = create_model(config)
        model = model.to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {n_params:,}")

        # Create loss function
        criterion = create_loss_function(config)

        # Create optimizer
        optimizer = create_optimizer(model, config)

        # Create scheduler
        steps_per_epoch = len(train_loader)
        scheduler = create_scheduler(optimizer, config, steps_per_epoch=steps_per_epoch)

        # Create fold-specific checkpoint directory
        fold_checkpoint_dir = base_checkpoint_dir / f'fold_{fold_idx}'
        fold_checkpoint_dir.mkdir(exist_ok=True, parents=True)

        # Create trainer
        print(f"\nInitializing trainer for Fold {fold_idx + 1}...")
        trainer = BiomassTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            checkpoint_dir=fold_checkpoint_dir,
            early_stopping_patience=config['early_stopping_patience'],
            denormalize_fn=train_dataset.denormalize_targets
        )

        # Train
        print(f"\nTraining Fold {fold_idx + 1}...")
        best_val_loss = trainer.train(num_epochs=config['num_epochs'])

        # Store results
        fold_results.append({
            'fold': fold_idx,
            'best_val_loss': best_val_loss,
            'checkpoint_dir': str(fold_checkpoint_dir)
        })

        print(f"\nFold {fold_idx + 1} completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Checkpoint saved to: {fold_checkpoint_dir}")

        # Clean up GPU memory
        del model, optimizer, scheduler, trainer, train_loader, val_loader
        del train_dataset, val_dataset
        torch.cuda.empty_cache()

    # Print final summary
    print(f"\n" + "=" * 70)
    print(f"K-FOLD CROSS-VALIDATION SUMMARY")
    print("=" * 70)
    print(f"\nResults for all {args.n_folds} folds:")
    print("-" * 70)

    for result in fold_results:
        print(f"Fold {result['fold'] + 1}: Val Loss = {result['best_val_loss']:.4f}")

    avg_val_loss = sum(r['best_val_loss'] for r in fold_results) / len(fold_results)
    std_val_loss = (sum((r['best_val_loss'] - avg_val_loss) ** 2 for r in fold_results) / len(fold_results)) ** 0.5

    print("-" * 70)
    print(f"Mean Val Loss:  {avg_val_loss:.4f} ± {std_val_loss:.4f}")
    print(f"\nAll checkpoints saved to: {base_checkpoint_dir}")
    print("=" * 70)

    # Save fold results to JSON
    results_path = base_checkpoint_dir / 'kfold_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'n_folds': args.n_folds,
            'mean_val_loss': avg_val_loss,
            'std_val_loss': std_val_loss,
            'fold_results': fold_results,
            'config': config
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print("\nNext steps:")
    print(f"  1. Generate ensemble predictions with: python scripts/generate_submission_kfold.py")
    print(f"  2. Each fold model can be found in: {base_checkpoint_dir}/fold_X/best_model.pth")


if __name__ == '__main__':
    main()
