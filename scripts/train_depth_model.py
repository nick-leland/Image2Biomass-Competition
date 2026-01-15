"""
Train RGB+Depth fusion model for biomass prediction.

This script trains a model that combines RGB image features with monocular
depth estimation for improved biomass prediction.

Usage:
    # Train on competition data only
    python scripts/train_depth_model.py

    # Train on combined data (competition + external)
    python scripts/train_depth_model.py --use_external

    # Use different fusion strategies
    python scripts/train_depth_model.py --fusion_type attention
"""

import sys
sys.path.insert(0, '/home/chaot/kaggle/Image2Biomass-Competition')

import argparse
import json
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime

from src.config import TRAIN_CSV, TRAIN_IMG_DIR, TARGET_NAMES
from src.utils.seed import set_seed, worker_init_fn
from src.data.dataset import BiomassDataset, create_dataloaders
from src.data.splitter import get_group_kfold_splits
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.model_factory import create_model
from src.models.loss_functions import create_loss_function
from src.training.optimizer_factory import create_optimizer, create_scheduler
from src.training.trainer_tensorboard import BiomassTrainerTensorBoard


def main():
    parser = argparse.ArgumentParser(description='Train RGB+Depth fusion model')
    parser.add_argument('--use_external', action='store_true',
                       help='Include external GrassClover data')
    parser.add_argument('--model_type', type=str, default='rgb_depth_fusion',
                       choices=['rgb_depth_fusion', 'rgbd'],
                       help='Model type: rgb_depth_fusion (dual encoder) or rgbd (4-channel)')
    parser.add_argument('--fusion_type', type=str, default='concat',
                       choices=['concat', 'add', 'attention'],
                       help='Fusion type for rgb_depth_fusion model')
    parser.add_argument('--depth_model', type=str, default='depth_anything_v2_small',
                       choices=['depth_anything_v2_small', 'depth_anything_v2_base', 'midas_small'],
                       help='Depth estimation model')
    parser.add_argument('--backbone', type=str, default='efficientnetv2_rw_m',
                       help='RGB encoder backbone')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs per fold')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size (smaller due to depth model memory)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--image_size', type=int, default=384,
                       help='Image size (smaller for memory)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()

    print("=" * 70)
    print("Image2Biomass - RGB+Depth Fusion Model Training")
    print("=" * 70)

    # Configuration
    config = {
        'model_type': args.model_type,
        'backbone': args.backbone,
        'depth_model': args.depth_model,
        'fusion_type': args.fusion_type,
        'freeze_depth': True,
        'pretrained': True,
        'dropout': 0.3,
        'loss_function': 'mse',
        'optimizer': 'adamw',
        'learning_rate': args.lr,
        'weight_decay': 0.01,
        'scheduler': 'cosine',
        'image_size': args.image_size,
        'batch_size': args.batch_size,
        'num_workers': 4,
        'num_epochs': args.epochs,
        'early_stopping_patience': 10,
        'seed': args.seed,
        'augmentation_level': 'moderate',
        'use_external_data': args.use_external,
        'task_weights': {name: 1.0 for name in TARGET_NAMES}
    }

    print("\nConfiguration:")
    print("-" * 70)
    for key, value in config.items():
        if key != 'task_weights':
            print(f"  {key:<25} {value}")
    print("-" * 70)

    # Set seed
    set_seed(config['seed'])

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load data
    if args.use_external:
        print("\nLoading combined data (competition + external)...")
        csv_path = Path('external_data/processed/combined_train.csv')
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Combined data not found at {csv_path}. "
                "Run scripts/prepare_external_data.py first."
            )
        img_dir = Path('.')  # Paths in CSV are relative to repo root
    else:
        print(f"\nLoading competition data from {TRAIN_CSV}...")
        csv_path = TRAIN_CSV
        img_dir = TRAIN_IMG_DIR

    df = pd.read_csv(csv_path)
    df['image_id'] = df['sample_id'].str.split('__').str[0]

    # Convert to wide format
    df_wide = df.pivot_table(
        index=['image_id', 'image_path', 'Sampling_Date', 'State', 'Species',
               'Pre_GSHH_NDVI', 'Height_Ave_cm'],
        columns='target_name',
        values='target'
    ).reset_index()

    print(f"Total images: {len(df_wide)}")

    # Create checkpoint directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    data_suffix = '_combined' if args.use_external else ''
    checkpoint_dir = Path(f'experiments/checkpoints_depth_{args.model_type}{data_suffix}_{timestamp}')
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    # Save config
    config_path = checkpoint_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Store fold results
    fold_results = []

    # K-Fold training
    print(f"\n" + "=" * 70)
    print(f"Starting {args.n_folds}-Fold Cross-Validation")
    print(f"TensorBoard: tensorboard --logdir {checkpoint_dir}")
    print("=" * 70)

    for fold_idx, (train_df, val_df) in get_group_kfold_splits(
        df_wide,
        n_folds=args.n_folds,
        group_by='location',
        random_seed=config['seed']
    ):
        print(f"\n{'=' * 70}")
        print(f"FOLD {fold_idx + 1}/{args.n_folds}")
        print(f"{'=' * 70}")

        # Reset seed for reproducibility
        set_seed(config['seed'] + fold_idx)

        # Get transforms
        train_transform = get_train_transforms(
            level=config['augmentation_level'],
            image_size=config['image_size']
        )
        val_transform = get_val_transforms(image_size=config['image_size'])

        # Convert to long format
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

        # Save temp CSVs
        temp_dir = Path('experiments/temp')
        temp_dir.mkdir(exist_ok=True, parents=True)
        train_csv_path = temp_dir / f'train_depth_fold{fold_idx}.csv'
        val_csv_path = temp_dir / f'val_depth_fold{fold_idx}.csv'
        train_long_df.to_csv(train_csv_path, index=False)
        val_long_df.to_csv(val_csv_path, index=False)

        # Create datasets
        train_dataset = BiomassDataset(
            csv_path=train_csv_path,
            img_dir=img_dir,
            transform=train_transform,
            is_test=False
        )

        val_dataset = BiomassDataset(
            csv_path=val_csv_path,
            img_dir=img_dir,
            transform=val_transform,
            is_test=False,
            target_stats=train_dataset.get_target_stats()
        )

        print(f"Train: {len(train_dataset)} samples ({len(train_df)} images)")
        print(f"Val: {len(val_dataset)} samples ({len(val_df)} images)")

        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            train_dataset,
            val_dataset,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            worker_init_fn=worker_init_fn
        )

        # Create model
        print(f"\nCreating {config['model_type']} model...")
        print(f"  RGB backbone: {config['backbone']}")
        print(f"  Depth model: {config['depth_model']}")
        if config['model_type'] == 'rgb_depth_fusion':
            print(f"  Fusion type: {config['fusion_type']}")

        model = create_model(config)
        model = model.to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {n_params:,}")

        # Create loss, optimizer, scheduler
        criterion = create_loss_function(config)
        optimizer = create_optimizer(model, config)

        steps_per_epoch = len(train_loader)
        scheduler = create_scheduler(optimizer, config, steps_per_epoch=steps_per_epoch)

        # Create fold checkpoint directory
        fold_checkpoint_dir = checkpoint_dir / f'fold_{fold_idx}'
        fold_checkpoint_dir.mkdir(exist_ok=True, parents=True)

        # Create trainer
        trainer = BiomassTrainerTensorBoard(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            checkpoint_dir=fold_checkpoint_dir,
            early_stopping_patience=config['early_stopping_patience'],
            denormalize_fn=train_dataset.denormalize_targets,
            tensorboard_dir=checkpoint_dir / 'tensorboard',
            fold_idx=fold_idx
        )

        # Train
        print(f"\nTraining Fold {fold_idx + 1}...")
        best_val_loss = trainer.train(num_epochs=config['num_epochs'])

        fold_results.append({
            'fold': fold_idx,
            'best_val_loss': best_val_loss,
            'checkpoint_dir': str(fold_checkpoint_dir)
        })

        print(f"\nFold {fold_idx + 1} completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")

        # Cleanup
        del model, optimizer, scheduler, trainer
        torch.cuda.empty_cache()

    # Summary
    print(f"\n" + "=" * 70)
    print(f"TRAINING COMPLETE - {config['model_type'].upper()}")
    print("=" * 70)

    for result in fold_results:
        print(f"Fold {result['fold'] + 1}: Val Loss = {result['best_val_loss']:.4f}")

    if fold_results:
        avg_loss = sum(r['best_val_loss'] for r in fold_results) / len(fold_results)
        std_loss = (sum((r['best_val_loss'] - avg_loss) ** 2 for r in fold_results) / len(fold_results)) ** 0.5

        print("-" * 70)
        print(f"Mean Val Loss: {avg_loss:.4f} +/- {std_loss:.4f}")
        print(f"\nCheckpoints saved to: {checkpoint_dir}")

        # Save results
        results_path = checkpoint_dir / 'kfold_results.json'
        with open(results_path, 'w') as f:
            json.dump({
                'n_folds': args.n_folds,
                'model_type': config['model_type'],
                'mean_val_loss': avg_loss,
                'std_val_loss': std_loss,
                'fold_results': fold_results,
                'config': config
            }, f, indent=2)


if __name__ == '__main__':
    main()
