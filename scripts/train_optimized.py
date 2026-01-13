"""
Train final model with Optuna-optimized hyperparameters.

Usage:
    python scripts/train_optimized.py
    python scripts/train_optimized.py --config path/to/config.json
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
from src.data.splitter import create_stratified_split
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.model_factory import create_model
from src.models.loss_functions import create_loss_function
from src.training.optimizer_factory import create_optimizer, create_scheduler
from src.training.trainer import BiomassTrainer


def main():
    parser = argparse.ArgumentParser(description='Train optimized model')
    parser.add_argument('--config', type=str,
                       default='experiments/optuna_studies/optuna_v1/best_config.json',
                       help='Path to config JSON file')
    parser.add_argument('--checkpoint_dir', type=str,
                       default='experiments/checkpoints_optimized',
                       help='Directory to save checkpoints')
    args = parser.parse_args()

    print("=" * 70)
    print("Image2Biomass - Training Optimized Model")
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

    # Create stratified split
    print(f"\nCreating stratified split...")
    train_df, val_df = create_stratified_split(
        df_wide,
        stratify_by=config['stratify_by'],
        val_split=config['val_split'],
        random_seed=config['seed']
    )

    # Get transforms
    train_transform = get_train_transforms(
        level=config['augmentation_level'],
        image_size=config['image_size']
    )
    val_transform = get_val_transforms(image_size=config['image_size'])

    # Convert to long format for datasets
    print("\nCreating datasets...")
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
    train_csv_path = temp_dir / 'train_optimized.csv'
    val_csv_path = temp_dir / 'val_optimized.csv'
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

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    print(f"\nCreating model: {config['backbone']}...")
    model = create_model(config)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # Create loss function
    criterion = create_loss_function(config)
    print(f"Loss function: {config['loss_function']}")

    # Create optimizer
    optimizer = create_optimizer(model, config)
    print(f"Optimizer: {config['optimizer']}")

    # Create scheduler
    steps_per_epoch = len(train_loader)
    scheduler = create_scheduler(optimizer, config, steps_per_epoch=steps_per_epoch)
    if scheduler:
        print(f"Scheduler: {config['scheduler']}")

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    # Create trainer
    print(f"\nInitializing trainer...")
    trainer = BiomassTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=checkpoint_dir,
        early_stopping_patience=config['early_stopping_patience'],
        denormalize_fn=train_dataset.denormalize_targets
    )

    # Train
    print(f"\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)

    best_val_loss = trainer.train(num_epochs=config['num_epochs'])

    print(f"\n" + "=" * 70)
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"\nComparison:")
    print(f"  Baseline model:   1.4178")
    print(f"  Optimized model:  {best_val_loss:.4f}")
    print(f"  Improvement:      {((1.4178 - best_val_loss) / 1.4178 * 100):.1f}%")
    print("=" * 70)


if __name__ == '__main__':
    main()
