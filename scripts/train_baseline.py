"""
Training script for baseline biomass prediction model.

Usage:
    python scripts/train_baseline.py
"""

import sys
sys.path.insert(0, '/home/chaot/kaggle/Image2Biomass-Competition')

import torch
import pandas as pd
from pathlib import Path

from src.config import (
    TRAIN_CSV, TRAIN_IMG_DIR, DEFAULT_CONFIG,
    CHECKPOINTS_DIR, TARGET_NAMES
)
from src.utils.seed import set_seed
from src.data.dataset import BiomassDataset, create_dataloaders
from src.data.splitter import create_stratified_split
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.model_factory import create_model
from src.models.loss_functions import create_loss_function
from src.training.optimizer_factory import create_optimizer, create_scheduler
from src.training.trainer import BiomassTrainer


def main():
    """Main training function."""
    print("=" * 70)
    print("Image2Biomass Baseline Training")
    print("=" * 70)

    # Set seed for reproducibility
    set_seed(DEFAULT_CONFIG['seed'])

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print(f"\nLoading data from {TRAIN_CSV}...")
    df = pd.read_csv(TRAIN_CSV)

    # Extract image_id and pivot to wide format
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
        stratify_by=DEFAULT_CONFIG['stratify_by'],
        val_split=DEFAULT_CONFIG['val_split'],
        random_seed=DEFAULT_CONFIG['seed']
    )

    # Get transforms
    train_transform = get_train_transforms(
        level=DEFAULT_CONFIG['augmentation_level'],
        image_size=DEFAULT_CONFIG['image_size']
    )
    val_transform = get_val_transforms(image_size=DEFAULT_CONFIG['image_size'])

    # Convert back to long format CSVs for dataset
    # (Dataset expects long format internally but we pass wide for splitting)
    print("\nCreating datasets...")

    # For simplicity, recreate long format dataframes
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
    train_csv_path = temp_dir / 'train_split.csv'
    val_csv_path = temp_dir / 'val_split.csv'
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
        target_stats=train_dataset.get_target_stats()  # Use training stats
    )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")

    # Create dataloaders
    from src.utils.seed import worker_init_fn
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=DEFAULT_CONFIG['batch_size'],
        num_workers=DEFAULT_CONFIG['num_workers'],
        worker_init_fn=worker_init_fn
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    print(f"\nCreating model: {DEFAULT_CONFIG['backbone']}...")
    model = create_model(DEFAULT_CONFIG)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # Create loss function
    criterion = create_loss_function(DEFAULT_CONFIG)
    print(f"Loss function: {DEFAULT_CONFIG['loss_function']}")

    # Create optimizer
    optimizer = create_optimizer(model, DEFAULT_CONFIG)
    print(f"Optimizer: {DEFAULT_CONFIG['optimizer']}")

    # Create scheduler
    scheduler = create_scheduler(optimizer, DEFAULT_CONFIG)
    if scheduler:
        print(f"Scheduler: {DEFAULT_CONFIG['scheduler']}")

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
        checkpoint_dir=CHECKPOINTS_DIR,
        early_stopping_patience=DEFAULT_CONFIG['early_stopping_patience'],
        denormalize_fn=train_dataset.denormalize_targets
    )

    # Train
    print(f"\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)

    best_val_loss = trainer.train(num_epochs=DEFAULT_CONFIG['num_epochs'])

    print(f"\n" + "=" * 70)
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {CHECKPOINTS_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
