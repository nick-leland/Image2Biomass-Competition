"""
Quick training script for Vision Transformer baseline.

Usage:
    python scripts/train_vit.py
"""

import sys
sys.path.insert(0, '/home/chaot/kaggle/Image2Biomass-Competition')

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
    print("=" * 70)
    print("Training Vision Transformer (ViT) Model")
    print("=" * 70)

    # ViT config based on best ResNet50 settings
    config = {
        'backbone': 'vit_base_patch16_384',  # Vision Transformer
        'pretrained': True,
        'dropout': 0.1,
        'head_hidden_dim': 256,
        'constraint_mode': 'none',
        'constraint_weight': 0.0,
        'loss_function': 'huber',
        'huber_delta': 2.0,
        'task_weights': {
            'Dry_Clover_g': 1.25,
            'Dry_Dead_g': 0.75,
            'Dry_Green_g': 1.25,
            'Dry_Total_g': 1.5,
            'GDM_g': 0.75
        },
        'optimizer': 'adamw',
        'learning_rate': 0.0003,  # Slightly higher for ViT
        'weight_decay': 0.0001,
        'momentum': 0.9,
        'scheduler': 'cosine',
        'scheduler_patience': 3,
        'augmentation_level': 'aggressive',
        'image_size': 384,  # ViT works well with 384x384
        'batch_size': 8,  # Smaller batch for larger model
        'num_epochs': 50,
        'early_stopping_patience': 10,
        'num_workers': 4,
        'seed': 42,
        'stratify_by': 'state',
        'val_split': 0.2
    }

    print("\nVision Transformer Configuration:")
    print(f"  Model: {config['backbone']}")
    print(f"  Image size: {config['image_size']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['learning_rate']}")

    # Set seed
    set_seed(config['seed'])

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Load and split data
    print(f"\nLoading data...")
    df = pd.read_csv(TRAIN_CSV)
    df['image_id'] = df['sample_id'].str.split('__').str[0]
    df_wide = df.pivot_table(
        index=['image_id', 'image_path', 'Sampling_Date', 'State', 'Species',
               'Pre_GSHH_NDVI', 'Height_Ave_cm'],
        columns='target_name',
        values='target'
    ).reset_index()

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

    # Save temporary CSVs
    temp_dir = Path('experiments/temp')
    temp_dir.mkdir(exist_ok=True, parents=True)
    train_csv_path = temp_dir / 'train_vit.csv'
    val_csv_path = temp_dir / 'val_vit.csv'
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

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        worker_init_fn=worker_init_fn
    )

    # Create model
    print(f"\nCreating Vision Transformer model...")
    model = create_model(config)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # Create loss, optimizer, scheduler
    criterion = create_loss_function(config)
    optimizer = create_optimizer(model, config)
    steps_per_epoch = len(train_loader)
    scheduler = create_scheduler(optimizer, config, steps_per_epoch=steps_per_epoch)

    # Create checkpoint directory
    checkpoint_dir = Path('experiments/checkpoints_vit')
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    # Create trainer
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
    print(f"\n{'=' * 70}")
    print("Starting training...")
    print(f"{'=' * 70}")

    best_val_loss = trainer.train(num_epochs=config['num_epochs'])

    print(f"\n{'=' * 70}")
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"\nComparison:")
    print(f"  ResNet50:  0.4387")
    print(f"  ViT:       {best_val_loss:.4f}")
    if best_val_loss < 0.4387:
        print(f"  Improvement: {((0.4387 - best_val_loss) / 0.4387 * 100):.1f}%")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
