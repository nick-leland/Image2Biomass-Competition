#!/usr/bin/env python3
"""
Train biomass prediction with foundation model backbones (DINOv2, SigLIP, EVA02).

These models have strong pretrained features that may generalize better.

Usage:
    # DINOv2 (recommended)
    python scripts/train_foundation_models.py --backbone dinov2_base --epochs 30

    # SigLIP
    python scripts/train_foundation_models.py --backbone siglip_base --epochs 30

    # EVA02
    python scripts/train_foundation_models.py --backbone eva02_base --epochs 30

    # With depth fusion
    python scripts/train_foundation_models.py --backbone dinov2_base --use_depth --epochs 30
"""

import sys
sys.path.insert(0, '/home/chaot/kaggle/Image2Biomass-Competition')

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import timm

from src.data.dataset import BiomassDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.data.splitter import get_group_kfold_splits
from src.training.trainer import BiomassTrainer as Trainer
from src.utils.seed import set_seed


# Backbone configurations
BACKBONE_CONFIGS = {
    'dinov2_base': {
        'model_name': 'vit_base_patch14_dinov2',
        'image_size': 518,  # Native size for DINOv2
        'features': 768,
    },
    'dinov2_base_reg': {
        'model_name': 'vit_base_patch14_reg4_dinov2',
        'image_size': 518,
        'features': 768,
    },
    'siglip_base': {
        'model_name': 'vit_base_patch16_siglip_384',
        'image_size': 384,
        'features': 768,
    },
    'siglip_base_512': {
        'model_name': 'vit_base_patch16_siglip_512',
        'image_size': 512,
        'features': 768,
    },
    'eva02_base': {
        'model_name': 'eva02_base_patch14_448',
        'image_size': 448,
        'features': 768,
    },
    'eva02_large': {
        'model_name': 'eva02_large_patch14_448',
        'image_size': 448,
        'features': 1024,
    },
}

TARGET_NAMES = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']


class FoundationModelRegressor(nn.Module):
    """Multi-task regressor using foundation model backbone."""

    def __init__(
        self,
        backbone_name: str = 'vit_base_patch14_dinov2',
        num_features: int = 768,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.target_names = TARGET_NAMES

        # Load backbone
        # Note: DINOv2 models don't work with global_pool='avg', use default
        if 'dinov2' in backbone_name:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=True,
                num_classes=0,
            )
        else:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=True,
                num_classes=0,
                global_pool='avg'
            )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Regression heads
        self.heads = nn.ModuleDict()
        for name in self.target_names:
            self.heads[name] = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            )

    def forward(self, x):
        features = self.backbone(x)
        return {name: self.heads[name](features).squeeze(-1) for name in self.target_names}


class FoundationModelWithDepth(nn.Module):
    """Foundation model + depth fusion."""

    def __init__(
        self,
        backbone_name: str = 'vit_base_patch14_dinov2',
        num_features: int = 768,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.target_names = TARGET_NAMES

        # RGB backbone
        # Note: DINOv2 models don't work with global_pool='avg', use default
        if 'dinov2' in backbone_name:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=True,
                num_classes=0,
            )
        else:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=True,
                num_classes=0,
                global_pool='avg'
            )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Depth estimator
        from src.models.depth_encoder import DepthEstimator
        self.depth_estimator = DepthEstimator(
            model_type='depth_anything_v2_small',
            freeze=True
        )

        # Depth encoder
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        fused_features = num_features + 256

        # Regression heads
        self.heads = nn.ModuleDict()
        for name in self.target_names:
            self.heads[name] = nn.Sequential(
                nn.Linear(fused_features, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            )

    def forward(self, x):
        # RGB features
        rgb_features = self.backbone(x)

        # Depth features
        with torch.no_grad():
            depth_maps = self.depth_estimator(x)
        depth_features = self.depth_encoder(depth_maps)

        # Fuse
        fused = torch.cat([rgb_features, depth_features], dim=1)

        return {name: self.heads[name](fused).squeeze(-1) for name in self.target_names}


def get_parameter_groups(model, backbone_lr: float, head_lr: float, weight_decay: float):
    """Create parameter groups with differential learning rates."""
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    return [
        {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': weight_decay},
        {'params': head_params, 'lr': head_lr, 'weight_decay': weight_decay},
    ]


def get_warmup_cosine_scheduler(optimizer, warmup_epochs: int, total_epochs: int, min_lr: float = 1e-7):
    """Create a scheduler with linear warmup followed by cosine annealing."""
    from torch.optim.lr_scheduler import LambdaLR
    import math

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def train_fold(
    fold_idx: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: dict,
    checkpoint_dir: Path,
    device: torch.device,
):
    """Train a single fold."""

    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx + 1}/{config['n_folds']}")
    print(f"{'='*60}")

    set_seed(config['seed'])

    # Get backbone config
    backbone_config = BACKBONE_CONFIGS[config['backbone']]
    image_size = backbone_config['image_size']

    # Create transforms
    train_transform = get_train_transforms(
        image_size=image_size,
        level='moderate'
    )
    val_transform = get_val_transforms(image_size=image_size)

    # Convert to long format and save temp CSVs
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
                'Sampling_Date': row.get('Sampling_Date', ''),
            })
    val_long_df = pd.DataFrame(val_long)

    # Save temp CSVs
    temp_dir = Path('experiments/temp')
    temp_dir.mkdir(exist_ok=True, parents=True)
    train_csv_path = temp_dir / f'train_foundation_fold{fold_idx}.csv'
    val_csv_path = temp_dir / f'val_foundation_fold{fold_idx}.csv'
    train_long_df.to_csv(train_csv_path, index=False)
    val_long_df.to_csv(val_csv_path, index=False)

    # Create datasets
    train_dataset = BiomassDataset(
        csv_path=train_csv_path,
        img_dir=Path(config['img_dir']),
        transform=train_transform,
        is_test=False
    )

    val_dataset = BiomassDataset(
        csv_path=val_csv_path,
        img_dir=Path(config['img_dir']),
        transform=val_transform,
        is_test=False,
        target_stats=train_dataset.get_target_stats()
    )

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    # Create model
    print(f"\nCreating {config['backbone']} model...")
    print(f"  Backbone: {backbone_config['model_name']}")
    print(f"  Image size: {image_size}")
    print(f"  Use depth: {config['use_depth']}")

    if config['use_depth']:
        model = FoundationModelWithDepth(
            backbone_name=backbone_config['model_name'],
            num_features=backbone_config['features'],
            dropout=config['dropout'],
            freeze_backbone=config['freeze_backbone'],
        )
    else:
        model = FoundationModelRegressor(
            backbone_name=backbone_config['model_name'],
            num_features=backbone_config['features'],
            dropout=config['dropout'],
            freeze_backbone=config['freeze_backbone'],
        )

    model = model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    # Create optimizer with differential learning rates
    backbone_lr = config['learning_rate'] * config.get('backbone_lr_scale', 0.1)
    head_lr = config['learning_rate']

    print(f"  Backbone LR: {backbone_lr:.2e}")
    print(f"  Head LR: {head_lr:.2e}")
    print(f"  Warmup epochs: {config.get('warmup_epochs', 3)}")
    print(f"  Gradient accumulation: {config.get('gradient_accumulation_steps', 4)}")

    param_groups = get_parameter_groups(
        model,
        backbone_lr=backbone_lr,
        head_lr=head_lr,
        weight_decay=config['weight_decay']
    )

    optimizer = torch.optim.AdamW(param_groups)

    # Create scheduler with warmup
    warmup_epochs = config.get('warmup_epochs', 3)
    scheduler = get_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=config['num_epochs'],
        min_lr=1e-7
    )

    # Create criterion (MSE loss for each target)
    from src.models.loss_functions import MultiTaskMSELoss
    criterion = MultiTaskMSELoss(
        task_weights={name: 1.0 for name in TARGET_NAMES}
    )

    # Create trainer
    fold_checkpoint_dir = checkpoint_dir / f'fold_{fold_idx}'
    fold_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
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
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 4),
        max_grad_norm=config.get('max_grad_norm', 1.0),
    )

    # Train
    print(f"\nTraining Fold {fold_idx + 1}...")
    best_val_loss = trainer.train(num_epochs=config['num_epochs'])

    return {
        'fold': fold_idx,
        'best_val_loss': best_val_loss,
        'checkpoint_dir': str(fold_checkpoint_dir),
    }


def main():
    parser = argparse.ArgumentParser(description='Train with foundation model backbones')
    parser.add_argument('--backbone', type=str, default='dinov2_base',
                        choices=list(BACKBONE_CONFIGS.keys()),
                        help='Backbone to use')
    parser.add_argument('--use_depth', action='store_true',
                        help='Add depth fusion')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze backbone weights')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of folds')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (for heads)')
    parser.add_argument('--backbone_lr_scale', type=float, default=0.1,
                        help='Backbone LR = lr * backbone_lr_scale (default: 0.1)')
    parser.add_argument('--warmup_epochs', type=int, default=3,
                        help='Number of warmup epochs')
    parser.add_argument('--gradient_accumulation', type=int, default=4,
                        help='Gradient accumulation steps (effective batch = batch_size * this)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Paths
    train_csv = Path('/home/chaot/kaggle/Image2Biomass-Competition/train.csv')
    img_dir = Path('/home/chaot/kaggle/Image2Biomass-Competition/train')

    # Config
    backbone_config = BACKBONE_CONFIGS[args.backbone]

    config = {
        'backbone': args.backbone,
        'backbone_model': backbone_config['model_name'],
        'image_size': backbone_config['image_size'],
        'use_depth': args.use_depth,
        'freeze_backbone': args.freeze_backbone,
        'n_folds': args.n_folds,
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'backbone_lr_scale': args.backbone_lr_scale,
        'warmup_epochs': args.warmup_epochs,
        'gradient_accumulation_steps': args.gradient_accumulation,
        'max_grad_norm': args.max_grad_norm,
        'weight_decay': 0.01,
        'dropout': 0.3,
        'optimizer': 'adamw',
        'scheduler': 'warmup_cosine',
        'early_stopping_patience': 10,
        'num_workers': 4,
        'seed': args.seed,
        'img_dir': str(img_dir),
    }

    # Create checkpoint directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f"{args.backbone}{'_depth' if args.use_depth else ''}"
    checkpoint_dir = Path(f'experiments/checkpoints_{model_name}_{timestamp}')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(checkpoint_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("=" * 70)
    print(f"Training {args.backbone.upper()} Model")
    print("=" * 70)
    print(f"\nConfig:")
    print(f"  Backbone: {backbone_config['model_name']}")
    print(f"  Image size: {backbone_config['image_size']}")
    print(f"  Use depth: {args.use_depth}")
    print(f"  Freeze backbone: {args.freeze_backbone}")
    print(f"  Folds: {args.n_folds}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Head LR: {args.lr}")
    print(f"  Backbone LR: {args.lr * args.backbone_lr_scale:.2e}")
    print(f"  Warmup epochs: {args.warmup_epochs}")
    print(f"  Gradient accumulation: {args.gradient_accumulation}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation}")
    print(f"  Max grad norm: {args.max_grad_norm}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print(f"\nLoading data from {train_csv}...")
    df = pd.read_csv(train_csv)
    df['image_id'] = df['sample_id'].str.split('__').str[0]

    # Convert to wide format (one row per image with all targets)
    image_df = df.pivot_table(
        index=['image_id', 'image_path', 'Sampling_Date', 'State'],
        columns='target_name',
        values='target'
    ).reset_index()

    print(f"Total images: {len(image_df)}")

    # K-fold cross-validation
    print(f"\n{'='*70}")
    print(f"Starting {args.n_folds}-Fold Cross-Validation")
    print(f"{'='*70}")

    fold_results = []

    for fold_idx, (train_df, val_df) in get_group_kfold_splits(
        image_df, n_folds=args.n_folds, group_by='location', random_seed=args.seed
    ):
        result = train_fold(
            fold_idx=fold_idx,
            train_df=train_df,
            val_df=val_df,
            config=config,
            checkpoint_dir=checkpoint_dir,
            device=device,
        )
        fold_results.append(result)

        # Clear GPU memory
        torch.cuda.empty_cache()

    # Summary
    val_losses = [r['best_val_loss'] for r in fold_results]
    mean_loss = np.mean(val_losses)
    std_loss = np.std(val_losses)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nResults by fold:")
    for r in fold_results:
        print(f"  Fold {r['fold'] + 1}: Val Loss = {r['best_val_loss']:.4f}")
    print(f"\nMean Val Loss: {mean_loss:.4f} +/- {std_loss:.4f}")
    print(f"\nCheckpoints: {checkpoint_dir}")

    # Save results
    results = {
        'n_folds': args.n_folds,
        'backbone': args.backbone,
        'use_depth': args.use_depth,
        'mean_val_loss': mean_loss,
        'std_val_loss': std_loss,
        'fold_results': fold_results,
        'config': config,
    }

    with open(checkpoint_dir / 'kfold_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {checkpoint_dir / 'kfold_results.json'}")


if __name__ == '__main__':
    main()
