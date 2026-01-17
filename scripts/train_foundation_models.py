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
    'dinov2_large': {
        'model_name': 'vit_large_patch14_dinov2',
        'image_size': 518,
        'features': 1024,
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


class FoundationModelWithDepthAttention(nn.Module):
    """Foundation model + depth with attention-based fusion.

    Key innovation: Target-specific attention allows each biomass target
    to learn its own weighting of RGB vs depth features. For example:
    - Dry_Dead_g might rely more on RGB color (brown/yellow)
    - Dry_Total_g might rely more on depth (vegetation height)
    """

    def __init__(
        self,
        backbone_name: str = 'vit_base_patch14_dinov2',
        num_features: int = 768,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
        fusion_dim: int = 512,
        num_attention_heads: int = 8,
    ):
        super().__init__()

        self.target_names = TARGET_NAMES
        self.fusion_dim = fusion_dim

        # RGB backbone
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

        # Depth encoder (same as before)
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

        # Project both modalities to same dimension for attention
        self.rgb_proj = nn.Sequential(
            nn.Linear(num_features, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )
        self.depth_proj = nn.Sequential(
            nn.Linear(256, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

        # Cross-modal attention: learns interactions between RGB and depth
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attention_norm = nn.LayerNorm(fusion_dim)

        # Gating mechanism: learns per-sample weighting of RGB vs depth
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, 2),
            nn.Softmax(dim=-1)
        )

        # Target-specific regression heads with their own feature refinement
        self.heads = nn.ModuleDict()
        for name in self.target_names:
            self.heads[name] = nn.Sequential(
                nn.Linear(fusion_dim, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            )

    def forward(self, x):
        # Extract RGB features from backbone
        rgb_features = self.backbone(x)  # (B, 768)

        # Extract depth features
        with torch.no_grad():
            depth_maps = self.depth_estimator(x)
        depth_features = self.depth_encoder(depth_maps)  # (B, 256)

        # Project to common dimension
        rgb_proj = self.rgb_proj(rgb_features)  # (B, fusion_dim)
        depth_proj = self.depth_proj(depth_features)  # (B, fusion_dim)

        # Stack as sequence for attention: (B, 2, fusion_dim)
        # Position 0 = RGB, Position 1 = Depth
        combined = torch.stack([rgb_proj, depth_proj], dim=1)

        # Cross-modal attention with residual
        attended, _ = self.cross_attention(combined, combined, combined)
        attended = self.attention_norm(attended + combined)  # (B, 2, fusion_dim)

        # Extract attended features
        rgb_attended = attended[:, 0, :]  # (B, fusion_dim)
        depth_attended = attended[:, 1, :]  # (B, fusion_dim)

        # Compute adaptive gate weights
        gate_input = torch.cat([rgb_attended, depth_attended], dim=1)  # (B, fusion_dim*2)
        gate_weights = self.gate(gate_input)  # (B, 2)

        # Gated fusion
        fused = gate_weights[:, 0:1] * rgb_attended + gate_weights[:, 1:2] * depth_attended

        # Predict each target
        return {name: self.heads[name](fused).squeeze(-1) for name in self.target_names}


class FoundationModelWithDepthAndVegIndices(nn.Module):
    """Foundation model + depth + vegetation indices fusion."""

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

        # Vegetation indices encoder (6 channels -> 64 features)
        from src.data.vegetation_indices import N_VEGETATION_INDICES
        self.veg_encoder = nn.Sequential(
            nn.Conv2d(N_VEGETATION_INDICES, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # RGB (768) + Depth (256) + VegIndices (64) = 1088
        fused_features = num_features + 256 + 64

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
        # RGB features from backbone
        rgb_features = self.backbone(x)

        # Depth features
        with torch.no_grad():
            depth_maps = self.depth_estimator(x)
        depth_features = self.depth_encoder(depth_maps)

        # Vegetation indices features
        from src.data.vegetation_indices import compute_vegetation_indices_torch
        veg_indices = compute_vegetation_indices_torch(x)  # (B, 6, H, W)
        veg_features = self.veg_encoder(veg_indices)

        # Fuse all features
        fused = torch.cat([rgb_features, depth_features, veg_features], dim=1)

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
    print(f"  Use attention: {config.get('use_attention', False)}")
    print(f"  Use veg indices: {config.get('use_veg_indices', False)}")

    if config['use_depth'] and config.get('use_veg_indices', False):
        model = FoundationModelWithDepthAndVegIndices(
            backbone_name=backbone_config['model_name'],
            num_features=backbone_config['features'],
            dropout=config['dropout'],
            freeze_backbone=config['freeze_backbone'],
        )
    elif config['use_depth'] and config.get('use_attention', False):
        model = FoundationModelWithDepthAttention(
            backbone_name=backbone_config['model_name'],
            num_features=backbone_config['features'],
            dropout=config['dropout'],
            freeze_backbone=config['freeze_backbone'],
        )
    elif config['use_depth']:
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

    # Get weighted R² from best epoch (last in history after loading best model)
    best_weighted_r2 = trainer.history['val_metrics'][-1].get('weighted_R2', 0.0)

    return {
        'fold': fold_idx,
        'best_val_loss': best_val_loss,
        'best_weighted_r2': best_weighted_r2,
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
    parser.add_argument('--stratified', action='store_true',
                        help='Use stratified group k-fold splits')
    parser.add_argument('--use_external', action='store_true',
                        help='Include external GrassClover data')
    parser.add_argument('--use_pseudo', action='store_true',
                        help='Include pseudo-labeled data (requires --use_external)')
    parser.add_argument('--use_veg_indices', action='store_true',
                        help='Add vegetation indices as features')
    parser.add_argument('--use_attention', action='store_true',
                        help='Use attention-based fusion instead of concatenation')
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
        'features': backbone_config['features'],
        'use_depth': args.use_depth,
        'use_veg_indices': args.use_veg_indices,
        'use_attention': args.use_attention,
        'freeze_backbone': args.freeze_backbone,
        'stratified': args.stratified,
        'use_external': args.use_external,
        'use_pseudo': args.use_pseudo,
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
    suffix_parts = [args.backbone]
    if args.use_depth:
        suffix_parts.append('depth')
    if args.use_attention:
        suffix_parts.append('attn')
    if args.use_veg_indices:
        suffix_parts.append('veg')
    if args.stratified:
        suffix_parts.append('strat')
    if args.use_external:
        suffix_parts.append('ext')
    if args.use_pseudo:
        suffix_parts.append('pseudo')
    model_name = '_'.join(suffix_parts)
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
    print(f"  Use attention fusion: {args.use_attention}")
    print(f"  Use veg indices: {args.use_veg_indices}")
    print(f"  Freeze backbone: {args.freeze_backbone}")
    print(f"  Stratified splits: {args.stratified}")
    print(f"  External data: {args.use_external}")
    print(f"  Pseudo-labeled data: {args.use_pseudo}")
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
        index=['image_id', 'image_path', 'Sampling_Date', 'State', 'Species',
               'Pre_GSHH_NDVI', 'Height_Ave_cm'],
        columns='target_name',
        values='target',
        aggfunc='first'
    ).reset_index()

    print(f"Competition images: {len(image_df)}")

    # Add external data if requested
    if args.use_external:
        external_csv = Path('/home/chaot/kaggle/Image2Biomass-Competition/external_data/processed/grassclover_long.csv')
        if external_csv.exists():
            print(f"Loading external data from {external_csv}...")
            ext_df = pd.read_csv(external_csv)
            ext_df['image_id'] = ext_df['sample_id'].str.split('__').str[0]
            ext_image_df = ext_df.pivot_table(
                index=['image_id', 'image_path', 'Sampling_Date', 'State', 'Species',
                       'Pre_GSHH_NDVI', 'Height_Ave_cm'],
                columns='target_name',
                values='target',
                aggfunc='first'
            ).reset_index()
            print(f"External labeled images: {len(ext_image_df)}")
            image_df = pd.concat([image_df, ext_image_df], ignore_index=True)
        else:
            print(f"Warning: External data not found at {external_csv}")

        # Add pseudo-labeled data if available
        pseudo_csv = Path('/home/chaot/kaggle/Image2Biomass-Competition/external_data/processed/pseudo_labels_long.csv')
        if pseudo_csv.exists() and args.use_pseudo:
            print(f"Loading pseudo-labeled data from {pseudo_csv}...")
            pseudo_df = pd.read_csv(pseudo_csv)
            pseudo_df['image_id'] = pseudo_df['sample_id'].str.split('__').str[0]
            pseudo_image_df = pseudo_df.pivot_table(
                index=['image_id', 'image_path', 'Sampling_Date', 'State', 'Species',
                       'Pre_GSHH_NDVI', 'Height_Ave_cm'],
                columns='target_name',
                values='target',
                aggfunc='first'
            ).reset_index()
            print(f"Pseudo-labeled images: {len(pseudo_image_df)}")
            image_df = pd.concat([image_df, pseudo_image_df], ignore_index=True)
        elif args.use_pseudo:
            print(f"Warning: Pseudo-labeled data not found at {pseudo_csv}")

    print(f"Total images: {len(image_df)}")

    # K-fold cross-validation
    print(f"\n{'='*70}")
    print(f"Starting {args.n_folds}-Fold Cross-Validation")
    if args.stratified:
        print("Using STRATIFIED GroupKFold splits")
    print(f"{'='*70}")

    fold_results = []

    # Choose splitter based on stratified flag
    if args.stratified:
        from src.data.stratified_splitter import get_stratified_group_kfold_splits
        split_generator = get_stratified_group_kfold_splits(
            image_df, img_dir, n_folds=args.n_folds, random_seed=args.seed
        )
    else:
        split_generator = get_group_kfold_splits(
            image_df, n_folds=args.n_folds, group_by='location', random_seed=args.seed
        )

    for fold_idx, (train_df, val_df) in split_generator:
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
    weighted_r2s = [r.get('best_weighted_r2', 0.0) for r in fold_results]
    mean_loss = np.mean(val_losses)
    std_loss = np.std(val_losses)
    mean_r2 = np.mean(weighted_r2s)
    std_r2 = np.std(weighted_r2s)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nResults by fold:")
    for r in fold_results:
        r2 = r.get('best_weighted_r2', 0.0)
        print(f"  Fold {r['fold'] + 1}: Val Loss = {r['best_val_loss']:.4f}, Weighted R2 = {r2:.4f}")
    print(f"\nMean Val Loss: {mean_loss:.4f} +/- {std_loss:.4f}")
    print(f"Mean Weighted R2 (Kaggle): {mean_r2:.4f} +/- {std_r2:.4f}")
    print(f"\nCheckpoints: {checkpoint_dir}")

    # Save results
    results = {
        'n_folds': args.n_folds,
        'backbone': args.backbone,
        'use_depth': args.use_depth,
        'mean_val_loss': mean_loss,
        'std_val_loss': std_loss,
        'mean_weighted_r2': mean_r2,
        'std_weighted_r2': std_r2,
        'fold_results': fold_results,
        'config': config,
    }

    with open(checkpoint_dir / 'kfold_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {checkpoint_dir / 'kfold_results.json'}")


if __name__ == '__main__':
    main()
