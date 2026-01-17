#!/usr/bin/env python3
"""
Generate pseudo-labels for unlabeled external data using V8 model.

This script:
1. Loads all 5 folds of the V8 (DINOv2 + Depth) model
2. Runs inference on unlabeled ailabdatasets test images
3. Averages predictions across folds
4. Saves pseudo-labels in competition format

Usage:
    python scripts/generate_pseudo_labels.py
"""

import sys
sys.path.insert(0, '/home/chaot/kaggle/Image2Biomass-Competition')

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.models.depth_encoder import DepthEstimator


TARGET_NAMES = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']


class FoundationModelWithDepth(nn.Module):
    """Foundation model + depth fusion (V8 architecture)."""

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


def get_inference_transform(image_size=518):
    """Get inference transform matching V8 training."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def load_models(checkpoint_dir: Path, device: torch.device):
    """Load all 5 fold models."""
    models = []

    for fold_idx in range(5):
        fold_dir = checkpoint_dir / f'fold_{fold_idx}'
        checkpoint_path = fold_dir / 'best_model.pth'

        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            continue

        # Create model
        model = FoundationModelWithDepth(
            backbone_name='vit_base_patch14_dinov2',
            num_features=768,
            dropout=0.3,
        )

        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        models.append(model)
        print(f"Loaded fold {fold_idx} from {checkpoint_path}")

    return models


def predict_single_image(models, image_path: Path, transform, device: torch.device):
    """Run inference on a single image with all fold models."""
    # Load image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)

    # Transform
    transformed = transform(image=image_np)
    image_tensor = transformed['image'].unsqueeze(0).to(device)

    # Predict with each fold
    all_predictions = {name: [] for name in TARGET_NAMES}

    with torch.no_grad():
        for model in models:
            outputs = model(image_tensor)
            for name in TARGET_NAMES:
                all_predictions[name].append(outputs[name].cpu().numpy()[0])

    # Average across folds
    avg_predictions = {}
    for name in TARGET_NAMES:
        avg_predictions[name] = np.mean(all_predictions[name])

    return avg_predictions


def load_target_stats():
    """Load target statistics from competition training data for denormalization."""
    df = pd.read_csv('train.csv')

    stats = {}
    for target in TARGET_NAMES:
        target_df = df[df['target_name'] == target]
        stats[target] = {
            'mean': target_df['target'].mean(),
            'std': target_df['target'].std()
        }
    return stats


def denormalize_predictions(predictions, target_stats):
    """Denormalize predictions using competition data statistics."""
    denorm = {}
    for name in TARGET_NAMES:
        # Model outputs are normalized: (x - mean) / std
        # To denormalize: x * std + mean
        denorm[name] = predictions[name] * target_stats[name]['std'] + target_stats[name]['mean']
    return denorm


def main():
    # Paths
    checkpoint_dir = Path('experiments/checkpoints_dinov2_base_depth_20260116_140724')
    unlabeled_dir = Path('external_data/ailabdatasets_full/test/images')
    output_dir = Path('external_data/processed')
    output_dir.mkdir(exist_ok=True, parents=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load target statistics for denormalization
    print("\nLoading target statistics...")
    target_stats = load_target_stats()
    for name, stats in target_stats.items():
        print(f"  {name}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")

    # Load models
    print("\nLoading V8 models...")
    models = load_models(checkpoint_dir, device)
    print(f"Loaded {len(models)} fold models")

    # Get transform
    transform = get_inference_transform(image_size=518)

    # Get list of unlabeled images
    image_paths = sorted(unlabeled_dir.glob('*.jpg'))
    print(f"\nFound {len(image_paths)} unlabeled images")

    # Generate predictions
    print("\nGenerating pseudo-labels...")
    results = []

    for image_path in tqdm(image_paths):
        predictions = predict_single_image(models, image_path, transform, device)

        # Denormalize predictions to real scale
        predictions = denormalize_predictions(predictions, target_stats)

        # Ensure non-negative predictions
        for name in TARGET_NAMES:
            predictions[name] = max(0, predictions[name])

        # Create row
        image_id = image_path.stem
        results.append({
            'image_id': image_id,
            'image_path': f'external_data/ailabdatasets_full/test/images/{image_path.name}',
            'Sampling_Date': '2017/1/1',  # Placeholder
            'State': 'Denmark_PseudoLabel',
            'Species': 'GrassClover',
            'Pre_GSHH_NDVI': 0.6,  # Placeholder
            'Height_Ave_cm': 5.0,  # Placeholder
            **predictions
        })

    # Create DataFrame
    df_wide = pd.DataFrame(results)

    print(f"\nGenerated pseudo-labels for {len(df_wide)} images")
    print(f"\nPseudo-label statistics:")
    for col in TARGET_NAMES:
        print(f"  {col:<15} mean: {df_wide[col].mean():>8.2f}  "
              f"std: {df_wide[col].std():>8.2f}  "
              f"min: {df_wide[col].min():>8.2f}  "
              f"max: {df_wide[col].max():>8.2f}")

    # Save wide format
    wide_csv_path = output_dir / 'pseudo_labels_wide.csv'
    df_wide.to_csv(wide_csv_path, index=False)
    print(f"\nSaved wide format to: {wide_csv_path}")

    # Create long format
    long_rows = []
    for _, row in df_wide.iterrows():
        for target_name in TARGET_NAMES:
            long_rows.append({
                'sample_id': f"{row['image_id']}__{target_name}",
                'image_id': row['image_id'],
                'image_path': row['image_path'],
                'Sampling_Date': row['Sampling_Date'],
                'State': row['State'],
                'Species': row['Species'],
                'Pre_GSHH_NDVI': row['Pre_GSHH_NDVI'],
                'Height_Ave_cm': row['Height_Ave_cm'],
                'target_name': target_name,
                'target': row[target_name]
            })

    df_long = pd.DataFrame(long_rows)
    long_csv_path = output_dir / 'pseudo_labels_long.csv'
    df_long.to_csv(long_csv_path, index=False)
    print(f"Saved long format to: {long_csv_path}")

    # Create combined dataset (competition + external + pseudo)
    print("\n" + "=" * 70)
    print("Creating Combined Dataset with Pseudo-Labels")
    print("=" * 70)

    # Load existing data
    comp_df = pd.read_csv('train.csv')
    ext_df = pd.read_csv('external_data/processed/grassclover_long.csv')

    print(f"Competition data: {len(comp_df)} rows ({len(comp_df)//5} images)")
    print(f"External labeled data: {len(ext_df)} rows ({len(ext_df)//5} images)")
    print(f"Pseudo-labeled data: {len(df_long)} rows ({len(df_long)//5} images)")

    # Combine all
    combined_df = pd.concat([comp_df, ext_df, df_long], ignore_index=True)
    combined_csv_path = output_dir / 'combined_with_pseudo_train.csv'
    combined_df.to_csv(combined_csv_path, index=False)

    total_images = len(combined_df) // 5
    print(f"\nCombined data: {len(combined_df)} rows ({total_images} images)")
    print(f"Saved to: {combined_csv_path}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Competition images: 357")
    print(f"  External labeled images: 261")
    print(f"  Pseudo-labeled images: {len(df_wide)}")
    print(f"  Total: {total_images} images")
    print(f"\nReady for training with pseudo-labels!")
    print("=" * 70)


if __name__ == '__main__':
    main()
