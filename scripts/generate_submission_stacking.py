#!/usr/bin/env python3
"""
Generate submission using stacking ensemble.

Usage:
    python scripts/generate_submission_stacking.py
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
import pickle
import json

from src.models.depth_encoder import DepthEstimator


TARGET_NAMES = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']


# ============================================================================
# Model Definitions
# ============================================================================

class MultiTaskEfficientNet(nn.Module):
    """V4 architecture."""

    def __init__(self, backbone_name='tf_efficientnetv2_m', dropout=0.5, head_hidden_dim=512):
        super().__init__()
        self.target_names = TARGET_NAMES

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        num_features = self.backbone.num_features

        self.heads = nn.ModuleDict()
        for name in self.target_names:
            self.heads[name] = nn.Sequential(
                nn.Linear(num_features, head_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, 1)
            )

    def forward(self, x):
        features = self.backbone(x)
        return {name: self.heads[name](features).squeeze(-1) for name in self.target_names}


class FoundationModel(nn.Module):
    """V7 architecture."""

    def __init__(self, backbone_name='vit_base_patch14_dinov2', num_features=768, dropout=0.3):
        super().__init__()
        self.target_names = TARGET_NAMES

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
        )

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
    """V8 architecture."""

    def __init__(self, backbone_name='vit_base_patch14_dinov2', num_features=768, dropout=0.3):
        super().__init__()
        self.target_names = TARGET_NAMES

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
        )

        self.depth_estimator = DepthEstimator(
            model_type='depth_anything_v2_small',
            freeze=True
        )

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
        rgb_features = self.backbone(x)

        with torch.no_grad():
            depth_maps = self.depth_estimator(x)
        depth_features = self.depth_encoder(depth_maps)

        fused = torch.cat([rgb_features, depth_features], dim=1)

        return {name: self.heads[name](fused).squeeze(-1) for name in self.target_names}


# ============================================================================
# Helper Functions
# ============================================================================

def get_transform(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def load_image(image_path, transform, device):
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    transformed = transform(image=image_np)
    return transformed['image'].unsqueeze(0).to(device)


def load_v4_model(checkpoint_path, device):
    model = MultiTaskEfficientNet(backbone_name='tf_efficientnetv2_m', dropout=0.5, head_hidden_dim=512)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def load_v7_model(checkpoint_path, device):
    model = FoundationModel(backbone_name='vit_base_patch14_dinov2', num_features=768, dropout=0.3)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def load_v8_model(checkpoint_path, device):
    model = FoundationModelWithDepth(backbone_name='vit_base_patch14_dinov2', num_features=768, dropout=0.3)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def predict_with_model(model, image_tensor, device):
    with torch.no_grad():
        outputs = model(image_tensor)
        return {name: outputs[name].cpu().numpy()[0] for name in TARGET_NAMES}


# ============================================================================
# Main
# ============================================================================

def main():
    # Paths
    stacking_dir = Path('experiments/stacking_ensemble')
    test_csv = Path('test.csv')
    test_dir = Path('test')
    output_path = Path('submission_stacking.csv')

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load meta-learners
    print("\nLoading meta-learners...")
    with open(stacking_dir / 'meta_learners.pkl', 'rb') as f:
        meta_learners = pickle.load(f)

    # Load config
    with open(stacking_dir / 'config.json', 'r') as f:
        config = json.load(f)

    model_names = config['model_names']
    print(f"Models: {model_names}")

    # Model configurations
    model_configs = {
        'V4_EfficientNet': {
            'checkpoint_dir': Path('experiments/checkpoints_kfold_mse'),
            'image_size': 512,
            'load_fn': load_v4_model,
        },
        'V7_DINOv2': {
            'checkpoint_dir': Path('experiments/checkpoints_dinov2_base_20260116_100625'),
            'image_size': 518,
            'load_fn': load_v7_model,
        },
        'V8_DINOv2_Depth': {
            'checkpoint_dir': Path('experiments/checkpoints_dinov2_base_depth_20260116_140724'),
            'image_size': 518,
            'load_fn': load_v8_model,
        },
    }

    # Load test data
    print("\nLoading test data...")
    test_df = pd.read_csv(test_csv)
    test_df['image_id'] = test_df['sample_id'].str.split('__').str[0]
    test_image_ids = test_df['image_id'].unique()
    print(f"Test images: {len(test_image_ids)}")

    # Generate predictions for each model (average across folds)
    print("\nGenerating test predictions for each model...")
    test_predictions = {}

    for model_name in model_names:
        cfg = model_configs[model_name]
        print(f"\n{model_name}:")

        transform = get_transform(cfg['image_size'])
        model_preds = {target: np.zeros(len(test_image_ids)) for target in TARGET_NAMES}
        n_models = 0

        for fold_idx in range(5):
            checkpoint_path = cfg['checkpoint_dir'] / f'fold_{fold_idx}' / 'best_model.pth'

            if not checkpoint_path.exists():
                print(f"  Fold {fold_idx}: Not found")
                continue

            model = cfg['load_fn'](checkpoint_path, device)
            print(f"  Fold {fold_idx}: Loaded")
            n_models += 1

            for idx, image_id in enumerate(test_image_ids):
                image_path = test_dir / f"{image_id}.jpg"
                image_tensor = load_image(image_path, transform, device)
                preds = predict_with_model(model, image_tensor, device)

                for target in TARGET_NAMES:
                    model_preds[target][idx] += preds[target]

            del model
            torch.cuda.empty_cache()

        # Average across folds
        for target in TARGET_NAMES:
            model_preds[target] /= n_models

        test_predictions[model_name] = model_preds
        print(f"  Averaged {n_models} folds")

    # Apply meta-learner
    print("\nApplying meta-learner...")
    final_predictions = {}

    for target in TARGET_NAMES:
        X = np.column_stack([
            test_predictions[model_name][target]
            for model_name in model_names
        ])
        final_predictions[target] = meta_learners[target].predict(X)

    # Enforce constraints
    print("\nEnforcing biological constraints...")
    for idx in range(len(test_image_ids)):
        green = final_predictions['Dry_Green_g'][idx]
        clover = final_predictions['Dry_Clover_g'][idx]
        dead = final_predictions['Dry_Dead_g'][idx]
        gdm = final_predictions['GDM_g'][idx]
        total = final_predictions['Dry_Total_g'][idx]

        # Ensure non-negative
        green = max(0, green)
        clover = max(0, clover)
        dead = max(0, dead)

        # Soft constraint: GDM = Green + Clover
        expected_gdm = green + clover
        adjusted_gdm = (gdm + expected_gdm) / 2

        # Soft constraint: Total = GDM + Dead
        expected_total = adjusted_gdm + dead
        adjusted_total = (total + expected_total) / 2

        final_predictions['Dry_Green_g'][idx] = green
        final_predictions['Dry_Clover_g'][idx] = clover
        final_predictions['Dry_Dead_g'][idx] = dead
        final_predictions['GDM_g'][idx] = max(0, adjusted_gdm)
        final_predictions['Dry_Total_g'][idx] = max(0, adjusted_total)

    # Create submission
    print("\nCreating submission...")
    submissions = []

    for idx, image_id in enumerate(test_image_ids):
        for target in TARGET_NAMES:
            sample_id = f"{image_id}__{target}"
            value = final_predictions[target][idx]
            submissions.append({
                'sample_id': sample_id,
                'target': value
            })

    submission_df = pd.DataFrame(submissions)
    submission_df.to_csv(output_path, index=False)

    print(f"\nSubmission saved to: {output_path}")
    print(f"Total rows: {len(submission_df)}")

    # Print prediction statistics
    print("\nPrediction statistics:")
    for target in TARGET_NAMES:
        values = final_predictions[target]
        print(f"  {target}: mean={values.mean():.2f}, std={values.std():.2f}, "
              f"min={values.min():.2f}, max={values.max():.2f}")


if __name__ == '__main__':
    main()
