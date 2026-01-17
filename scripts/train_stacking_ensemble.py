#!/usr/bin/env python3
"""
Stacking Ensemble with Meta-Learner for Biomass Prediction.

This script:
1. Loads multiple trained models (V4, V7, V8)
2. Generates out-of-fold (OOF) predictions on training data
3. Trains a meta-learner (Ridge/XGBoost) to combine predictions
4. Generates final ensemble predictions

Usage:
    python scripts/train_stacking_ensemble.py
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
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
import pickle
import json

from src.models.depth_encoder import DepthEstimator


TARGET_NAMES = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']


# ============================================================================
# Model Definitions (same as training scripts)
# ============================================================================

class MultiTaskEfficientNet(nn.Module):
    """V4 architecture - EfficientNetV2 multi-task model."""

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

        # Simpler head architecture matching original V4
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
    """V7 architecture - DINOv2 foundation model."""

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
    """V8 architecture - DINOv2 + Depth fusion."""

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
# Data Loading
# ============================================================================

def get_transform(image_size):
    """Get inference transform."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def load_image(image_path, transform, device):
    """Load and transform a single image."""
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    transformed = transform(image=image_np)
    return transformed['image'].unsqueeze(0).to(device)


# ============================================================================
# Model Loading
# ============================================================================

def load_v4_model(checkpoint_path, device):
    """Load V4 EfficientNetV2 model."""
    model = MultiTaskEfficientNet(backbone_name='tf_efficientnetv2_m', dropout=0.5, head_hidden_dim=512)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def load_v7_model(checkpoint_path, device):
    """Load V7 DINOv2 model."""
    model = FoundationModel(backbone_name='vit_base_patch14_dinov2', num_features=768, dropout=0.3)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def load_v8_model(checkpoint_path, device):
    """Load V8 DINOv2 + Depth model."""
    model = FoundationModelWithDepth(backbone_name='vit_base_patch14_dinov2', num_features=768, dropout=0.3)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


# ============================================================================
# Prediction Functions
# ============================================================================

def predict_with_model(model, image_tensor, device):
    """Get predictions from a single model."""
    with torch.no_grad():
        outputs = model(image_tensor)
        return {name: outputs[name].cpu().numpy()[0] for name in TARGET_NAMES}


def generate_oof_predictions(
    model_configs,
    df,
    img_dir,
    device,
    n_folds=5
):
    """
    Generate out-of-fold predictions for stacking.

    For each model, load each fold and predict on that fold's validation set.
    This ensures OOF predictions (no data leakage).
    """
    oof_predictions = {name: [] for name in model_configs.keys()}
    oof_targets = []
    oof_image_ids = []

    # Create fold assignments
    groups = df['State'].astype(str) + '_' + df['Sampling_Date'].astype(str)
    gkf = GroupKFold(n_splits=n_folds)
    fold_indices = list(gkf.split(df, groups=groups))

    print(f"\nGenerating OOF predictions for {len(model_configs)} models...")

    for model_name, config in model_configs.items():
        print(f"\n{model_name}:")
        model_preds = {target: np.zeros(len(df)) for target in TARGET_NAMES}

        transform = get_transform(config['image_size'])

        for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
            checkpoint_path = config['checkpoint_dir'] / f'fold_{fold_idx}' / 'best_model.pth'

            if not checkpoint_path.exists():
                print(f"  Fold {fold_idx}: Checkpoint not found, skipping")
                continue

            # Load model for this fold
            model = config['load_fn'](checkpoint_path, device)
            print(f"  Fold {fold_idx}: Loaded, predicting on {len(val_idx)} samples...")

            # Predict on validation set for this fold
            val_df = df.iloc[val_idx]

            for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc=f"  Fold {fold_idx}", leave=False):
                image_path = img_dir / f"{row['image_id']}.jpg"
                if not image_path.exists():
                    # Try external data path
                    if 'image_path' in row and pd.notna(row['image_path']):
                        image_path = Path(row['image_path'])

                if not image_path.exists():
                    continue

                image_tensor = load_image(image_path, transform, device)
                preds = predict_with_model(model, image_tensor, device)

                for target in TARGET_NAMES:
                    model_preds[target][idx] = preds[target]

            del model
            torch.cuda.empty_cache()

        oof_predictions[model_name] = model_preds

    return oof_predictions, fold_indices


def generate_test_predictions(model_configs, test_df, img_dir, device):
    """
    Generate test predictions by averaging all folds of each model.
    """
    test_predictions = {}

    print(f"\nGenerating test predictions for {len(model_configs)} models...")

    for model_name, config in model_configs.items():
        print(f"\n{model_name}:")
        model_preds = {target: np.zeros(len(test_df)) for target in TARGET_NAMES}
        n_models = 0

        transform = get_transform(config['image_size'])

        # Load all fold models and average predictions
        for fold_idx in range(5):
            checkpoint_path = config['checkpoint_dir'] / f'fold_{fold_idx}' / 'best_model.pth'

            if not checkpoint_path.exists():
                continue

            model = config['load_fn'](checkpoint_path, device)
            print(f"  Fold {fold_idx}: Loaded")
            n_models += 1

            for idx, (_, row) in enumerate(test_df.iterrows()):
                image_path = img_dir / f"{row['image_id']}.jpg"
                if not image_path.exists():
                    image_path = Path(f"test/{row['image_id']}.jpg")

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

    return test_predictions


# ============================================================================
# Meta-Learner
# ============================================================================

def train_meta_learner(oof_predictions, df, model_names):
    """
    Train a Ridge regression meta-learner for each target.

    Features: OOF predictions from each base model
    Target: Actual values
    """
    meta_learners = {}

    print("\nTraining meta-learners...")

    for target in TARGET_NAMES:
        # Build feature matrix: predictions from each model
        X = np.column_stack([
            oof_predictions[model_name][target]
            for model_name in model_names
        ])
        y = df[target].values

        # Remove samples with missing predictions (zeros)
        mask = X.sum(axis=1) != 0
        X_valid = X[mask]
        y_valid = y[mask]

        # Train Ridge regression
        meta = Ridge(alpha=1.0)
        meta.fit(X_valid, y_valid)

        # Print coefficients (model weights)
        print(f"\n{target}:")
        for i, model_name in enumerate(model_names):
            print(f"  {model_name}: {meta.coef_[i]:.4f}")
        print(f"  Intercept: {meta.intercept_:.4f}")

        meta_learners[target] = meta

    return meta_learners


def predict_with_meta_learner(meta_learners, predictions, model_names):
    """
    Generate final predictions using meta-learner.
    """
    final_predictions = {}

    n_samples = len(list(predictions.values())[0][TARGET_NAMES[0]])

    for target in TARGET_NAMES:
        # Build feature matrix
        X = np.column_stack([
            predictions[model_name][target]
            for model_name in model_names
        ])

        # Predict
        final_predictions[target] = meta_learners[target].predict(X)

    return final_predictions


# ============================================================================
# Main
# ============================================================================

def main():
    # Paths
    img_dir = Path('train')
    output_dir = Path('experiments/stacking_ensemble')
    output_dir.mkdir(exist_ok=True, parents=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

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

    # Load training data
    print("\nLoading training data...")
    df = pd.read_csv('train.csv')
    df['image_id'] = df['sample_id'].str.split('__').str[0]

    # Convert to wide format (one row per image)
    image_df = df.pivot_table(
        index=['image_id', 'image_path', 'Sampling_Date', 'State', 'Species',
               'Pre_GSHH_NDVI', 'Height_Ave_cm'],
        columns='target_name',
        values='target',
        aggfunc='first'
    ).reset_index()

    print(f"Training images: {len(image_df)}")

    # Generate OOF predictions
    oof_predictions, fold_indices = generate_oof_predictions(
        model_configs,
        image_df,
        img_dir,
        device,
        n_folds=5
    )

    # Save OOF predictions
    oof_path = output_dir / 'oof_predictions.pkl'
    with open(oof_path, 'wb') as f:
        pickle.dump(oof_predictions, f)
    print(f"\nSaved OOF predictions to {oof_path}")

    # Train meta-learner
    model_names = list(model_configs.keys())
    meta_learners = train_meta_learner(oof_predictions, image_df, model_names)

    # Save meta-learners
    meta_path = output_dir / 'meta_learners.pkl'
    with open(meta_path, 'wb') as f:
        pickle.dump(meta_learners, f)
    print(f"Saved meta-learners to {meta_path}")

    # Evaluate on training set (OOF)
    print("\n" + "=" * 70)
    print("OOF Evaluation (Stacked Ensemble)")
    print("=" * 70)

    oof_final = predict_with_meta_learner(meta_learners, oof_predictions, model_names)

    for target in TARGET_NAMES:
        # Get valid samples (non-zero predictions)
        mask = np.array([
            oof_predictions[model_names[0]][target][i] != 0
            for i in range(len(image_df))
        ])

        y_true = image_df[target].values[mask]
        y_pred = oof_final[target][mask]

        # Calculate metrics
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        mae = np.mean(np.abs(y_true - y_pred))

        print(f"{target}: R2 = {r2:.4f}, MAE = {mae:.4f}")

    # Save configuration
    config = {
        'model_names': model_names,
        'model_configs': {
            name: {
                'checkpoint_dir': str(cfg['checkpoint_dir']),
                'image_size': cfg['image_size']
            }
            for name, cfg in model_configs.items()
        }
    }

    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nStacking ensemble trained and saved to {output_dir}")
    print("\nTo generate test predictions, run:")
    print("  python scripts/generate_submission_stacking.py")


if __name__ == '__main__':
    main()
