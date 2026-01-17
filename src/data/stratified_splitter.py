"""
Stratified GroupKFold splitter with weighted multi-factor stratification.

Considers:
1. Target values (weighted by competition importance)
2. Image quality metrics (sharpness, brightness, contrast, noise)
3. Metadata (State, Species, NDVI, Height)
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Generator
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import KBinsDiscretizer
from tqdm import tqdm

from src.config import TARGET_NAMES


# Competition weights for targets
TARGET_WEIGHTS = {
    'Dry_Total_g': 0.50,
    'GDM_g': 0.20,
    'Dry_Green_g': 0.10,
    'Dry_Dead_g': 0.10,
    'Dry_Clover_g': 0.10,
}


def compute_image_metrics(img_path: Path) -> Dict[str, float]:
    """
    Compute image quality metrics.

    Returns:
        Dict with sharpness, brightness, contrast, green_ratio, noise
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return {
            'sharpness': 0.0,
            'brightness': 0.0,
            'contrast': 0.0,
            'green_ratio': 0.0,
            'noise': 0.0,
        }

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sharpness (Laplacian variance) - higher = sharper
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()

    # Brightness (mean intensity)
    brightness = gray.mean()

    # Contrast (std of intensity)
    contrast = gray.std()

    # Green ratio - vegetation indicator
    b, g, r = cv2.split(img)
    green_ratio = g.mean() / (r.mean() + b.mean() + 1e-8)

    # Noise estimate (high-freq residual)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = np.abs(gray.astype(float) - blur.astype(float)).mean()

    return {
        'sharpness': sharpness,
        'brightness': brightness,
        'contrast': contrast,
        'green_ratio': green_ratio,
        'noise': noise,
    }


def compute_weighted_stratification_label(
    df: pd.DataFrame,
    img_dir: Path,
    n_bins: int = 10,
    target_weight: float = 0.5,
    image_weight: float = 0.3,
    meta_weight: float = 0.2,
) -> np.ndarray:
    """
    Create weighted stratification labels combining multiple factors.

    Args:
        df: DataFrame in wide format with targets and metadata
        img_dir: Directory containing images
        n_bins: Number of bins for discretization
        target_weight: Weight for target-based stratification
        image_weight: Weight for image quality stratification
        meta_weight: Weight for metadata stratification

    Returns:
        Array of stratification labels (integers)
    """
    print("Computing stratification features...")

    # 1. Target-based score (weighted by competition weights)
    target_scores = np.zeros(len(df))
    for target, weight in TARGET_WEIGHTS.items():
        if target in df.columns:
            # Normalize to 0-1 range
            vals = df[target].values
            normalized = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
            target_scores += weight * normalized

    # 2. Image quality metrics
    print("  Extracting image metrics...")
    image_scores = np.zeros(len(df))

    metrics_list = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Images"):
        img_path = img_dir / Path(row['image_path']).name
        metrics = compute_image_metrics(img_path)
        metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)

    # Normalize each metric and combine
    for col in ['sharpness', 'brightness', 'contrast', 'green_ratio', 'noise']:
        vals = metrics_df[col].values
        normalized = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
        image_scores += normalized / 5  # Equal weight for each metric

    # 3. Metadata score
    meta_scores = np.zeros(len(df))

    # NDVI (normalized)
    if 'Pre_GSHH_NDVI' in df.columns:
        vals = df['Pre_GSHH_NDVI'].values
        meta_scores += 0.4 * (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)

    # Height (normalized)
    if 'Height_Ave_cm' in df.columns:
        vals = df['Height_Ave_cm'].values
        meta_scores += 0.4 * (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)

    # State encoding (simple hash)
    if 'State' in df.columns:
        state_map = {s: i for i, s in enumerate(df['State'].unique())}
        state_vals = df['State'].map(state_map).values
        meta_scores += 0.2 * state_vals / (len(state_map) - 1 + 1e-8)

    # Combine all scores with weights
    combined_scores = (
        target_weight * target_scores +
        image_weight * image_scores +
        meta_weight * meta_scores
    )

    # Discretize into bins
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    labels = discretizer.fit_transform(combined_scores.reshape(-1, 1)).astype(int).flatten()

    print(f"  Created {n_bins} stratification bins")
    print(f"  Bin distribution: {np.bincount(labels)}")

    return labels


def get_stratified_group_kfold_splits(
    df: pd.DataFrame,
    img_dir: Path,
    n_folds: int = 5,
    n_bins: int = 10,
    target_weight: float = 0.5,
    image_weight: float = 0.3,
    meta_weight: float = 0.2,
    random_seed: int = 42,
) -> Generator[Tuple[int, Tuple[pd.DataFrame, pd.DataFrame]], None, None]:
    """
    Create stratified GroupKFold splits with weighted multi-factor stratification.

    Groups by State + Sampling_Date (prevents data leakage).
    Stratifies by weighted combination of targets, image quality, and metadata.

    Args:
        df: DataFrame in wide format (one row per image)
        img_dir: Directory containing images
        n_folds: Number of folds
        n_bins: Number of stratification bins
        target_weight: Weight for target-based stratification (default 0.5)
        image_weight: Weight for image quality stratification (default 0.3)
        meta_weight: Weight for metadata stratification (default 0.2)
        random_seed: Random seed

    Yields:
        fold_idx, (train_df, val_df) for each fold
    """
    # Create group labels (same location = same group)
    groups = df['State'].astype(str) + '_' + df['Sampling_Date'].astype(str)

    # Create stratification labels
    strat_labels = compute_weighted_stratification_label(
        df, img_dir, n_bins, target_weight, image_weight, meta_weight
    )

    # Map stratification labels to groups (use mode of labels per group)
    group_strat = {}
    for group in groups.unique():
        mask = groups == group
        group_labels = strat_labels[mask]
        # Use mode (most common label in group)
        group_strat[group] = int(np.bincount(group_labels).argmax())

    group_strat_labels = np.array([group_strat[g] for g in groups])

    print(f"\nStratified GroupKFold:")
    print(f"  Total images: {len(df)}")
    print(f"  Unique groups: {groups.nunique()}")
    print(f"  Stratification bins: {n_bins}")
    print(f"  Weights: target={target_weight}, image={image_weight}, meta={meta_weight}")

    # Create StratifiedGroupKFold splitter
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    # Generate folds
    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(df, group_strat_labels, groups)):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        # Print fold statistics
        train_groups = groups.iloc[train_idx].nunique()
        val_groups = groups.iloc[val_idx].nunique()

        print(f"\nFold {fold_idx + 1}/{n_folds}:")
        print(f"  Train: {len(train_df)} images from {train_groups} groups")
        print(f"  Val: {len(val_df)} images from {val_groups} groups")

        # Print target distribution comparison
        for target in TARGET_NAMES:
            if target in df.columns:
                train_mean = train_df[target].mean()
                val_mean = val_df[target].mean()
                diff_pct = (val_mean - train_mean) / (train_mean + 1e-8) * 100
                print(f"    {target}: train={train_mean:.2f}, val={val_mean:.2f} ({diff_pct:+.1f}%)")

        yield fold_idx, (train_df, val_df)


if __name__ == '__main__':
    # Test the stratified splitter
    from src.config import DATA_DIR

    df = pd.read_csv(DATA_DIR / 'train.csv')
    df['image_id'] = df['sample_id'].str.split('__').str[0]
    df_wide = df.pivot_table(
        index=['image_id', 'image_path', 'Sampling_Date', 'State', 'Species',
               'Pre_GSHH_NDVI', 'Height_Ave_cm'],
        columns='target_name',
        values='target',
        aggfunc='first'
    ).reset_index()

    img_dir = DATA_DIR / 'train'

    print("Testing stratified group k-fold splits...")
    for fold_idx, (train_df, val_df) in get_stratified_group_kfold_splits(
        df_wide, img_dir, n_folds=5
    ):
        pass

    print("\nDone!")
