"""
Stratified split utilities for creating train/val splits.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Tuple

from src.config import TARGET_NAMES


def create_stratified_split(
    df: pd.DataFrame,
    stratify_by: str = 'state',
    val_split: float = 0.2,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/validation split.

    Args:
        df: DataFrame in wide format (one row per image)
        stratify_by: Stratification strategy:
            - 'state': Stratify by State column
            - 'species': Stratify by Species column
            - 'target_bin': Stratify by binned Dry_Total_g
            - 'combined': Stratify by State_TargetBin
        val_split: Fraction of data for validation (0 < val_split < 1)
        random_seed: Random seed for reproducibility

    Returns:
        train_df, val_df (both in wide format)
    """
    # Create stratification labels
    if stratify_by == 'state':
        if 'State' not in df.columns:
            raise ValueError("State column not found in dataframe")
        stratify_labels = df['State']

    elif stratify_by == 'species':
        if 'Species' not in df.columns:
            raise ValueError("Species column not found in dataframe")
        stratify_labels = df['Species']

    elif stratify_by == 'target_bin':
        if 'Dry_Total_g' not in df.columns:
            raise ValueError("Dry_Total_g column not found in dataframe")
        # Bin targets into quartiles
        stratify_labels = pd.qcut(
            df['Dry_Total_g'],
            q=4,
            labels=['Q1', 'Q2', 'Q3', 'Q4'],
            duplicates='drop'
        )

    elif stratify_by == 'combined':
        if 'State' not in df.columns or 'Dry_Total_g' not in df.columns:
            raise ValueError("State and Dry_Total_g columns required for combined stratification")
        # Combine state and target bin
        target_bins = pd.qcut(
            df['Dry_Total_g'],
            q=4,
            labels=['Q1', 'Q2', 'Q3', 'Q4'],
            duplicates='drop'
        )
        stratify_labels = df['State'].astype(str) + '_' + target_bins.astype(str)

    else:
        raise ValueError(f"Unknown stratify_by: {stratify_by}")

    # Perform stratified split
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_split,
        random_state=random_seed
    )

    # Get train/val indices
    train_idx, val_idx = next(splitter.split(df, stratify_labels))

    # Split dataframe
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    print(f"\nStratified split by '{stratify_by}':")
    print(f"  Train size: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val size: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")

    # Print distribution stats
    if stratify_by in ['state', 'species']:
        col_name = 'State' if stratify_by == 'state' else 'Species'
        print(f"\nDistribution of {col_name}:")
        print(f"  Train: {dict(train_df[col_name].value_counts().to_dict())}")
        print(f"  Val: {dict(val_df[col_name].value_counts().to_dict())}")

    return train_df, val_df


def get_kfold_splits(
    df: pd.DataFrame,
    n_folds: int = 5,
    stratify_by: str = 'state',
    random_seed: int = 42
):
    """
    Create K-Fold stratified splits for cross-validation.

    Args:
        df: DataFrame in wide format (one row per image)
        n_folds: Number of folds
        stratify_by: Stratification strategy (same options as create_stratified_split)
        random_seed: Random seed for reproducibility

    Yields:
        fold_idx, (train_df, val_df) for each fold
    """
    from sklearn.model_selection import StratifiedKFold

    # Create stratification labels (same logic as create_stratified_split)
    if stratify_by == 'state':
        stratify_labels = df['State']
    elif stratify_by == 'species':
        stratify_labels = df['Species']
    elif stratify_by == 'target_bin':
        stratify_labels = pd.qcut(
            df['Dry_Total_g'],
            q=4,
            labels=['Q1', 'Q2', 'Q3', 'Q4'],
            duplicates='drop'
        )
    elif stratify_by == 'combined':
        target_bins = pd.qcut(
            df['Dry_Total_g'],
            q=4,
            labels=['Q1', 'Q2', 'Q3', 'Q4'],
            duplicates='drop'
        )
        stratify_labels = df['State'].astype(str) + '_' + target_bins.astype(str)
    else:
        raise ValueError(f"Unknown stratify_by: {stratify_by}")

    # Create K-Fold splitter
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    # Generate folds
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, stratify_labels)):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        print(f"\nFold {fold_idx + 1}/{n_folds}:")
        print(f"  Train size: {len(train_df)}")
        print(f"  Val size: {len(val_df)}")

        yield fold_idx, (train_df, val_df)
