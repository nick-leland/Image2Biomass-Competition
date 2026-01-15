"""
Dataset class for Image2Biomass competition.

Handles loading images and targets from long-format CSV.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Optional, Dict, Any

from src.config import TARGET_NAMES, METADATA_COLS


class BiomassDataset(Dataset):
    """
    Dataset for multi-target biomass regression.

    Args:
        csv_path: Path to train.csv or test.csv
        img_dir: Directory containing images
        transform: Albumentations transform pipeline
        is_test: Whether this is test data (no targets)
        target_stats: Dict with 'mean' and 'std' for each target (for normalization)
    """

    def __init__(
        self,
        csv_path: Path,
        img_dir: Path,
        transform=None,
        is_test: bool = False,
        target_stats: Optional[Dict[str, Dict[str, float]]] = None
    ):
        self.csv_path = csv_path
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.is_test = is_test
        self.target_stats = target_stats

        # Extract image_size from transform (for TTA)
        self.image_size = self._extract_image_size(transform)

        # Load and process CSV
        self.df = pd.read_csv(csv_path)
        self._process_data()

    def _extract_image_size(self, transform):
        """Extract image size from albumentations transform."""
        if transform is None:
            return 512  # Default
        # Look for Resize transform in the pipeline
        if hasattr(transform, 'transforms'):
            for t in transform.transforms:
                if t.__class__.__name__ == 'Resize':
                    return t.height  # Assuming square images
        return 512  # Default if not found

    def _process_data(self):
        """Process long-format CSV to wide format (one row per image)."""
        if self.is_test:
            # Test data: extract image info
            self.df['image_id'] = self.df['sample_id'].str.split('__').str[0]
            self.image_ids = self.df['image_id'].unique()
            self.image_paths = self.df.groupby('image_id')['image_path'].first().to_dict()
        else:
            # Training data: pivot to wide format
            self.df['image_id'] = self.df['sample_id'].str.split('__').str[0]

            # Pivot targets
            targets_wide = self.df.pivot_table(
                index='image_id',
                columns='target_name',
                values='target',
                aggfunc='first'
            )

            # Get metadata (one row per image)
            metadata_df = self.df.groupby('image_id').first().reset_index()
            metadata_cols_present = [col for col in METADATA_COLS if col in metadata_df.columns]

            # Merge targets and metadata
            self.wide_df = targets_wide.merge(
                metadata_df[['image_id', 'image_path'] + metadata_cols_present],
                on='image_id',
                how='left'
            )

            self.image_ids = self.wide_df['image_id'].values
            self.image_paths = dict(zip(self.wide_df['image_id'], self.wide_df['image_path']))

            # Store targets
            self.targets = {
                target: self.wide_df[target].values.astype(np.float32)
                for target in TARGET_NAMES
            }

            # Store metadata
            self.metadata = {}
            for col in metadata_cols_present:
                self.metadata[col] = self.wide_df[col].values

            # Compute target statistics if not provided
            if self.target_stats is None:
                self.target_stats = self._compute_target_stats()

    def _compute_target_stats(self) -> Dict[str, Dict[str, float]]:
        """Compute mean and std for each target."""
        stats = {}
        for target_name in TARGET_NAMES:
            values = self.targets[target_name]
            stats[target_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)) + 1e-8  # Add small epsilon to avoid division by zero
            }
        return stats

    def normalize_targets(self, target_dict: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize target values using stored statistics.

        Args:
            target_dict: Dict of target values

        Returns:
            Normalized target dict
        """
        if self.target_stats is None:
            return target_dict

        normalized = {}
        for target_name, value in target_dict.items():
            stats = self.target_stats[target_name]
            normalized[target_name] = (value - stats['mean']) / stats['std']
        return normalized

    def denormalize_targets(self, target_dict: Dict[str, float]) -> Dict[str, float]:
        """
        Denormalize target values back to original scale.

        Args:
            target_dict: Dict of normalized target values

        Returns:
            Denormalized target dict
        """
        if self.target_stats is None:
            return target_dict

        denormalized = {}
        for target_name, value in target_dict.items():
            stats = self.target_stats[target_name]
            denormalized[target_name] = (value * stats['std']) + stats['mean']
        return denormalized

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.

        Returns:
            Dict with:
                - 'image': Tensor of shape [C, H, W]
                - 'targets': Dict of target values (if not test)
                - 'image_id': Image identifier
                - 'image_path': Path to image file
                - 'metadata': Dict of metadata values (if available)
        """
        image_id = self.image_ids[idx]
        # Handle image path - support both relative paths and img_dir + filename
        csv_image_path = Path(self.image_paths[image_id])

        # Try paths in order: full csv path, img_dir + csv path, img_dir + filename
        if csv_image_path.exists():
            image_path = csv_image_path
        elif (self.img_dir / csv_image_path).exists():
            image_path = self.img_dir / csv_image_path
        else:
            # Fall back to img_dir + just filename
            image_path = self.img_dir / csv_image_path.name

        # Load image
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)

        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Convert to tensor if no transform provided
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Prepare output
        sample = {
            'image': image,
            'image_id': image_id,
            'image_path': str(image_path),
        }

        # Add targets if training
        if not self.is_test:
            targets = {
                target_name: self.targets[target_name][idx]
                for target_name in TARGET_NAMES
            }
            # Normalize targets
            targets = self.normalize_targets(targets)
            sample['targets'] = targets

            # Add metadata if available
            if self.metadata:
                sample['metadata'] = {
                    col: self.metadata[col][idx]
                    for col in self.metadata.keys()
                }

        return sample

    def get_target_stats(self) -> Dict[str, Dict[str, float]]:
        """Get target normalization statistics."""
        return self.target_stats

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for DataLoader.

        Args:
            batch: List of samples from __getitem__

        Returns:
            Batched dict with stacked tensors
        """
        # Stack images
        images = torch.stack([sample['image'] for sample in batch])

        # Collect image IDs and paths
        image_ids = [sample['image_id'] for sample in batch]
        image_paths = [sample['image_path'] for sample in batch]

        batched = {
            'image': images,
            'image_id': image_ids,
            'image_path': image_paths,
        }

        # Stack targets if present
        if 'targets' in batch[0]:
            targets = {}
            for target_name in TARGET_NAMES:
                targets[target_name] = torch.tensor(
                    [sample['targets'][target_name] for sample in batch],
                    dtype=torch.float32
                )
            batched['targets'] = targets

        # Collect metadata if present
        if 'metadata' in batch[0]:
            metadata = {}
            metadata_keys = batch[0]['metadata'].keys()
            for key in metadata_keys:
                metadata[key] = [sample['metadata'][key] for sample in batch]
            batched['metadata'] = metadata

        return batched


def create_dataloaders(
    train_dataset: BiomassDataset,
    val_dataset: BiomassDataset,
    batch_size: int = 16,
    num_workers: int = 4,
    worker_init_fn=None
):
    """
    Create train and validation DataLoaders.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
        worker_init_fn: Worker initialization function for reproducibility

    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=BiomassDataset.collate_fn,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=BiomassDataset.collate_fn,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    return train_loader, val_loader
