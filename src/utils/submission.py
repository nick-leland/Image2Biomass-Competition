"""
Utilities for generating Kaggle submissions.
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm

from src.config import TARGET_NAMES


def predict_on_test_set(model, test_loader, device='cuda', denormalize_fn=None):
    """
    Run inference on test set.

    Args:
        model: Trained PyTorch model
        test_loader: Test DataLoader
        device: Device to run inference on
        denormalize_fn: Function to denormalize predictions

    Returns:
        predictions: Dict mapping image_id to dict of target predictions
    """
    model.eval()
    predictions = {}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating predictions"):
            images = batch['image'].to(device)
            image_ids = batch['image_id']

            # Get predictions
            pred = model(images)

            # Store predictions for each image
            for i, image_id in enumerate(image_ids):
                pred_dict = {
                    target_name: pred[target_name][i].cpu().item()
                    for target_name in TARGET_NAMES
                }
                predictions[image_id] = pred_dict

    # Denormalize if function provided
    if denormalize_fn is not None:
        predictions = {
            image_id: denormalize_fn(pred_dict)
            for image_id, pred_dict in predictions.items()
        }

    return predictions


def apply_constraint_enforcement(predictions: Dict[str, Dict[str, float]],
                                 method: str = 'biological') -> Dict[str, Dict[str, float]]:
    """
    Enforce biological constraints on biomass predictions.

    Biological relationships:
        - GDM_g = Dry_Green_g + Dry_Clover_g (Green Dry Matter)
        - Dry_Total_g = GDM_g + Dry_Dead_g = Dry_Green_g + Dry_Clover_g + Dry_Dead_g

    Args:
        predictions: Dict mapping image_id to dict of target predictions
        method: Enforcement method:
            - 'biological': Enforce all biological constraints (recommended)
            - 'average': Average predicted total with sum of components
            - 'hard_override': Recalculate derived values from components
            - 'trust_model': No adjustment

    Returns:
        predictions: Updated predictions with constraints enforced
    """
    enforced = {}

    for image_id, pred_dict in predictions.items():
        pred = pred_dict.copy()

        # Step 1: Clip all predictions to non-negative (biomass can't be negative!)
        for key in pred:
            pred[key] = max(0.0, pred[key])

        clover = pred['Dry_Clover_g']
        dead = pred['Dry_Dead_g']
        green = pred['Dry_Green_g']
        gdm = pred['GDM_g']
        total = pred['Dry_Total_g']

        if method == 'biological':
            # Enforce: GDM = Green + Clover
            # Enforce: Total = GDM + Dead = Green + Clover + Dead

            # Calculate what GDM should be based on components
            gdm_from_components = green + clover

            # Average predicted GDM with calculated GDM
            adjusted_gdm = (gdm + gdm_from_components) / 2

            # If components exist, scale them to match adjusted GDM
            if gdm_from_components > 0:
                gdm_scale = adjusted_gdm / gdm_from_components
                pred['Dry_Green_g'] = green * gdm_scale
                pred['Dry_Clover_g'] = clover * gdm_scale
            pred['GDM_g'] = adjusted_gdm

            # Now enforce Total = GDM + Dead
            total_from_components = adjusted_gdm + dead

            # Average predicted total with calculated total
            adjusted_total = (total + total_from_components) / 2

            # Scale dead matter to match (GDM is already fixed)
            if adjusted_total > adjusted_gdm:
                pred['Dry_Dead_g'] = adjusted_total - adjusted_gdm
            else:
                # Total can't be less than GDM, so set dead to 0
                pred['Dry_Dead_g'] = 0.0
                adjusted_total = adjusted_gdm

            pred['Dry_Total_g'] = adjusted_total

        elif method == 'average':
            # Legacy: Average total with sum of components
            component_sum = clover + dead + green
            new_total = (total + component_sum) / 2

            if component_sum > 0:
                scale = new_total / component_sum
                pred['Dry_Clover_g'] = clover * scale
                pred['Dry_Dead_g'] = dead * scale
                pred['Dry_Green_g'] = green * scale
                pred['Dry_Total_g'] = new_total
                pred['GDM_g'] = pred['Dry_Green_g'] + pred['Dry_Clover_g']
            else:
                pred['Dry_Total_g'] = 0.0
                pred['GDM_g'] = 0.0

        elif method == 'hard_override':
            # Recalculate derived values from base components
            pred['GDM_g'] = green + clover
            pred['Dry_Total_g'] = green + clover + dead

        elif method == 'trust_model':
            # No adjustment beyond clipping
            pass

        enforced[image_id] = pred

    return enforced


def apply_test_time_augmentation_with_flips(model, test_dataset, device='cuda',
                                           denormalize_fn=None, batch_size=16,
                                           include_rotations=True):
    """
    Apply test-time augmentation using flips and rotations.

    Averages predictions over multiple augmented versions:
    - With rotations (default): 8 transforms (4 flip combos x 2 rotations)
    - Without rotations: 4 transforms (4 flip combos only)

    Args:
        model: Trained PyTorch model
        test_dataset: Test Dataset (will create new loaders with TTA transforms)
        device: Device to run inference on
        denormalize_fn: Function to denormalize predictions
        batch_size: Batch size for inference
        include_rotations: If True, include 90-degree rotations (8 transforms total)

    Returns:
        predictions: Dict mapping image_id to dict of averaged target predictions
    """
    from src.data.transforms import get_tta_transforms
    from torch.utils.data import DataLoader

    model.eval()
    all_predictions = {target: {} for target in TARGET_NAMES}

    # Get TTA transforms
    tta_transforms = get_tta_transforms(
        image_size=test_dataset.image_size,
        include_rotations=include_rotations
    )

    tta_desc = "flips + rotations" if include_rotations else "flips only"
    print(f"Using {len(tta_transforms)} TTA transforms ({tta_desc})")

    with torch.no_grad():
        for tta_idx, tta_transform in enumerate(tta_transforms):
            # Create temporary dataset with this TTA transform
            test_dataset_tta = type(test_dataset)(
                csv_path=test_dataset.csv_path,
                img_dir=test_dataset.img_dir,
                transform=tta_transform,
                is_test=test_dataset.is_test,
                target_stats=test_dataset.target_stats
            )

            # Create loader
            tta_loader = DataLoader(
                test_dataset_tta,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=type(test_dataset).collate_fn
            )

            for batch in tqdm(tta_loader, desc=f"TTA {tta_idx + 1}/{len(tta_transforms)}"):
                images = batch['image'].to(device)
                image_ids = batch['image_id']

                # Get predictions
                pred = model(images)

                # Store predictions
                for i, image_id in enumerate(image_ids):
                    for target_name in TARGET_NAMES:
                        if image_id not in all_predictions[target_name]:
                            all_predictions[target_name][image_id] = []
                        all_predictions[target_name][image_id].append(
                            pred[target_name][i].cpu().item()
                        )

    # Average predictions
    predictions = {}
    for image_id in all_predictions[TARGET_NAMES[0]].keys():
        pred_dict = {
            target_name: np.mean(all_predictions[target_name][image_id])
            for target_name in TARGET_NAMES
        }
        predictions[image_id] = pred_dict

    # Denormalize if function provided
    if denormalize_fn is not None:
        predictions = {
            image_id: denormalize_fn(pred_dict)
            for image_id, pred_dict in predictions.items()
        }

    return predictions


def apply_test_time_augmentation(model, test_loader, device='cuda',
                                 denormalize_fn=None, n_tta: int = 5):
    """
    Apply test-time augmentation by averaging predictions over multiple augmented versions.

    DEPRECATED: Use apply_test_time_augmentation_with_flips instead for better TTA.

    Args:
        model: Trained PyTorch model
        test_loader: Test DataLoader (should have TTA transforms)
        device: Device to run inference on
        denormalize_fn: Function to denormalize predictions
        n_tta: Number of TTA iterations

    Returns:
        predictions: Dict mapping image_id to dict of averaged target predictions
    """
    model.eval()
    all_predictions = {target: {} for target in TARGET_NAMES}

    with torch.no_grad():
        for tta_idx in range(n_tta):
            for batch in tqdm(test_loader, desc=f"TTA iteration {tta_idx + 1}/{n_tta}"):
                images = batch['image'].to(device)
                image_ids = batch['image_id']

                # Get predictions
                pred = model(images)

                # Store predictions
                for i, image_id in enumerate(image_ids):
                    for target_name in TARGET_NAMES:
                        if image_id not in all_predictions[target_name]:
                            all_predictions[target_name][image_id] = []
                        all_predictions[target_name][image_id].append(
                            pred[target_name][i].cpu().item()
                        )

    # Average predictions
    predictions = {}
    for image_id in all_predictions[TARGET_NAMES[0]].keys():
        pred_dict = {
            target_name: np.mean(all_predictions[target_name][image_id])
            for target_name in TARGET_NAMES
        }
        predictions[image_id] = pred_dict

    # Denormalize if function provided
    if denormalize_fn is not None:
        predictions = {
            image_id: denormalize_fn(pred_dict)
            for image_id, pred_dict in predictions.items()
        }

    return predictions


def create_submission_csv(predictions: Dict[str, Dict[str, float]],
                         output_path: Path,
                         test_csv_path: Path):
    """
    Create submission CSV in Kaggle format.

    Args:
        predictions: Dict mapping image_id to dict of target predictions
        output_path: Path to save submission CSV
        test_csv_path: Path to test.csv (for sample_id ordering)
    """
    # Load test.csv to get correct sample_id ordering
    test_df = pd.read_csv(test_csv_path)

    # Create submission rows
    submission_rows = []
    for _, row in test_df.iterrows():
        sample_id = row['sample_id']
        image_id = sample_id.split('__')[0]
        target_name = row['target_name']

        # Get prediction
        pred_value = predictions[image_id][target_name]

        submission_rows.append({
            'sample_id': sample_id,
            'target': pred_value
        })

    # Create DataFrame and save
    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv(output_path, index=False)

    print(f"\nSubmission saved to: {output_path}")
    print(f"Submission shape: {submission_df.shape}")
    print(f"\nFirst few predictions:")
    print(submission_df.head(10))
    print(f"\nSummary statistics:")
    print(submission_df['target'].describe())


def generate_submission(model, test_loader=None, test_dataset=None, device='cuda',
                       denormalize_fn=None,
                       constraint_method: Optional[str] = 'biological',
                       use_tta: bool = False,
                       include_rotations: bool = True,
                       batch_size: int = 16,
                       output_path: Path = None,
                       test_csv_path: Path = None):
    """
    Complete submission generation pipeline.

    Args:
        model: Trained PyTorch model
        test_loader: Test DataLoader (required if not using TTA)
        test_dataset: Test Dataset (required if using TTA)
        device: Device to run inference on
        denormalize_fn: Function to denormalize predictions
        constraint_method: Method for constraint enforcement:
            - 'biological': Enforce GDM=Green+Clover, Total=GDM+Dead (recommended)
            - 'average': Average predicted total with sum of components
            - 'hard_override': Recalculate derived values
            - 'trust_model': No adjustment (only clips negatives)
            - None: No post-processing at all
        use_tta: Whether to use test-time augmentation
        include_rotations: If True, TTA includes 90-degree rotations (8 transforms)
        batch_size: Batch size for TTA inference
        output_path: Path to save submission CSV
        test_csv_path: Path to test.csv

    Returns:
        predictions: Dict of predictions
    """
    print("=" * 70)
    print("Generating Kaggle Submission")
    print("=" * 70)

    # Run inference
    if use_tta:
        if test_dataset is None:
            raise ValueError("test_dataset is required when use_tta=True")
        tta_desc = "flips + rotations" if include_rotations else "flips only"
        print(f"\nUsing Test-Time Augmentation (TTA) with {tta_desc}...")
        predictions = apply_test_time_augmentation_with_flips(
            model, test_dataset, device, denormalize_fn, batch_size, include_rotations
        )
    else:
        if test_loader is None:
            raise ValueError("test_loader is required when use_tta=False")
        print("\nRunning inference on test set...")
        predictions = predict_on_test_set(model, test_loader, device, denormalize_fn)

    print(f"Generated predictions for {len(predictions)} images")

    # Apply constraint enforcement
    if constraint_method is not None:
        print(f"\nApplying constraint enforcement (method: {constraint_method})...")
        predictions = apply_constraint_enforcement(predictions, method=constraint_method)

    # Create submission CSV
    if output_path is not None and test_csv_path is not None:
        create_submission_csv(predictions, output_path, test_csv_path)

    return predictions
