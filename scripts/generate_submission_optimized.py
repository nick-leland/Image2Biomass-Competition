"""
Generate Kaggle submission from optimized model.

Usage:
    python scripts/generate_submission_optimized.py
"""

import sys
sys.path.insert(0, '/home/chaot/kaggle/Image2Biomass-Competition')

import json
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime

from src.config import TEST_CSV, TEST_IMG_DIR, TRAIN_CSV, TRAIN_IMG_DIR, TARGET_NAMES, SUBMISSIONS_DIR
from src.utils.seed import set_seed, worker_init_fn
from src.data.dataset import BiomassDataset
from src.data.transforms import get_val_transforms
from src.models.model_factory import create_model
from src.utils.submission import generate_submission


def main():
    print("=" * 70)
    print("Image2Biomass - Generate Submission (Optimized Model)")
    print("=" * 70)

    # Load optimized config
    config_path = Path('experiments/optuna_studies/optuna_v1/best_config.json')
    print(f"\nLoading optimized config from: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"\nModel: {config['backbone']}")
    print(f"Image size: {config['image_size']}")
    print(f"Loss: {config['loss_function']}")

    # Set seed
    set_seed(config['seed'])

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create model with optimized config
    print(f"\nCreating model: {config['backbone']}...")
    model = create_model(config)
    model = model.to(device)

    # Load checkpoint
    checkpoint_path = Path('experiments/checkpoints_optimized/best_model.pth')
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Best Val Loss: {checkpoint['best_val_loss']:.4f}")

    # Get target statistics from training data
    print(f"\nLoading training data to get normalization statistics...")
    df = pd.read_csv(TRAIN_CSV)
    df['image_id'] = df['sample_id'].str.split('__').str[0]

    # Convert to long format for temporary CSV
    temp_dir = Path('experiments/temp')
    temp_dir.mkdir(exist_ok=True, parents=True)
    temp_train_csv = temp_dir / 'train_for_stats.csv'
    df.to_csv(temp_train_csv, index=False)

    # Create temporary training dataset just to get normalization stats
    val_transform = get_val_transforms(image_size=config['image_size'])
    train_dataset = BiomassDataset(
        csv_path=temp_train_csv,
        img_dir=TRAIN_IMG_DIR,
        transform=val_transform,
        is_test=False
    )
    target_stats = train_dataset.get_target_stats()

    print("\nTarget normalization statistics:")
    for target_name in TARGET_NAMES:
        stats = target_stats[target_name]
        print(f"  {target_name:<20} mean: {stats['mean']:>8.2f}  std: {stats['std']:>8.2f}")

    # Create test dataset
    print(f"\nLoading test data from {TEST_CSV}...")
    test_dataset = BiomassDataset(
        csv_path=TEST_CSV,
        img_dir=TEST_IMG_DIR,
        transform=val_transform,
        is_test=True,
        target_stats=target_stats
    )

    print(f"Test dataset: {len(test_dataset)} samples")

    # Create test loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=BiomassDataset.collate_fn,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    # Generate output filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f'submission_optimized_{timestamp}.csv'
    output_path = SUBMISSIONS_DIR / output_filename

    # Ensure submissions directory exists
    SUBMISSIONS_DIR.mkdir(exist_ok=True, parents=True)

    # Generate submission (set use_tta=True for Test-Time Augmentation)
    predictions = generate_submission(
        model=model,
        test_loader=test_loader,
        test_dataset=test_dataset,  # For TTA support
        device=device,
        denormalize_fn=train_dataset.denormalize_targets,
        constraint_method='average',
        use_tta=True,  # Enable TTA for better performance
        batch_size=config['batch_size'],
        output_path=output_path,
        test_csv_path=TEST_CSV
    )

    print("\n" + "=" * 70)
    print("Submission generation complete!")
    print(f"Saved to: {output_path}")
    print("=" * 70)

    # Print predictions summary
    print("\nPredictions by target:")
    for target_name in TARGET_NAMES:
        values = [pred[target_name] for pred in predictions.values()]
        print(f"  {target_name:<20} mean: {sum(values)/len(values):>8.2f}  "
              f"min: {min(values):>8.2f}  max: {max(values):>8.2f}")

    # Check constraint violations
    print("\nConstraint check (Dry_Total vs sum of components):")
    violations = []
    for image_id, pred in predictions.items():
        total = pred['Dry_Total_g']
        component_sum = pred['Dry_Clover_g'] + pred['Dry_Dead_g'] + pred['Dry_Green_g']
        violation = abs(total - component_sum)
        violations.append(violation)

    print(f"  Mean violation: {sum(violations)/len(violations):.6f}g")
    print(f"  Max violation: {max(violations):.6f}g")
    print(f"  All exact: {all(v < 1e-6 for v in violations)}")

    print(f"\n{'=' * 70}")
    print("Ready to submit to Kaggle!")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
