"""
Generate Kaggle submission from trained model.

Usage:
    python scripts/generate_submission.py
    python scripts/generate_submission.py --checkpoint path/to/checkpoint.pth
    python scripts/generate_submission.py --tta --n_tta 10
"""

import sys
sys.path.insert(0, '/home/chaot/kaggle/Image2Biomass-Competition')

import argparse
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime

from src.config import (
    TEST_CSV, TEST_IMG_DIR, TRAIN_CSV, TRAIN_IMG_DIR,
    DEFAULT_CONFIG, CHECKPOINTS_DIR, SUBMISSIONS_DIR,
    TARGET_NAMES
)
from src.utils.seed import set_seed, worker_init_fn
from src.data.dataset import BiomassDataset
from src.data.transforms import get_val_transforms
from src.models.model_factory import create_model
from src.utils.submission import generate_submission


def main():
    parser = argparse.ArgumentParser(description='Generate Kaggle submission')
    parser.add_argument('--checkpoint', type=str, default='best_model.pth',
                       help='Checkpoint filename (default: best_model.pth)')
    parser.add_argument('--constraint_method', type=str, default='average',
                       choices=['average', 'trust_model', 'hard_override', 'none'],
                       help='Constraint enforcement method (default: average)')
    parser.add_argument('--tta', action='store_true',
                       help='Use test-time augmentation')
    parser.add_argument('--n_tta', type=int, default=5,
                       help='Number of TTA iterations (default: 5)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (default: from config)')
    parser.add_argument('--output_name', type=str, default=None,
                       help='Output filename (default: auto-generated with timestamp)')
    args = parser.parse_args()

    print("=" * 70)
    print("Image2Biomass - Generate Submission")
    print("=" * 70)

    # Set seed
    set_seed(DEFAULT_CONFIG['seed'])

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create model
    print(f"\nCreating model: {DEFAULT_CONFIG['backbone']}...")
    model = create_model(DEFAULT_CONFIG)
    model = model.to(device)

    # Load checkpoint
    checkpoint_path = CHECKPOINTS_DIR / args.checkpoint
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
    val_transform = get_val_transforms(image_size=DEFAULT_CONFIG['image_size'])
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
    batch_size = args.batch_size if args.batch_size is not None else DEFAULT_CONFIG['batch_size']
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=DEFAULT_CONFIG['num_workers'],
        collate_fn=BiomassDataset.collate_fn,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    # Generate output filename
    if args.output_name is not None:
        output_filename = args.output_name
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        tta_suffix = f'_tta{args.n_tta}' if args.tta else ''
        constraint_suffix = f'_{args.constraint_method}' if args.constraint_method != 'none' else ''
        output_filename = f'submission_{timestamp}{tta_suffix}{constraint_suffix}.csv'

    output_path = SUBMISSIONS_DIR / output_filename

    # Ensure submissions directory exists
    SUBMISSIONS_DIR.mkdir(exist_ok=True, parents=True)

    # Generate submission
    constraint_method = None if args.constraint_method == 'none' else args.constraint_method

    predictions = generate_submission(
        model=model,
        test_loader=test_loader,
        device=device,
        denormalize_fn=train_dataset.denormalize_targets,
        constraint_method=constraint_method,
        use_tta=args.tta,
        n_tta=args.n_tta,
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
    if constraint_method is not None:
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


if __name__ == '__main__':
    main()
