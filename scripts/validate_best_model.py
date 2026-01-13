"""
Quick script to validate the best trained model.
"""

import sys
sys.path.insert(0, '/home/chaot/kaggle/Image2Biomass-Competition')

import torch
import pandas as pd
from pathlib import Path

from src.config import (
    TRAIN_CSV, TRAIN_IMG_DIR, DEFAULT_CONFIG,
    CHECKPOINTS_DIR, TARGET_NAMES
)
from src.utils.seed import set_seed, worker_init_fn
from src.data.dataset import BiomassDataset, create_dataloaders
from src.data.splitter import create_stratified_split
from src.data.transforms import get_val_transforms
from src.models.model_factory import create_model
from src.evaluation.metrics import print_metrics, compute_metrics

def main():
    print("=" * 70)
    print("Validating Best Model")
    print("=" * 70)

    # Set seed
    set_seed(DEFAULT_CONFIG['seed'])

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Load data and create split (same as training)
    print(f"\nLoading data from {TRAIN_CSV}...")
    df = pd.read_csv(TRAIN_CSV)
    df['image_id'] = df['sample_id'].str.split('__').str[0]
    df_wide = df.pivot_table(
        index=['image_id', 'image_path', 'Sampling_Date', 'State', 'Species',
               'Pre_GSHH_NDVI', 'Height_Ave_cm'],
        columns='target_name',
        values='target'
    ).reset_index()

    # Create same split as training
    train_df, val_df = create_stratified_split(
        df_wide,
        stratify_by=DEFAULT_CONFIG['stratify_by'],
        val_split=DEFAULT_CONFIG['val_split'],
        random_seed=DEFAULT_CONFIG['seed']
    )

    # Recreate validation dataset
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
                'Species': row.get('Species', ''),
                'Pre_GSHH_NDVI': row.get('Pre_GSHH_NDVI', 0),
                'Height_Ave_cm': row.get('Height_Ave_cm', 0),
                'Sampling_Date': row.get('Sampling_Date', ''),
            })
    val_long_df = pd.DataFrame(val_long)

    # Save temporary CSV
    temp_dir = Path('experiments/temp')
    temp_dir.mkdir(exist_ok=True, parents=True)
    val_csv_path = temp_dir / 'val_split.csv'
    val_long_df.to_csv(val_csv_path, index=False)

    # Create dataset (need train dataset first for stats)
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
                'Species': row.get('Species', ''),
                'Pre_GSHH_NDVI': row.get('Pre_GSHH_NDVI', 0),
                'Height_Ave_cm': row.get('Height_Ave_cm', 0),
                'Sampling_Date': row.get('Sampling_Date', ''),
            })
    train_long_df = pd.DataFrame(train_long)
    train_csv_path = temp_dir / 'train_split.csv'
    train_long_df.to_csv(train_csv_path, index=False)

    val_transform = get_val_transforms(image_size=DEFAULT_CONFIG['image_size'])

    train_dataset = BiomassDataset(
        csv_path=train_csv_path,
        img_dir=TRAIN_IMG_DIR,
        transform=val_transform,
        is_test=False
    )

    val_dataset = BiomassDataset(
        csv_path=val_csv_path,
        img_dir=TRAIN_IMG_DIR,
        transform=val_transform,
        is_test=False,
        target_stats=train_dataset.get_target_stats()
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=DEFAULT_CONFIG['batch_size'],
        shuffle=False,
        num_workers=DEFAULT_CONFIG['num_workers'],
        collate_fn=BiomassDataset.collate_fn,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    print(f"Val dataset: {len(val_dataset)} samples")

    # Create model
    print(f"\nCreating model: {DEFAULT_CONFIG['backbone']}...")
    model = create_model(DEFAULT_CONFIG)
    model = model.to(device)

    # Load best checkpoint
    checkpoint_path = CHECKPOINTS_DIR / 'best_model.pth'
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Best Val Loss: {checkpoint['best_val_loss']:.4f}")

    # Validate
    print("\nRunning validation...")
    model.eval()
    all_predictions = {name: [] for name in TARGET_NAMES}
    all_targets = {name: [] for name in TARGET_NAMES}

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            targets = {k: v.to(device) for k, v in batch['targets'].items()}

            predictions = model(images)

            for name in TARGET_NAMES:
                all_predictions[name].append(predictions[name])
                all_targets[name].append(targets[name])

    # Concatenate all batches
    all_predictions = {name: torch.cat(preds) for name, preds in all_predictions.items()}
    all_targets = {name: torch.cat(tgts) for name, tgts in all_targets.items()}

    # Compute and print metrics
    val_metrics = compute_metrics(all_predictions, all_targets, train_dataset.denormalize_targets)
    print_metrics(val_metrics, prefix="Final Validation")

    print("\n" + "=" * 70)
    print("Validation complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
