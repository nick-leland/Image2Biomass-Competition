"""
Calculate validation R² for the EfficientNetV2-M model.
"""

import sys
sys.path.insert(0, '/home/chaot/kaggle/Image2Biomass-Competition')

import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score

from src.config import TRAIN_CSV, TRAIN_IMG_DIR, TARGET_NAMES
from src.utils.seed import set_seed, worker_init_fn
from src.data.dataset import BiomassDataset
from src.data.splitter import create_stratified_split
from src.data.transforms import get_val_transforms
from src.models.model_factory import create_model

def main():
    print("=" * 70)
    print("Calculate Validation R² - EfficientNetV2-M")
    print("=" * 70)

    # Load config
    config_path = Path('experiments/optuna_studies/advanced_models_20260111_013829/best_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"\nModel: {config['backbone']}")
    print(f"Image size: {config['image_size']}")

    # Set seed
    set_seed(config['seed'])

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load data and create split
    print(f"\nLoading data from {TRAIN_CSV}...")
    df = pd.read_csv(TRAIN_CSV)
    df['image_id'] = df['sample_id'].str.split('__').str[0]
    df_wide = df.pivot_table(
        index=['image_id', 'image_path', 'Sampling_Date', 'State', 'Species',
               'Pre_GSHH_NDVI', 'Height_Ave_cm'],
        columns='target_name',
        values='target'
    ).reset_index()

    # Create same split
    train_df, val_df = create_stratified_split(
        df_wide,
        stratify_by=config['stratify_by'],
        val_split=config['val_split'],
        random_seed=config['seed']
    )

    # Convert to long format
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

    # Save temporary CSVs
    temp_dir = Path('experiments/temp')
    temp_dir.mkdir(exist_ok=True, parents=True)
    val_csv_path = temp_dir / 'val_split.csv'
    val_long_df.to_csv(val_csv_path, index=False)
    train_csv_path = temp_dir / 'train_split.csv'
    train_long_df.to_csv(train_csv_path, index=False)

    # Create datasets
    val_transform = get_val_transforms(image_size=config['image_size'])

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
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=BiomassDataset.collate_fn,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    print(f"Val dataset: {len(val_dataset)} samples")

    # Create model
    print(f"\nCreating model: {config['backbone']}...")
    model = create_model(config)
    model = model.to(device)

    # Load checkpoint
    checkpoint_path = Path('experiments/checkpoints_final_efficientnetv2/best_model.pth')
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Best Val Loss (Huber): {checkpoint['best_val_loss']:.4f}")

    # Run validation
    print("\nRunning validation...")
    model.eval()

    all_predictions_denorm = {name: [] for name in TARGET_NAMES}
    all_targets_denorm = {name: [] for name in TARGET_NAMES}

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            targets = {k: v.to(device) for k, v in batch['targets'].items()}

            predictions = model(images)

            # Denormalize for each sample in batch
            batch_size = images.size(0)
            for i in range(batch_size):
                # Create dict for this sample
                pred_dict = {name: predictions[name][i].item() for name in TARGET_NAMES}
                tgt_dict = {name: targets[name][i].item() for name in TARGET_NAMES}

                # Denormalize
                pred_denorm = train_dataset.denormalize_targets(pred_dict)
                tgt_denorm = train_dataset.denormalize_targets(tgt_dict)

                # Store
                for name in TARGET_NAMES:
                    all_predictions_denorm[name].append(pred_denorm[name])
                    all_targets_denorm[name].append(tgt_denorm[name])

    # Calculate R² for each target
    print("\n" + "=" * 70)
    print("VALIDATION METRICS (R²)")
    print("=" * 70)

    # Collect all predictions and targets for global weighted R²
    all_preds = []
    all_targs = []
    all_weights = []

    # Per-target weights (you may need to adjust these based on competition spec)
    target_weights = {
        'Dry_Clover_g': 1.0,
        'Dry_Dead_g': 1.0,
        'Dry_Green_g': 1.0,
        'Dry_Total_g': 1.0,
        'GDM_g': 1.0
    }

    for name in TARGET_NAMES:
        preds = np.array(all_predictions_denorm[name])
        tgts = np.array(all_targets_denorm[name])

        # Calculate per-target R²
        r2 = r2_score(tgts, preds)
        mae = np.mean(np.abs(preds - tgts))

        print(f"{name:<20} R²: {r2:>7.4f}  MAE: {mae:>8.2f} (n={len(preds)})")

        # Collect for weighted R²
        all_preds.extend(preds)
        all_targs.extend(tgts)
        all_weights.extend([target_weights[name]] * len(preds))

    # Calculate global weighted R²
    all_preds = np.array(all_preds)
    all_targs = np.array(all_targs)
    all_weights = np.array(all_weights)

    # Weighted R² calculation
    mean_target = np.average(all_targs, weights=all_weights)
    ss_tot = np.sum(all_weights * (all_targs - mean_target) ** 2)
    ss_res = np.sum(all_weights * (all_targs - all_preds) ** 2)
    weighted_r2 = 1 - (ss_res / ss_tot)

    # Simple unweighted R²
    simple_r2 = r2_score(all_targs, all_preds)

    print(f"\n{'Global Weighted R²':<20} {weighted_r2:>7.4f}")
    print(f"{'Global Simple R²':<20} {simple_r2:>7.4f}")

    print("\n" + "=" * 70)
    print(f"KAGGLE PUBLIC SCORE:   0.42 R²")
    print(f"VALIDATION R² (wtd):   {weighted_r2:.4f}")
    print(f"VALIDATION R² (simple):{simple_r2:.4f}")
    print(f"GAP (weighted):        {abs(0.42 - weighted_r2):.4f}")
    print(f"GAP (simple):          {abs(0.42 - simple_r2):.4f}")
    print("=" * 70)

if __name__ == '__main__':
    main()
