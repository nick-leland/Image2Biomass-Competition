"""
Prepare external GrassClover biomass data for pretraining.

Maps the AILab GrassClover dataset columns to match our competition format:
    External Column  ->  Competition Target
    dry_grass        ->  Dry_Green_g
    dry_clover       ->  Dry_Clover_g
    dry_weeds        ->  Dry_Dead_g (partial - weeds can be dead/alive)
    dry_total        ->  Dry_Total_g
    (computed)       ->  GDM_g (= dry_grass + dry_clover)

Usage:
    python scripts/prepare_external_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import shutil


def prepare_grassclover_data():
    """Convert GrassClover format to competition format."""

    external_dir = Path('external_data')
    output_dir = Path('external_data/processed')
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 70)
    print("Preparing GrassClover External Data")
    print("=" * 70)

    # Load external training data
    train_csv = external_dir / 'train' / 'biomass_train_data.csv'
    df = pd.read_csv(train_csv, sep=';', encoding='utf-8-sig')

    print(f"\nLoaded {len(df)} samples from GrassClover dataset")
    print(f"Columns: {list(df.columns)}")

    # Map columns to competition format
    # External has: dry_grass, dry_clover (total), dry_weeds, dry_total
    # Competition needs: Dry_Green_g, Dry_Clover_g, Dry_Dead_g, Dry_Total_g, GDM_g

    processed_rows = []

    for idx, row in df.iterrows():
        image_file = row['image_file_name']
        image_path = f"external_data/train/images/{image_file}"

        # Map targets (values are in grams)
        dry_grass = row.get('dry_grass', 0) or 0
        dry_clover = row.get('dry_clover', 0) or 0
        dry_weeds = row.get('dry_weeds', 0) or 0
        dry_total = row.get('dry_total', 0) or 0

        # Handle NaN values
        dry_grass = 0 if pd.isna(dry_grass) else float(dry_grass)
        dry_clover = 0 if pd.isna(dry_clover) else float(dry_clover)
        dry_weeds = 0 if pd.isna(dry_weeds) else float(dry_weeds)
        dry_total = 0 if pd.isna(dry_total) else float(dry_total)

        # Map to competition targets
        # Dry_Green_g = dry_grass (grass is green matter)
        # Dry_Clover_g = dry_clover
        # Dry_Dead_g = dry_weeds (approximation - weeds can include dead matter)
        # Dry_Total_g = dry_total
        # GDM_g = dry_grass + dry_clover (Green Dry Matter)

        targets = {
            'Dry_Green_g': dry_grass,
            'Dry_Clover_g': dry_clover,
            'Dry_Dead_g': dry_weeds,
            'Dry_Total_g': dry_total,
            'GDM_g': dry_grass + dry_clover,
        }

        # Create image_id from filename
        image_id = image_file.replace('.jpg', '')

        processed_rows.append({
            'image_id': image_id,
            'image_path': image_path,
            'Sampling_Date': f"{row['acquisition_year']}/1/1",  # Approximate
            'State': 'Denmark',  # External data is from Denmark
            'Species': 'GrassClover',
            'Pre_GSHH_NDVI': 0.6,  # Placeholder
            'Height_Ave_cm': 5.0,  # Placeholder
            **targets
        })

    # Create wide format DataFrame
    df_wide = pd.DataFrame(processed_rows)

    print(f"\nProcessed {len(df_wide)} images")
    print(f"\nTarget statistics:")
    for col in ['Dry_Green_g', 'Dry_Clover_g', 'Dry_Dead_g', 'Dry_Total_g', 'GDM_g']:
        print(f"  {col:<15} mean: {df_wide[col].mean():>8.2f}  "
              f"std: {df_wide[col].std():>8.2f}  "
              f"min: {df_wide[col].min():>8.2f}  "
              f"max: {df_wide[col].max():>8.2f}")

    # Save wide format
    wide_csv_path = output_dir / 'grassclover_wide.csv'
    df_wide.to_csv(wide_csv_path, index=False)
    print(f"\nSaved wide format to: {wide_csv_path}")

    # Also create long format (like competition train.csv)
    long_rows = []
    target_names = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']

    for _, row in df_wide.iterrows():
        for target_name in target_names:
            long_rows.append({
                'sample_id': f"{row['image_id']}__{target_name}",
                'image_path': row['image_path'],
                'Sampling_Date': row['Sampling_Date'],
                'State': row['State'],
                'Species': row['Species'],
                'Pre_GSHH_NDVI': row['Pre_GSHH_NDVI'],
                'Height_Ave_cm': row['Height_Ave_cm'],
                'target_name': target_name,
                'target': row[target_name]
            })

    df_long = pd.DataFrame(long_rows)
    long_csv_path = output_dir / 'grassclover_long.csv'
    df_long.to_csv(long_csv_path, index=False)
    print(f"Saved long format to: {long_csv_path}")

    # Create combined dataset (competition + external)
    print("\n" + "=" * 70)
    print("Creating Combined Dataset")
    print("=" * 70)

    # Load competition data
    comp_df = pd.read_csv('train.csv')
    print(f"Competition data: {len(comp_df)} rows ({len(comp_df)//5} images)")
    print(f"External data: {len(df_long)} rows ({len(df_long)//5} images)")

    # Combine
    combined_df = pd.concat([comp_df, df_long], ignore_index=True)
    combined_csv_path = output_dir / 'combined_train.csv'
    combined_df.to_csv(combined_csv_path, index=False)

    print(f"\nCombined data: {len(combined_df)} rows ({len(combined_df)//5} images)")
    print(f"Saved to: {combined_csv_path}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Competition images: 357")
    print(f"  External images: {len(df_wide)}")
    print(f"  Combined total: {357 + len(df_wide)} images")
    print(f"\nReady for pretraining/combined training!")
    print("=" * 70)

    return df_wide, df_long, combined_df


if __name__ == '__main__':
    prepare_grassclover_data()
