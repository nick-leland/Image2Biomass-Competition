"""
Package foundation model checkpoints for Kaggle upload.

Creates a minimal package with:
- 5 fold best_model.pth files
- Config and results JSON

Usage:
    python scripts/package_foundation_model_kaggle.py --model siglip
    python scripts/package_foundation_model_kaggle.py --model dinov2
"""

import sys
sys.path.insert(0, '/home/chaot/kaggle/Image2Biomass-Competition')

import argparse
import shutil
from pathlib import Path
from glob import glob


def main():
    parser = argparse.ArgumentParser(description='Package foundation model for Kaggle')
    parser.add_argument('--model', type=str, required=True,
                        choices=['siglip', 'dinov2', 'eva02'],
                        help='Model type to package')
    args = parser.parse_args()

    print("=" * 70)
    print(f"Packaging {args.model.upper()} Foundation Model for Kaggle")
    print("=" * 70)

    # Find latest checkpoint for this model
    pattern = f'experiments/checkpoints_{args.model}_*'
    checkpoint_dirs = sorted(glob(pattern))

    if not checkpoint_dirs:
        raise FileNotFoundError(f"No {args.model} checkpoints found matching {pattern}")

    source_dir = Path(checkpoint_dirs[-1])
    print(f"\nSource: {source_dir}")

    # Create output directory
    output_dir = Path(f'kaggle_upload/{args.model}_foundation_model')
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    print(f"Output: {output_dir}")

    # Copy config and results
    print("\nCopying config files...")
    for json_file in ['config.json', 'kfold_results.json']:
        src = source_dir / json_file
        if src.exists():
            shutil.copy(src, output_dir / json_file)
            print(f"  Copied {json_file}")
        else:
            print(f"  Missing {json_file}")

    # Copy only best_model.pth from each fold
    print("\nCopying best models...")
    total_copied = 0
    for fold_idx in range(5):
        fold_src = source_dir / f'fold_{fold_idx}'
        fold_dst = output_dir / f'fold_{fold_idx}'

        if not fold_src.exists():
            print(f"  Fold {fold_idx}: MISSING")
            continue

        fold_dst.mkdir(exist_ok=True)

        best_model = fold_src / 'best_model.pth'
        if best_model.exists():
            shutil.copy(best_model, fold_dst / 'best_model.pth')
            size_mb = best_model.stat().st_size / (1024 * 1024)
            print(f"  Fold {fold_idx}: {size_mb:.1f} MB")
            total_copied += 1
        else:
            print(f"  Fold {fold_idx}: best_model.pth MISSING")

    # Calculate total size
    total_size = sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file()) / (1024 * 1024)

    print("\n" + "=" * 70)
    print("Packaging Complete!")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print(f"Folds copied: {total_copied}/5")
    print(f"Total size: {total_size:.1f} MB")

    if total_copied == 5:
        print("\nTo create tar.gz for upload:")
        print(f"  cd kaggle_upload && tar -czvf {args.model}_foundation_kaggle.tar.gz {args.model}_foundation_model/")
        print("\nTo upload to Kaggle:")
        print("  1. Create a new Dataset on Kaggle")
        print("  2. Upload the tar.gz file")
        print("  3. Use the dataset in your inference notebook")
    else:
        print(f"\nWARNING: Only {total_copied}/5 folds found. Training may still be in progress.")

    print("=" * 70)

    return output_dir


if __name__ == '__main__':
    main()
