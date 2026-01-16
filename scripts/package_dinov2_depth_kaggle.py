"""
Package DINOv2 + Depth fusion model for Kaggle upload.

Creates a package with:
- 5 fold best_model.pth files
- Depth Anything v2 model weights
- Config and results JSON

Usage:
    python scripts/package_dinov2_depth_kaggle.py
"""

import sys
sys.path.insert(0, '/home/chaot/kaggle/Image2Biomass-Competition')

import shutil
from pathlib import Path
from glob import glob


def main():
    print("=" * 70)
    print("Packaging DINOv2 + Depth Fusion Model for Kaggle")
    print("=" * 70)

    # Find latest depth checkpoint
    pattern = 'experiments/checkpoints_dinov2_base_depth_*'
    checkpoint_dirs = sorted(glob(pattern))

    if not checkpoint_dirs:
        raise FileNotFoundError(f"No DINOv2+Depth checkpoints found matching {pattern}")

    source_dir = Path(checkpoint_dirs[-1])
    print(f"\nSource: {source_dir}")

    # Create output directory
    output_dir = Path('kaggle_upload/dinov2_depth_model')
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

    # Save Depth Anything v2 model
    print("\nSaving Depth Anything v2 model...")
    depth_dir = output_dir / 'depth_anything_v2_small'
    depth_dir.mkdir(exist_ok=True)

    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    model_id = 'depth-anything/Depth-Anything-V2-Small-hf'
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)

    processor.save_pretrained(depth_dir)
    model.save_pretrained(depth_dir)

    depth_size = sum(f.stat().st_size for f in depth_dir.rglob('*') if f.is_file()) / (1024 * 1024)
    print(f"  Depth model: {depth_size:.1f} MB")

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
        print(f"  cd kaggle_upload && tar -czvf dinov2_depth_kaggle.tar.gz dinov2_depth_model/")
    else:
        print(f"\nWARNING: Only {total_copied}/5 folds found.")

    print("=" * 70)

    return output_dir


if __name__ == '__main__':
    main()
