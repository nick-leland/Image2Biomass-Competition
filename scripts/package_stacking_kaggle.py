#!/usr/bin/env python3
"""
Package stacking ensemble models for Kaggle submission.

Creates a tarball with:
- V4 (EfficientNetV2) checkpoints (5 folds)
- V7 (DINOv2) checkpoints (5 folds)
- V8 (DINOv2 + Depth) checkpoints (5 folds)
- Meta-learners (Ridge regression)
- Depth Anything v2 model

Usage:
    python scripts/package_stacking_kaggle.py
"""

import shutil
import tarfile
from pathlib import Path
import torch


def main():
    output_dir = Path('kaggle_upload/stacking_ensemble')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model checkpoint directories
    v4_dir = Path('experiments/checkpoints_kfold_mse')
    v7_dir = Path('experiments/checkpoints_dinov2_base_20260116_100625')
    v8_dir = Path('experiments/checkpoints_dinov2_base_depth_20260116_140724')
    stacking_dir = Path('experiments/stacking_ensemble')

    print("Packaging stacking ensemble for Kaggle...")
    print("=" * 60)

    # Copy V4 checkpoints
    print("\n1. Copying V4 (EfficientNetV2) checkpoints...")
    v4_out = output_dir / 'v4_checkpoints'
    v4_out.mkdir(exist_ok=True)
    for fold_idx in range(5):
        src = v4_dir / f'fold_{fold_idx}' / 'best_model.pth'
        if src.exists():
            dst = v4_out / f'fold_{fold_idx}_best_model.pth'
            shutil.copy(src, dst)
            size_mb = dst.stat().st_size / (1024 * 1024)
            print(f"   Fold {fold_idx}: {size_mb:.1f} MB")

    # Copy V7 checkpoints
    print("\n2. Copying V7 (DINOv2) checkpoints...")
    v7_out = output_dir / 'v7_checkpoints'
    v7_out.mkdir(exist_ok=True)
    for fold_idx in range(5):
        src = v7_dir / f'fold_{fold_idx}' / 'best_model.pth'
        if src.exists():
            dst = v7_out / f'fold_{fold_idx}_best_model.pth'
            shutil.copy(src, dst)
            size_mb = dst.stat().st_size / (1024 * 1024)
            print(f"   Fold {fold_idx}: {size_mb:.1f} MB")

    # Copy V8 checkpoints
    print("\n3. Copying V8 (DINOv2 + Depth) checkpoints...")
    v8_out = output_dir / 'v8_checkpoints'
    v8_out.mkdir(exist_ok=True)
    for fold_idx in range(5):
        src = v8_dir / f'fold_{fold_idx}' / 'best_model.pth'
        if src.exists():
            dst = v8_out / f'fold_{fold_idx}_best_model.pth'
            shutil.copy(src, dst)
            size_mb = dst.stat().st_size / (1024 * 1024)
            print(f"   Fold {fold_idx}: {size_mb:.1f} MB")

    # Copy stacking meta-learners and config
    print("\n4. Copying stacking meta-learners...")
    shutil.copy(stacking_dir / 'meta_learners.pkl', output_dir / 'meta_learners.pkl')
    shutil.copy(stacking_dir / 'config.json', output_dir / 'config.json')
    print("   meta_learners.pkl copied")
    print("   config.json copied")

    # Download and save Depth Anything v2
    print("\n5. Downloading Depth Anything v2 model...")
    depth_out = output_dir / 'depth_anything_v2'
    depth_out.mkdir(exist_ok=True)

    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")

        processor.save_pretrained(depth_out)
        model.save_pretrained(depth_out)
        print("   Depth Anything v2 Small saved")
    except Exception as e:
        print(f"   Warning: Could not save Depth Anything: {e}")
        print("   Will need to download during inference")

    # Create tarball
    print("\n6. Creating tarball...")
    tarball_path = Path('kaggle_upload/stacking_ensemble_kaggle.tar.gz')

    with tarfile.open(tarball_path, 'w:gz') as tar:
        tar.add(output_dir, arcname='stacking_ensemble')

    tarball_size = tarball_path.stat().st_size / (1024 * 1024 * 1024)
    print(f"   Created: {tarball_path}")
    print(f"   Size: {tarball_size:.2f} GB")

    # Summary
    print("\n" + "=" * 60)
    print("PACKAGING COMPLETE")
    print("=" * 60)
    print(f"\nOutput: {tarball_path}")
    print(f"Size: {tarball_size:.2f} GB")
    print("\nContents:")
    print("  - v4_checkpoints/ (5 folds, EfficientNetV2)")
    print("  - v7_checkpoints/ (5 folds, DINOv2)")
    print("  - v8_checkpoints/ (5 folds, DINOv2 + Depth)")
    print("  - meta_learners.pkl (Ridge regression)")
    print("  - config.json")
    print("  - depth_anything_v2/ (Depth Anything v2 Small)")
    print("\nUpload to Kaggle Datasets, then use in inference notebook.")


if __name__ == '__main__':
    main()
