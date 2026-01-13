# Latest Improvements Summary

## Overview
Major updates to handle advanced architectures (ViT, ConvNeXt, Swin), more aggressive augmentations, Test-Time Augmentation (TTA), and live Optuna monitoring.

## 1. Augmentation Improvements

### New "Extreme" Augmentation Level
Added a 4th augmentation level for training Vision Transformers and other models prone to overfitting on small datasets.

**File**: `src/data/transforms.py`

**Extreme augmentation includes:**
- Stronger geometric transforms (30-degree rotation, 20% shift, 30% scale)
- More aggressive color jitter (0.4 brightness/contrast/saturation, 0.15 hue)
- Higher probability of blur/noise (50%)
- Random gamma correction
- More coarse dropout (12 holes, 48x48 max size)
- Random shadows and perspective transforms

**Comparison:**
- **Conservative**: Minimal (flips only)
- **Moderate**: Standard augmentation
- **Aggressive**: Current best (used for ResNet50)
- **Extreme**: Maximum regularization for large models

## 2. Advanced Model Architectures

### Updated Model Support
**File**: `src/models/resnet_baseline.py`

Added `_load_timm_model()` method to support any timm architecture:
- Vision Transformers (ViT tiny/small/base)
- ConvNeXt (tiny/small/base)
- EfficientNetV2 (s/m)
- Swin Transformer (tiny/small)

### Optuna Search Space Expansion
**File**: `src/optuna_optimization/hyperparameters.py`

**Added 17 model options:**
- `vit_tiny_patch16_224` (5.7M params - good for small datasets)
- `vit_tiny_patch16_384` (5.7M params)
- `vit_small_patch16_224` (22M params)
- `vit_small_patch16_384` (22M params)
- `vit_base_patch16_224` (86M params)
- `vit_base_patch16_384` (86M params)
- `convnext_tiny` (28M params)
- `convnext_small` (50M params)
- `convnext_base` (89M params)
- `tf_efficientnetv2_s` (21M params)
- `tf_efficientnetv2_m` (54M params)
- `swin_tiny_patch4_window7_224` (28M params)
- `swin_small_patch4_window7_224` (50M params)
- `resnet50` (baseline - 25M params)
- `efficientnet_b4` (19M params)

## 3. Test-Time Augmentation (TTA)

### Improved TTA Implementation
**File**: `src/utils/submission.py`

New function: `apply_test_time_augmentation_with_flips()`
- Uses proper flip-based TTA (4 transforms total)
- Original image
- Horizontal flip
- Vertical flip
- Both flips

**Enabled by default** in `scripts/generate_submission_optimized.py`:
```python
use_tta=True  # Now enabled
```

### Dataset Support for TTA
**File**: `src/data/dataset.py`

Added `image_size` attribute extraction from transform:
```python
self.image_size = self._extract_image_size(transform)
```

This allows TTA to create new datasets with flip transforms at the correct image size.

## 4. Live Optuna Dashboard

### Dashboard Installation
```bash
uv pip install optuna-dashboard
```

### Helper Script
**File**: `scripts/run_optuna_with_dashboard.sh`

Launches both Optuna optimization and dashboard in one command:
```bash
bash scripts/run_optuna_with_dashboard.sh [n_trials] [port]
```

**Default**: 100 trials, port 8080

**Features:**
- Starts dashboard in background
- Runs optimization
- Shows live URL: `http://localhost:8080`
- Auto-cleanup on completion

### Manual Dashboard Launch
```bash
source .venv/bin/activate
optuna-dashboard sqlite:///experiments/optuna_studies/STUDY_NAME.db --port 8080
```

### Updated Default Trials
**File**: `scripts/optimize_hyperparams.py`

Changed default from 50 to 100 trials for more thorough search.

## 5. Results So Far

### Baseline Performance
- **ResNet34**: 1.4178 val loss
- **ResNet50 (optimized)**: 0.4387 val loss (69% improvement)
- **ViT Base 384**: 0.6640 val loss (51% worse than ResNet50)

### Kaggle Submission
- Best submission: 0.30 (very poor - rank 2776/3267)
- Despite perfect local validation performance
- Indicates severe overfitting or distribution shift

## 6. Next Steps

### Immediate: Run Full Optuna Study
```bash
bash scripts/run_optuna_with_dashboard.sh 100 8080
```

**Expected duration**: 12-24 hours

**What it will optimize:**
- 17 different architectures (including smaller ViTs)
- 4 augmentation levels (including extreme)
- Learning rates, optimizers, schedulers
- Loss functions, task weights
- Image sizes, batch sizes

### After Optuna Completes

1. **Train best model**:
   ```bash
   python scripts/train_optimized.py
   ```

2. **Generate submission with TTA** (now enabled by default):
   ```bash
   python scripts/generate_submission_optimized.py
   ```

3. **Upload to Kaggle** and check if TTA + better architecture improves score

## 7. Why These Changes Help

### Problem Analysis
- **Small dataset (357 images)**: ViT base too large, needs tiny/small variants
- **Overfitting**: Aggressive augmentation + TTA should help
- **Poor test performance**: TTA provides ensemble effect, reducing variance

### Expected Improvements
1. **Smaller ViTs**: Better sample efficiency than ViT base
2. **ConvNeXt**: Modern CNN with ViT-like design, sample-efficient
3. **Extreme augmentation**: More regularization for large models
4. **TTA**: 4x predictions averaged, more robust
5. **Systematic search**: 100 trials >> manual tuning

## 8. Monitoring Progress

### Dashboard Features
- Real-time trial progress
- Optimization history plot
- Parameter importance analysis
- Parallel coordinate plot
- Trial details and hyperparameters

### Live URL
Once started: `http://localhost:8080`

## Files Modified

1. `src/data/transforms.py` - Added extreme augmentation
2. `src/models/resnet_baseline.py` - Added timm model support
3. `src/optuna_optimization/hyperparameters.py` - Expanded search space
4. `src/data/dataset.py` - Added image_size attribute
5. `src/utils/submission.py` - Improved TTA implementation
6. `scripts/optimize_hyperparams.py` - Increased trials, added dashboard info
7. `scripts/generate_submission_optimized.py` - Enabled TTA by default
8. `scripts/run_optuna_with_dashboard.sh` - NEW: Helper script
9. `scripts/train_vit.py` - NEW: Quick ViT training script

## Summary

We've transformed the pipeline to handle advanced architectures with proper regularization and ensemble techniques. The live dashboard will let you monitor the 100-trial optimization in real-time as it searches for the best model configuration.
