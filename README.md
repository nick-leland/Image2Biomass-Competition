# CSIRO Image2Biomass Competition

PyTorch-based solution for the [CSIRO Image2Biomass Kaggle competition](https://www.kaggle.com/competitions/csiro-biomass) - predicting biomass measurements from pasture images.

## Leaderboard Progress

| Version | Model | CV R² | LB R² | Key Changes |
|---------|-------|-------|-------|-------------|
| V1 | ResNet34 baseline | - | 0.30 | Initial submission |
| V2 | EfficientNetV2-M | - | 0.42 | Better backbone |
| V3 | 5-Fold Ensemble | - | 0.46 | K-fold cross-validation |
| V4 | MSE Loss + Tuning | 0.74 | 0.50 | Overfitting detected |
| V5 | RGB+Depth Fusion | 0.48 | 0.57 | Depth Anything v2 + external data |
| V6 | Ensemble (V4+V5) | - | 0.57 | Weighted ensemble (no improvement) |
| **V7** | **DINOv2 ViT-Base** | **0.50** | **0.58** | Foundation model backbone |
| V8 | DINOv2 + Depth | 0.50 | Pending | Foundation model + depth fusion |

**Current Best: V7 with R² = 0.58**

## Competition Overview

**Task**: Multi-target regression predicting 5 biomass measurements from pasture images:
- `Dry_Clover_g` - Dry clover weight in grams
- `Dry_Dead_g` - Dry dead matter weight in grams
- `Dry_Green_g` - Dry green matter weight in grams
- `Dry_Total_g` - Total dry matter weight in grams
- `GDM_g` - Green dry matter weight in grams

**Biological Constraints**:
- GDM = Dry_Green + Dry_Clover
- Total = GDM + Dry_Dead

**Dataset**: 357 training images + 261 external images (Danish GrassClover dataset)

**Evaluation Metric**: Weighted R² with log-stabilizing transform:
- `Dry_Total_g`: 50% weight
- `GDM_g`: 20% weight
- `Dry_Green_g`, `Dry_Dead_g`, `Dry_Clover_g`: 10% each

## Key Innovations

### 1. RGB+Depth Fusion Architecture

Our best model uses a dual-encoder architecture that combines RGB features with depth information:

```
RGB Image ──> EfficientNetV2-M ──────┐
                                     ├──> Concat ──> Regression Heads
Depth Map ──> Lightweight CNN ───────┘
     ↑
     └── Depth Anything v2 (frozen)
```

**Why depth helps**: Vegetation height correlates with biomass. Monocular depth estimation captures this 3D structure from 2D images.

### 2. External Data Integration

Added 261 images from the Danish GrassClover dataset (AILab Denmark):
- Mapped columns: `dry_grass` -> `Dry_Green_g`, `dry_clover` -> `Dry_Clover_g`, etc.
- Despite domain shift (Denmark vs Australia), external data improved generalization (+0.07 R²)

### 3. GroupKFold Cross-Validation

Prevents data leakage by keeping images from the same location together:
- Groups by: `State + Sampling_Date`
- Ensures model generalizes to new locations, not just memorizes

### 4. Biological Constraints Post-Processing

Enforces known relationships in predictions:
```python
# Enforce GDM = Green + Clover
adjusted_gdm = (predicted_gdm + (green + clover)) / 2

# Enforce Total = GDM + Dead
adjusted_total = (predicted_total + (gdm + dead)) / 2
```

### 5. Test-Time Augmentation (TTA)

8 transforms (4 flips x 2 rotations) averaged for more stable predictions.

## Architecture Details

### V7/V8: DINOv2 Foundation Model

| Component | Details |
|-----------|---------|
| RGB Encoder | DINOv2 ViT-Base (vit_base_patch14_dinov2, 86M params) |
| Depth Model | Depth Anything v2 Small (V8 only, frozen) |
| Depth Encoder | Lightweight CNN (4 conv layers -> 256 features) |
| Fusion | V7: RGB only (768 features), V8: Concat (768 + 256 = 1024 features) |
| Heads | 5 separate MLPs (256 -> 64 -> 1) |
| Image Size | 518x518 (native DINOv2 resolution) |
| Training | 30 epochs, AdamW, warmup + cosine LR, gradient accumulation 8 |
| Learning Rate | Backbone: 1e-5, Heads: 1e-4 (differential) |

### V5: RGB+Depth Fusion Model

| Component | Details |
|-----------|---------|
| RGB Encoder | EfficientNetV2-RW-M (54M params) |
| Depth Model | Depth Anything v2 Small (25M params, frozen) |
| Depth Encoder | Lightweight CNN (4 conv layers -> 256 features) |
| Fusion | Concatenation (1792 + 256 = 2048 features) |
| Heads | 5 separate MLPs (256 -> 64 -> 1) |
| Image Size | 384x384 |
| Training | 30 epochs, AdamW, cosine LR schedule |

### V4: Baseline Model

| Component | Details |
|-----------|---------|
| Backbone | EfficientNetV2-M (tf_efficientnetv2_m) |
| Heads | 5 separate MLPs (512 hidden dim) |
| Image Size | 512x512 |
| Loss | MSE |
| Training | 50 epochs, 5-fold CV |

## Repository Structure

```
Image2Biomass-Competition/
├── src/
│   ├── data/
│   │   ├── dataset.py          # BiomassDataset with multi-path support
│   │   ├── transforms.py       # Augmentations + TTA transforms
│   │   └── splitter.py         # GroupKFold + stratified splits
│   ├── models/
│   │   ├── depth_encoder.py    # DepthEstimator, RGBDepthFusionEncoder
│   │   ├── model_factory.py    # Model creation with config
│   │   └── multi_task_model.py # Base multi-task architecture
│   ├── training/
│   │   └── trainer.py          # Training loop with early stopping
│   └── utils/
│       └── submission.py       # Constraint enforcement, submission generation
├── scripts/
│   ├── train_depth_model.py           # Train RGB+Depth fusion
│   ├── train_kfold_mse.py             # Train baseline with K-fold
│   ├── train_foundation_models.py     # Train DINOv2/SigLIP models
│   ├── prepare_external_data.py       # Process GrassClover dataset
│   ├── generate_submission_depth.py   # Generate depth model submission
│   ├── generate_submission_kfold.py   # Generate baseline submission
│   ├── package_depth_model_kaggle.py  # Package models for Kaggle
│   ├── package_foundation_model_kaggle.py  # Package foundation models
│   └── package_dinov2_depth_kaggle.py # Package DINOv2+Depth models
├── kaggle_upload/
│   ├── dinov2_depth_kaggle.tar.gz     # V8 DINOv2+Depth (5.1 GB)
│   ├── dinov2_foundation_kaggle.tar.gz # V7 DINOv2 (4.5 GB)
│   ├── ensemble_v4_v5.tar.gz          # V6 ensemble models (5.3 GB)
│   ├── depth_fusion_model.tar.gz      # V5 depth fusion (3.4 GB)
│   └── depth_fusion_model_componly.tar.gz  # V5 without external data
├── kaggle_inference_notebook_v8_dinov2_depth.ipynb # V8 submission notebook
├── kaggle_inference_notebook_v7_foundation.ipynb   # V7 submission notebook
├── kaggle_inference_notebook_v6_ensemble.ipynb     # V6 submission notebook
├── kaggle_inference_notebook_v5_depth_fusion.ipynb # V5 submission notebook
├── CLAUDE.md              # Instructions for Claude Code
├── COMPETITION_INSIGHTS.md # Research findings from Kaggle forums
└── README.md              # This file
```

## Setup

```bash
# Clone repository
git clone https://github.com/nick-leland/Image2Biomass-Competition.git
cd Image2Biomass-Competition

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies (using uv)
uv pip install torch torchvision timm albumentations pandas scikit-learn
uv pip install transformers  # For Depth Anything v2
```

## Training

### Train Depth Fusion Model (V5)

```bash
# With external data (best results)
python scripts/train_depth_model.py --use_external --epochs 30 --n_folds 5

# Competition data only
python scripts/train_depth_model.py --epochs 30 --n_folds 5
```

### Train Baseline Model (V4)

```bash
python scripts/train_kfold_mse.py --n_folds 5 --epochs 50
```

## Generating Submissions

```bash
# V5: Depth fusion
python scripts/generate_submission_depth.py --checkpoint_dir experiments/checkpoints_depth_*

# V4: Baseline
python scripts/generate_submission_kfold.py --checkpoint_dir experiments/checkpoints_kfold_mse
```

## Kaggle Submission

1. Upload model package to Kaggle Datasets
2. Create notebook from `kaggle_inference_notebook_v6_ensemble.ipynb`
3. Add model dataset + competition data
4. **Set Internet to OFF**
5. Submit

## What Worked

| Technique | Impact | Notes |
|-----------|--------|-------|
| DINOv2 foundation model | +0.01 R² | Best backbone so far (V7) |
| Depth fusion | +0.07 R² | Biggest single improvement |
| External data | +0.03 R² | Danish GrassClover dataset |
| GroupKFold CV | Better generalization | Prevents location leakage |
| MSE loss | +0.04 R² | Better aligned with R² metric |
| 5-fold ensemble | +0.04 R² | Reduces variance |
| Biological constraints | Cleaner predictions | Post-processing enforcement |
| TTA (8 transforms) | +0.01-0.02 R² | More stable predictions |
| Differential learning rates | Stability | 10x lower LR for backbone vs heads |
| Gradient accumulation | Effective batch 32 | Better with foundation models |

## What Didn't Work

| Technique | Result | Notes |
|-----------|--------|-------|
| ViT without proper training | Worse | Needed differential LR + warmup + grad accumulation |
| SigLIP backbone | Val loss 2.85 | DINOv2 (2.18) worked better |
| V4+V5 ensemble | No improvement | V6 got same 0.57 as V5 alone |
| Aggressive augmentation | Worse | Conservative augmentation better |
| Hard constraint learning | No improvement | Post-processing works better |
| Training on competition data only (depth model) | Worse CV but timed out on Kaggle | External data crucial |

## CV vs Leaderboard Correlation

We discovered that **CV score does not directly predict LB score** - the gap between them reveals generalization:

| Model | CV R² | CV Std | LB R² | Gap | Interpretation |
|-------|-------|--------|-------|-----|----------------|
| V4 (EfficientNetV2) | 0.74 | 0.04 | 0.50 | -0.24 | Severe overfitting |
| V5 (Depth + external) | 0.48 | 0.30 | 0.57 | +0.09 | Good generalization |
| V7 (DINOv2) | 0.50 | 0.09 | 0.58 | +0.08 | Good generalization |
| V8 (DINOv2 + Depth) | 0.50 | 0.07 | Pending | ? | Lowest variance |

### Key Insights

1. **High CV can mean overfitting**: V4 had the highest CV (0.74) but lowest LB (0.50). The model memorized training data.

2. **Foundation models generalize better**: DINOv2's pre-trained representations transfer well to unseen data, even with lower CV scores.

3. **Lower variance = better generalization**: Models with consistent performance across folds (low std) tend to perform better on the test set.

4. **The competition metric matters**: We were tracking MSE loss, but the actual metric is weighted R² with log transform. Different targets have different weights:
   - `Dry_Total_g`: 50%
   - `GDM_g`: 20%
   - `Dry_Green_g`, `Dry_Dead_g`, `Dry_Clover_g`: 10% each

5. **Fold 2 is problematic**: Across all models, Fold 2 shows poor performance due to +113% distribution shift in `Dry_Clover_g` between train and validation.

## Future Ideas

- [x] Try DINOv2 or SigLIP backbones (DINOv2 worked best - V7)
- [x] DINOv2 + Depth fusion (V8 - pending results)
- [ ] Attention fusion instead of concatenation
- [ ] More external data sources
- [ ] Pseudo-labeling on test set
- [ ] Stacking ensemble with meta-learner

## Acknowledgments

Built with Claude Code (claude.ai/code) using:
- PyTorch 2.6+
- timm (PyTorch Image Models)
- Transformers (Depth Anything v2)
- Albumentations

External data from [AILab Denmark GrassClover dataset](https://huggingface.co/datasets/AILabDenmark/GrassClover).

## License

MIT License - See competition rules for data usage restrictions.
