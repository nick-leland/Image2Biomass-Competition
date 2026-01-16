# CSIRO Image2Biomass Competition

PyTorch-based solution for the [CSIRO Image2Biomass Kaggle competition](https://www.kaggle.com/competitions/csiro-biomass) - predicting biomass measurements from pasture images.

## Leaderboard Progress

| Version | Model | Kaggle R² | Key Changes |
|---------|-------|-----------|-------------|
| V1 | ResNet34 baseline | 0.30 | Initial submission |
| V2 | EfficientNetV2-M | 0.42 | Better backbone |
| V3 | 5-Fold Ensemble | 0.46 | K-fold cross-validation |
| V4 | MSE Loss + Tuning | 0.50 | Loss aligned with metric |
| **V5** | **RGB+Depth Fusion** | **0.57** | Depth Anything v2 + external data |
| V6 | Ensemble (V4+V5) | Pending | Weighted ensemble |

**Current Best: V5 with R² = 0.57**

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

**Evaluation Metric**: R² (coefficient of determination)

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
│   ├── prepare_external_data.py       # Process GrassClover dataset
│   ├── generate_submission_depth.py   # Generate depth model submission
│   ├── generate_submission_kfold.py   # Generate baseline submission
│   └── package_depth_model_kaggle.py  # Package models for Kaggle
├── kaggle_upload/
│   ├── ensemble_v4_v5.tar.gz          # V6 ensemble models (5.3 GB)
│   ├── depth_fusion_model.tar.gz      # V5 depth fusion (3.4 GB)
│   └── depth_fusion_model_componly.tar.gz  # V5 without external data
├── kaggle_inference_notebook_v6_ensemble.ipynb    # V6 submission notebook
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
| Depth fusion | +0.07 R² | Biggest single improvement |
| External data | +0.03 R² | Danish GrassClover dataset |
| GroupKFold CV | Better generalization | Prevents location leakage |
| MSE loss | +0.04 R² | Better aligned with R² metric |
| 5-fold ensemble | +0.04 R² | Reduces variance |
| Biological constraints | Cleaner predictions | Post-processing enforcement |
| TTA (8 transforms) | +0.01-0.02 R² | More stable predictions |

## What Didn't Work

| Technique | Result | Notes |
|-----------|--------|-------|
| Vision Transformers | Worse | Too large for 357 images, overfitted |
| Aggressive augmentation | Worse | Conservative augmentation better |
| Hard constraint learning | No improvement | Post-processing works better |
| Training on competition data only (depth model) | Worse CV but timed out on Kaggle | External data crucial |

## Future Ideas

- [ ] Try DINOv2 or SigLIP backbones
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
