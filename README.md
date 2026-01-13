# CSIRO Image2Biomass Competition

PyTorch-based solution for the [CSIRO Image2Biomass Kaggle competition](https://www.kaggle.com/competitions/csiro-biomass) - predicting biomass measurements from pasture images.

## Competition Overview

**Task**: Multi-target regression predicting 5 biomass measurements from pasture images:
- `Dry_Clover_g` - Dry clover weight in grams
- `Dry_Dead_g` - Dry dead matter weight in grams
- `Dry_Green_g` - Dry green matter weight in grams
- `Dry_Total_g` - Total dry matter weight in grams (constrained: = Clover + Dead + Green)
- `GDM_g` - Green dry matter weight in grams

**Dataset**: 357 training images, 1 test image (public leaderboard)

**Evaluation Metric**: Mean Absolute Error (MAE)

## Current Best Result

**Model**: EfficientNetV2-M with Test-Time Augmentation (TTA)
- **Validation Loss**: 0.3147 (Huber loss, delta=2.0)
- **Improvement**: 78% better than baseline ResNet34 (1.4178 → 0.3147)
- **Kaggle Score**: Pending submission results

## Approach

### Architecture Evolution

We explored multiple architectures through systematic Optuna hyperparameter optimization:

1. **Baseline (ResNet34)**: 1.4178 val loss
2. **Optimized ResNet50**: 0.4387 val loss (69% improvement)
3. **Vision Transformer Base (vit_base_patch16_384)**: 0.6640 val loss (too large for dataset)
4. **EfficientNetV2-M (Final)**: 0.3147 val loss (78% improvement) ✅

### Key Design Decisions

**Multi-Task Learning**
- Shared backbone with separate prediction heads for each target
- Task-specific loss weighting (found via Optuna):
  - Clover: 1.75, Dead: 0.75, Green: 1.0, Total: 1.0, GDM: 1.75
- Huber loss (delta=2.0) for robustness to outliers

**Constraint Enforcement**
- Biological constraint: `Dry_Total = Dry_Clover + Dry_Dead + Dry_Green`
- Method: Average predicted total with sum of components, then proportionally scale
- Result: Perfect constraint satisfaction (0g violation)

**Data Augmentation**
- Best: Conservative augmentation (horizontal/vertical flips, minimal color jitter)
- Finding: Aggressive augmentation hurt performance on small dataset (357 images)

**Test-Time Augmentation (TTA)**
- 4-way ensemble: Original + 3 flips (H, V, HV)
- Averages predictions for more stable results

### Hyperparameter Optimization

**Optuna Study**: 92 trials over ~14 hours
- Explored: 17 model architectures (ResNet, EfficientNet, ViT, ConvNeXt, Swin)
- Tuned: Learning rate, optimizer, scheduler, augmentation, image size, batch size, loss weights
- Best config found at trial 41 (EfficientNetV2-M)

**Search Space**:
```python
Backbones: [resnet50, efficientnet_b4,
            vit_tiny/small/base (224/384),
            convnext_tiny/small/base,
            efficientnetv2_s/m,
            swin_tiny/small]
Image sizes: [224, 384, 448, 512]
Augmentation: [conservative, moderate, aggressive, extreme]
Batch sizes: [8, 16, 32]
```

## What We Tried

### Successful Approaches ✅

1. **EfficientNetV2-M**: Best architecture for small dataset (54M params)
2. **Conservative augmentation**: Less is more with 357 images
3. **Large image size (512px)**: Higher resolution helped
4. **Huber loss**: More robust than MSE
5. **RMSprop optimizer**: Outperformed Adam/AdamW
6. **Test-Time Augmentation**: Improved stability
7. **Constraint enforcement**: Perfect satisfaction via averaging method

### Unsuccessful Approaches ❌

1. **Vision Transformers (ViT Base)**: Too large for dataset (86M params), overfitted
   - Val loss: 0.6640 vs ResNet50's 0.4387
2. **Aggressive augmentation**: Hurt performance
   - Conservative outperformed aggressive by ~15%
3. **Hard constraint learning**: Model learned relationships naturally
   - constraint_mode='none' worked best
4. **Larger image sizes (>512px)**: Diminishing returns + OOM issues

### Challenges Overcome

1. **GPU OOM errors**: Added memory cleanup between Optuna trials
2. **PyTorch 2.6+ compatibility**: Added `weights_only=False` to `torch.load()`
3. **Kaggle submission workflow**: Created internet-free inference notebook
4. **Constraint violations**: Implemented proportional scaling method (0g violation)

## Repository Structure

```
Image2Biomass-Competition/
├── src/
│   ├── data/              # Dataset, transforms, splitters
│   ├── models/            # Model architectures, loss functions
│   ├── training/          # Trainer, optimizers, schedulers
│   ├── optuna_optimization/ # Hyperparameter search
│   ├── evaluation/        # Metrics
│   └── utils/             # Submission, seeding, helpers
├── scripts/
│   ├── train_baseline.py            # Train ResNet baseline
│   ├── train_optimized.py           # Train with best config
│   ├── optimize_hyperparams.py      # Optuna optimization
│   ├── generate_submission_optimized.py # Generate submission with TTA
│   └── run_optuna_with_dashboard.sh # Launch optimization with live dashboard
├── notebooks/
│   ├── eda_initial.ipynb            # Exploratory data analysis
│   ├── image2biomass-efficientnetv2.ipynb  # Kaggle inference notebook
│   └── kaggle_inference_notebook.ipynb     # Alternative inference
├── CLAUDE.md              # Instructions for Claude Code
├── IMPROVEMENTS.md        # Latest improvements summary
└── README.md              # This file
```

## Setup

```bash
# Clone repository
git clone https://github.com/nick-leland/Image2Biomass-Competition.git
cd Image2Biomass-Competition

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (using uv for speed)
uv pip install torch torchvision timm albumentations pandas scikit-learn optuna optuna-dashboard
```

## Usage

### Training the Best Model

```bash
# Train EfficientNetV2-M with optimized hyperparameters
source .venv/bin/activate
python scripts/train_optimized.py \
    --config experiments/optuna_studies/advanced_models_20260111_013829/best_config.json \
    --checkpoint_dir experiments/checkpoints_final
```

### Generating Submissions

```bash
# Generate submission with TTA (4-flip ensemble)
python scripts/generate_submission_optimized.py
```

### Running Hyperparameter Optimization

```bash
# Start Optuna optimization with live dashboard
bash scripts/run_optuna_with_dashboard.sh 100 8080
# Opens dashboard at http://localhost:8080
```

## Results Timeline

| Model | Val Loss | Improvement | Notes |
|-------|----------|-------------|-------|
| ResNet34 baseline | 1.4178 | - | Initial baseline |
| ResNet50 (Optuna) | 0.4387 | 69% | First optimization pass |
| ViT Base 384 | 0.6640 | -51% | Too large, overfitted |
| **EfficientNetV2-M + TTA** | **0.3147** | **78%** | **Current best** |

## Key Findings

1. **Model size matters with small datasets**: EfficientNetV2-M (54M) > ViT Base (86M)
2. **Augmentation less is more**: Conservative > Aggressive with 357 images
3. **Resolution helps**: 512px > 384px > 224px
4. **Constraint learning**: Model learns naturally, no hard constraints needed
5. **TTA improves stability**: 4-flip ensemble reduces variance

## Future Work

- [ ] Ensemble multiple architectures (EfficientNetV2-M + ConvNeXt)
- [ ] Semi-supervised learning with unlabeled images
- [ ] Incorporate metadata (NDVI, height, state, species) as auxiliary inputs
- [ ] Cross-validation instead of single train/val split
- [ ] Fine-tune on full dataset (no validation split) for final submission
- [ ] Try EfficientNetV2-L or ConvNeXt-Base

## Competition Links

- **Competition**: https://www.kaggle.com/competitions/csiro-biomass
- **Leaderboard**: TBD (awaiting submission results)

## Acknowledgments

Built with Claude Code (claude.ai/code) using:
- PyTorch 2.6+
- timm (PyTorch Image Models)
- Albumentations for augmentation
- Optuna for hyperparameter optimization

## License

MIT License - See competition rules for data usage restrictions.
