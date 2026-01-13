# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Personal Preferences

- **Package Manager**: Use `uv` instead of `pip` for all package installations
- **Typography**: No em dashes or emojis in code or documentation
- **ASCII Faces**: ASCII emoticons are welcome! :D

## Competition Overview

This repository contains code for the CSIRO Image2Biomass Kaggle competition, which predicts biomass measurements from pasture images.

**Task**: Multi-target regression predicting 5 biomass measurements per image:
- `Dry_Clover_g` - Dry clover weight in grams
- `Dry_Dead_g` - Dry dead matter weight in grams
- `Dry_Green_g` - Dry green matter weight in grams
- `Dry_Total_g` - Total dry matter weight in grams
- `GDM_g` - Green dry matter weight in grams

## Data Structure

### Dataset Layout
```
train/               # 357 training images (JPG format, ~2-4MB each)
test/                # 1 test image
train.csv            # 1,785 rows (357 images × 5 targets each)
test.csv             # 5 rows (1 image × 5 targets)
sample_submission.csv # Expected submission format: sample_id,target
csiro-biomass.zip    # Original data archive (1.1GB)
```

### CSV Format
The training data is in **long format** where each image appears 5 times (once per target):
- Each row has a unique `sample_id` = `{image_id}__{target_name}`
- `image_path`: relative path to JPG (e.g., `train/ID1011485656.jpg`)
- `Sampling_Date`, `State`, `Species`: metadata about the sample
- `Pre_GSHH_NDVI`: NDVI value before sampling
- `Height_Ave_cm`: average plant height in centimeters
- `target_name`: which of the 5 biomass measurements this row represents
- `target`: the actual biomass value in grams (float)

### Prediction Format
Submissions must match the long format with columns: `sample_id,target`

## Environment

**Python Environment**: `.venv/` (virtual environment present)
- Activate with: `source .venv/bin/activate`
- Currently minimal packages installed (numpy 2.4.0 present)

**Typical ML Stack** (not yet installed, add as needed):
```bash
uv pip install torch torchvision  # or tensorflow/keras
uv pip install pandas scikit-learn matplotlib pillow
uv pip install albumentations  # for image augmentation
```

## Git Workflow

**IMPORTANT**: Follow this branching strategy:

1. **DEV Branch**: Continuously commit all work-in-progress to the `dev` branch
   - Commit frequently as you make progress
   - Push regularly to keep remote updated
   - This is the active development branch

2. **Main Branch**: Merge to `main` only when making a Kaggle submission
   - Main should only contain code that has been submitted to the competition
   - Each merge to main represents a competition submission milestone
   - Tag merges with submission scores when available

**Example workflow**:
```bash
# Working on improvements
git checkout dev
git add .
git commit -m "Add feature X"
git push origin dev

# Ready to submit
git checkout main
git merge dev
git push origin main
# Submit to Kaggle, then tag with score
git tag -a "submission-v2-score-0.25" -m "EfficientNetV2-M with TTA"
git push origin --tags
```

## Development Workflow

1. **Exploratory Data Analysis**: Start with EDA notebooks to understand image characteristics, target distributions, and correlations between features
2. **Baseline Model**: Build simple baseline (e.g., linear regression on image features)
3. **Deep Learning Models**: CNNs/Vision Transformers for image feature extraction
4. **Multi-target Strategy**: Handle 5 correlated targets (shared backbone vs. separate heads)
5. **Submission**: Generate predictions in the correct long format for `sample_submission.csv`

## Notes

- No existing Python scripts or notebooks yet - this is a fresh competition setup
- Image data is large (~1GB compressed) - consider efficient data loading strategies
- Targets are continuous regression values (grams of biomass)
- Consider using metadata features (NDVI, height, species, state) alongside images
- The 5 targets are biologically related and may benefit from multi-task learning
