# CSIRO Image2Biomass Competition - Forum Research Findings

*Research conducted: January 2026*
*Current LB Score: 0.50 | Target: 0.68+*

---

## Table of Contents
1. [Top Performing Approaches](#top-performing-approaches)
2. [Model Architectures](#model-architectures)
3. [Training Strategies](#training-strategies)
4. [Post-Processing Techniques](#post-processing-techniques)
5. [Cross-Validation Insights](#cross-validation-insights)
6. [Data Characteristics](#data-characteristics)
7. [Known Challenges](#known-challenges)
8. [Action Items](#action-items)

---

## Top Performing Approaches

### 3-Model Ensemble (Score: 0.68)
The best public notebook uses a combination of three different model types:

1. **SigLIP + ML Ensemble**
   - Patch-based SigLIP embeddings
   - Fed into multiple regressors (Ridge, Lasso, BayesianRidge, SVR, GradientBoosting, ExtraTrees, PLS)
   - Predictions averaged across regressors

2. **MVP Image Model Ensemble**
   - 10 checkpoints split into two groups (A: first 5, B: last 5)
   - Each group runs TTA
   - Combined with: `pred = 0.95 * pred_A + 0.075 * pred_B`

3. **DINOv2 Image Model (5-fold)**
   - 5-fold cross-validation checkpoints
   - Averaged fold outputs
   - TTA applied

### Test-Time Augmentation (TTA)
All top solutions use TTA with these transforms:
- Resize
- HorizontalFlip
- VerticalFlip
- RandomRotate90

---

## Model Architectures

### Vision Transformers (Most Popular)

| Model | Notes | Reported Scores |
|-------|-------|-----------------|
| DINOv2-large | Recommended over huge | ~0.67 |
| DINOv3-large | Good balance of performance/compute | ~0.67 |
| DINOv3-huge | May overfit on small dataset | ~0.69 |
| SigLIP | Patch-based embeddings work well | - |
| Swin Transformer | SSL approach being explored | - |

### Key Finding: DINOv3-huge vs DINOv3-large
- DINOv3-huge has embedding size of 1280
- Larger model doesn't always perform better
- DINOv3-large preferred for compute efficiency
- Most experiments done on large variant

### Patch-Based Approach
```python
# Split images into patches for SigLIP
patch_size = 520
overlap = 16
stride = patch_size - overlap
```
- Images split into overlapping patches
- Each patch gets embedding
- Embeddings aggregated for final prediction

---

## Training Strategies

### Multi-Regressor Ensemble
Top solutions use multiple sklearn regressors on extracted embeddings:
- LinearRegression
- Ridge
- Lasso
- BayesianRidge
- SVR / LinearSVR
- GradientBoostingRegressor
- HistGradientBoostingRegressor
- ExtraTreesRegressor
- PLSRegression

### Teacher-Student Distillation
- Knowledge distillation techniques being explored
- Teacher model trains student for better generalization

### Training Tips
- More training time generally improves CV scores
- Watch for overfitting - CV may not correlate with LB
- Use proper random seeding for reproducibility

---

## Post-Processing Techniques

### Biological Constraints (Critical!)
The 5 targets have biological relationships that should be enforced:

```python
# Enforce non-negativity
predictions = predictions.clip(min=0)

# Enforce biological constraints
# GDM (Green Dry Matter) = Green + Clover
# Total = GDM + Dead

def post_process_biomass(df):
    # Reconcile predictions to be consistent
    df['GDM_g'] = df['Dry_Green_g'] + df['Dry_Clover_g']
    df['Dry_Total_g'] = df['GDM_g'] + df['Dry_Dead_g']
    return df
```

### Target Relationships
```
Dry_Total_g = Dry_Green_g + Dry_Clover_g + Dry_Dead_g
GDM_g = Dry_Green_g + Dry_Clover_g
```

---

## Cross-Validation Insights

### Data Leakage Warning!
**Critical finding**: Multiple images come from the same farm/location

- Typically 4-6 very similar images per location
- Images share: State, Sampling_Date, and similar targets
- Standard KFold will leak information!

### Recommended CV Strategy
```python
from sklearn.model_selection import GroupKFold

# Group by location/farm identifier
# Images from same location must be in same fold
group_kfold = GroupKFold(n_splits=5)
```

### CV Score Benchmarks
| Target | Achievable R2 |
|--------|---------------|
| Dry_Dead_g | ~0.63 |
| Dry_Clover_g | ~0.85 |
| Dry_Green_g | ~0.87 |
| GDM_g | ~0.855 |
| Dry_Total_g | ~0.79 |

**Overall CV**: 0.80-0.85 is achievable with good models

### CV vs Leaderboard Correlation
- **Warning**: CV scores don't correlate well with LB
- Private LB has very limited samples
- Expect high variance in final rankings
- Don't overfit to public LB

---

## Data Characteristics

### Camera Types
Two distinct camera types in the dataset:
- Smartphones (heavy noise reduction, sharpening, "waxy" textures)
- DSLRs (different processing characteristics)

**Detection method**:
```python
import cv2

def analyze_image_quality(image_path):
    img = cv2.imread(image_path, 0)  # grayscale

    # Blurriness (Laplacian Variance)
    blur_score = cv2.Laplacian(img, cv2.CV_64F).var()

    # Noise estimation
    # Smooth image and subtract from original
    smoothed = cv2.GaussianBlur(img, (5, 5), 0)
    noise = img.astype(float) - smoothed.astype(float)
    noise_score = noise.std()

    return blur_score, noise_score
```

**Finding**: Camera type clusters correlate with State - may not add much value beyond State feature.

### Image Groupings
- Images taken same day in same field
- Some images have timestamps in EXIF data
- State and Sampling_Date are important grouping variables

---

## Known Challenges

### 1. Dead Matter Visibility
> "The dead matter is probably hidden underneath the grass which we can't see from the image"
> - bogoconic1 (10th place)

- Dead matter often buried under live vegetation
- Explains why Dry_Dead_g has lowest R2 (~0.63)
- Visual features alone may be insufficient

### 2. Similar Images, Different Targets
Example from forum: Two nearly identical images have:
- Image A: 10g dead matter
- Image B: 50g dead matter

Even humans cannot distinguish these visually!

### 3. Small Dataset
- Only 357 training images
- High risk of overfitting
- Need strong regularization and augmentation

### 4. Competition Metric
- Weighted average of R2 scores per target
- Not exactly mean R2 - check official metric implementation
- Some targets weighted differently

---

## Action Items

### Immediate Improvements (High Impact)
- [ ] Implement proper GroupKFold CV (avoid location leakage)
- [ ] Add post-processing constraints (GDM = Green + Clover, etc.)
- [ ] Add TTA (HorizontalFlip, VerticalFlip, RandomRotate90)
- [ ] Clip predictions to non-negative values

### Model Upgrades
- [ ] Try DINOv2/DINOv3 backbone instead of EfficientNet
- [ ] Experiment with SigLIP patch-based embeddings
- [ ] Implement multi-regressor ensemble on top of embeddings
- [ ] Consider teacher-student distillation

### Ensemble Strategy
- [ ] Train multiple model types (DINO, SigLIP, current EfficientNet)
- [ ] Blend predictions with optimized weights
- [ ] Use different folds/checkpoints for diversity

### Data Augmentation
- [ ] Increase augmentation strength
- [ ] Add RandomRotate90 if not present
- [ ] Consider MixUp/CutMix for regularization

---

## References

### Public Notebooks
- [0.68] CSIRO | 3-Model Ensemble + Post-process (24 upvotes)
- CSIRO Image2Biomass Prediction - VK (Score: 0.67)
- Teacher-Student Distillation
- ssl swin (Swin Transformer approach)

### Key Discussion Threads
- "identify the camera?" - Camera type analysis
- "CV scores - final phase of the competition" - CV benchmarks
- "Does Dinov3-huge underperform Dinov3-large?" - Model selection
- "10 gms vs 50 gms Dead" - Task difficulty illustration

---

*Document generated from Kaggle forum research. Update as new insights emerge.*
