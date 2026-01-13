"""
Configuration file for Image2Biomass competition.

Contains paths, constants, and default hyperparameters.
"""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT
TRAIN_CSV = PROJECT_ROOT / 'train.csv'
TEST_CSV = PROJECT_ROOT / 'test.csv'
TRAIN_IMG_DIR = PROJECT_ROOT / 'train'
TEST_IMG_DIR = PROJECT_ROOT / 'test'
SAMPLE_SUBMISSION = PROJECT_ROOT / 'sample_submission.csv'

# Experiments
EXPERIMENTS_DIR = PROJECT_ROOT / 'experiments'
CHECKPOINTS_DIR = EXPERIMENTS_DIR / 'checkpoints'
LOGS_DIR = EXPERIMENTS_DIR / 'logs'
OPTUNA_DIR = EXPERIMENTS_DIR / 'optuna_studies'
SUBMISSIONS_DIR = PROJECT_ROOT / 'submissions'

# Create directories
for dir_path in [EXPERIMENTS_DIR, CHECKPOINTS_DIR, LOGS_DIR, OPTUNA_DIR, SUBMISSIONS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Target names
TARGET_NAMES = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
NUM_TARGETS = len(TARGET_NAMES)

# Image properties
IMAGE_HEIGHT = 1000
IMAGE_WIDTH = 2000
IMAGE_CHANNELS = 3

# ImageNet normalization (for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Default training configuration
DEFAULT_CONFIG = {
    # Model architecture
    'backbone': 'resnet34',
    'pretrained': True,
    'dropout': 0.3,
    'head_hidden_dim': 256,

    # Constraint handling
    'constraint_mode': 'none',  # Options: 'none', 'soft', 'hard'
    'constraint_weight': 0.1,  # For soft constraint

    # Loss function
    'loss_function': 'mse',  # Options: 'mse', 'huber'
    'huber_delta': 1.0,

    # Optimizer
    'optimizer': 'adamw',
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'momentum': 0.9,  # For SGD

    # Scheduler
    'scheduler': 'cosine',  # Options: 'cosine', 'plateau', 'step', 'onecycle', None
    'scheduler_patience': 5,  # For plateau
    'T_max': 50,  # For cosine

    # Data
    'augmentation_level': 'moderate',  # Options: 'conservative', 'moderate', 'aggressive'
    'image_size': 512,
    'batch_size': 16,
    'num_workers': 4,

    # Training
    'num_epochs': 50,
    'early_stopping_patience': 10,
    'device': 'cuda',  # Will be checked at runtime

    # Task weights (for multi-task loss)
    'weight_clover': 1.0,
    'weight_dead': 1.0,
    'weight_green': 1.0,
    'weight_total': 1.0,
    'weight_gdm': 1.0,

    # Regularization
    'use_mixup': False,
    'mixup_alpha': 0.2,
    'label_smoothing': 0.0,

    # Reproducibility
    'seed': 42,

    # Validation
    'val_split': 0.2,
    'stratify_by': 'state',  # Options: 'state', 'species', 'target_bin', 'combined'
}

# Metadata columns
METADATA_COLS = ['State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm', 'Sampling_Date']

# Data statistics (from EDA)
DATA_STATS = {
    'n_train_images': 357,
    'n_test_images': 1,
    'n_states': 4,
    'n_species': 15,
}
