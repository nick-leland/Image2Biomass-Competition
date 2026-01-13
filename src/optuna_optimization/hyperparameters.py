"""
Hyperparameter search space definition for Optuna.
"""

from typing import Dict, Any
import optuna


def define_hyperparameter_space(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Define the hyperparameter search space for Optuna.

    Args:
        trial: Optuna trial object

    Returns:
        config: Dictionary of hyperparameters for this trial
    """
    config = {}

    # Model architecture - ADVANCED MODELS
    config['backbone'] = trial.suggest_categorical(
        'backbone',
        [
            # Vision Transformers - Small models for limited data
            'vit_tiny_patch16_224',
            'vit_tiny_patch16_384',
            'vit_small_patch16_224',
            'vit_small_patch16_384',
            'vit_base_patch16_224',
            'vit_base_patch16_384',
            # ConvNeXt (modern CNNs)
            'convnext_tiny',
            'convnext_small',
            'convnext_base',
            # Advanced EfficientNets
            'tf_efficientnetv2_s',
            'tf_efficientnetv2_m',
            # Swin Transformer
            'swin_tiny_patch4_window7_224',
            'swin_small_patch4_window7_224',
            # Still include best from before
            'resnet50',
            'efficientnet_b4'
        ]
    )
    config['pretrained'] = True  # Always use pretrained
    config['dropout'] = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
    config['head_hidden_dim'] = trial.suggest_categorical(
        'head_hidden_dim',
        [128, 256, 512]
    )

    # Constraint handling
    config['constraint_mode'] = trial.suggest_categorical(
        'constraint_mode',
        ['none', 'soft']
    )
    if config['constraint_mode'] == 'soft':
        config['constraint_weight'] = trial.suggest_float(
            'constraint_weight',
            0.01, 0.5, log=True
        )
    else:
        config['constraint_weight'] = 0.0

    # Loss function
    config['loss_function'] = trial.suggest_categorical(
        'loss_function',
        ['mse', 'huber']
    )
    if config['loss_function'] == 'huber':
        config['huber_delta'] = trial.suggest_float('huber_delta', 0.5, 2.0, step=0.5)
    else:
        config['huber_delta'] = 1.0

    # Task weights (per-target loss weights)
    config['task_weights'] = {
        'Dry_Clover_g': trial.suggest_float('weight_clover', 0.5, 2.0, step=0.25),
        'Dry_Dead_g': trial.suggest_float('weight_dead', 0.5, 2.0, step=0.25),
        'Dry_Green_g': trial.suggest_float('weight_green', 0.5, 2.0, step=0.25),
        'Dry_Total_g': trial.suggest_float('weight_total', 0.5, 2.0, step=0.25),
        'GDM_g': trial.suggest_float('weight_gdm', 0.5, 2.0, step=0.25),
    }

    # Optimizer
    config['optimizer'] = trial.suggest_categorical(
        'optimizer',
        ['adam', 'adamw', 'sgd', 'rmsprop']
    )
    config['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    config['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    if config['optimizer'] == 'sgd':
        config['momentum'] = trial.suggest_float('momentum', 0.8, 0.99, step=0.05)
    else:
        config['momentum'] = 0.9  # Default for non-SGD

    # Scheduler
    config['scheduler'] = trial.suggest_categorical(
        'scheduler',
        ['cosine', 'plateau', 'step', 'onecycle', 'none']
    )
    if config['scheduler'] == 'plateau':
        config['scheduler_patience'] = trial.suggest_int('scheduler_patience', 3, 10)
    else:
        config['scheduler_patience'] = 5  # Default

    # Data augmentation
    config['augmentation_level'] = trial.suggest_categorical(
        'augmentation_level',
        ['conservative', 'moderate', 'aggressive', 'extreme']
    )
    config['image_size'] = trial.suggest_categorical('image_size', [224, 384, 448, 512])

    # Batch size
    config['batch_size'] = trial.suggest_categorical('batch_size', [8, 16, 32])

    # Training settings (fixed for optimization)
    config['num_epochs'] = 30  # Shorter for optimization speed
    config['early_stopping_patience'] = 10
    config['num_workers'] = 4
    config['seed'] = 42
    config['stratify_by'] = 'state'
    config['val_split'] = 0.2

    return config


def suggest_config_for_trial(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Convenience wrapper for defining hyperparameter space.

    Args:
        trial: Optuna trial object

    Returns:
        config: Dictionary of hyperparameters
    """
    return define_hyperparameter_space(trial)
