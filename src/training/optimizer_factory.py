"""
Factory for creating optimizers and learning rate schedulers.
"""

import torch
from typing import Dict, Optional


def create_optimizer(model, config: Dict):
    """
    Create optimizer based on configuration.

    Args:
        model: PyTorch model
        config: Configuration dict with optimizer settings

    Returns:
        PyTorch optimizer
    """
    optimizer_name = config.get('optimizer', 'adamw')
    lr = config.get('learning_rate', 1e-4)
    weight_decay = config.get('weight_decay', 1e-4)

    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    elif optimizer_name == 'sgd':
        momentum = config.get('momentum', 0.9)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )

    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizer


def create_scheduler(optimizer, config: Dict, steps_per_epoch: Optional[int] = None):
    """
    Create learning rate scheduler based on configuration.

    Args:
        optimizer: PyTorch optimizer
        config: Configuration dict with scheduler settings
        steps_per_epoch: Number of steps per epoch (for OneCycleLR)

    Returns:
        PyTorch scheduler or None
    """
    scheduler_name = config.get('scheduler', None)

    if scheduler_name is None or scheduler_name == 'none':
        return None

    elif scheduler_name == 'cosine':
        T_max = config.get('T_max', 50)
        eta_min = config.get('eta_min', 1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min
        )

    elif scheduler_name == 'plateau':
        factor = config.get('scheduler_factor', 0.5)
        patience = config.get('scheduler_patience', 5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience
        )

    elif scheduler_name == 'step':
        step_size = config.get('step_size', 10)
        gamma = config.get('gamma', 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )

    elif scheduler_name == 'exponential':
        gamma = config.get('gamma', 0.95)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma
        )

    elif scheduler_name == 'onecycle':
        max_lr = config.get('learning_rate', 1e-3)
        epochs = config.get('num_epochs', 50)

        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch required for OneCycleLR")

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs
        )

    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    return scheduler
