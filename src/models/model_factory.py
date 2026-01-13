"""
Factory for creating models based on configuration.
"""

from typing import Dict
from src.models.resnet_baseline import MultiTaskResNet, ConstraintAwareResNet


def create_model(config: Dict):
    """
    Create model based on configuration.

    Args:
        config: Configuration dict with model settings

    Returns:
        PyTorch model
    """
    constraint_mode = config.get('constraint_mode', 'none')

    if constraint_mode == 'hard':
        # Hard constraint: predict 4 targets, derive Total from sum
        model = ConstraintAwareResNet(
            backbone=config.get('backbone', 'resnet34'),
            pretrained=config.get('pretrained', True),
            dropout=config.get('dropout', 0.3),
            head_hidden_dim=config.get('head_hidden_dim', 256)
        )
    else:
        # Standard multi-task model or soft constraint (handled by loss function)
        model = MultiTaskResNet(
            backbone=config.get('backbone', 'resnet34'),
            pretrained=config.get('pretrained', True),
            dropout=config.get('dropout', 0.3),
            head_hidden_dim=config.get('head_hidden_dim', 256)
        )

    return model
