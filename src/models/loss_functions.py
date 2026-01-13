"""
Loss functions for multi-task biomass regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class MultiTaskMSELoss(nn.Module):
    """
    Weighted MSE loss for multi-task regression.

    Args:
        task_weights: Dict mapping target names to loss weights
    """

    def __init__(self, task_weights=None):
        super(MultiTaskMSELoss, self).__init__()
        self.task_weights = task_weights or {}

    def forward(self, predictions, targets):
        """
        Compute weighted MSE loss.

        Args:
            predictions: Dict of predictions [batch_size] for each target
            targets: Dict of targets [batch_size] for each target

        Returns:
            total_loss: Weighted sum of losses
            loss_dict: Dict of individual losses for logging
        """
        losses = {}
        total_loss = 0.0

        for target_name in predictions.keys():
            pred = predictions[target_name]
            target = targets[target_name]

            loss = F.mse_loss(pred, target)
            losses[f'{target_name}_loss'] = loss.item()

            weight = self.task_weights.get(target_name, 1.0)
            total_loss += weight * loss

        return total_loss, losses


class MultiTaskHuberLoss(nn.Module):
    """
    Weighted Huber loss for multi-task regression (robust to outliers).

    Args:
        delta: Huber loss threshold (default: 1.0)
        task_weights: Dict mapping target names to loss weights
    """

    def __init__(self, delta=1.0, task_weights=None):
        super(MultiTaskHuberLoss, self).__init__()
        self.delta = delta
        self.task_weights = task_weights or {}

    def forward(self, predictions, targets):
        """
        Compute weighted Huber loss.

        Args:
            predictions: Dict of predictions [batch_size] for each target
            targets: Dict of targets [batch_size] for each target

        Returns:
            total_loss: Weighted sum of losses
            loss_dict: Dict of individual losses for logging
        """
        losses = {}
        total_loss = 0.0

        for target_name in predictions.keys():
            pred = predictions[target_name]
            target = targets[target_name]

            loss = F.smooth_l1_loss(pred, target, beta=self.delta)
            losses[f'{target_name}_loss'] = loss.item()

            weight = self.task_weights.get(target_name, 1.0)
            total_loss += weight * loss

        return total_loss, losses


class ConstraintAwareLoss(nn.Module):
    """
    MSE loss + soft penalty for violating the constraint: Dry_Total = Clover + Dead + Green.

    Args:
        constraint_weight: Weight for constraint violation penalty (default: 0.1)
        task_weights: Dict mapping target names to loss weights
    """

    def __init__(self, constraint_weight=0.1, task_weights=None):
        super(ConstraintAwareLoss, self).__init__()
        self.constraint_weight = constraint_weight
        self.task_weights = task_weights or {}

    def forward(self, predictions, targets):
        """
        Compute MSE loss + constraint penalty.

        Args:
            predictions: Dict of predictions [batch_size] for each target
            targets: Dict of targets [batch_size] for each target

        Returns:
            total_loss: MSE + constraint penalty
            loss_dict: Dict of individual losses for logging
        """
        losses = {}
        total_loss = 0.0

        # Standard MSE loss for each target
        for target_name in predictions.keys():
            pred = predictions[target_name]
            target = targets[target_name]

            loss = F.mse_loss(pred, target)
            losses[f'{target_name}_loss'] = loss.item()

            weight = self.task_weights.get(target_name, 1.0)
            total_loss += weight * loss

        # Constraint violation penalty
        pred_total = predictions['Dry_Total_g']
        pred_sum = (predictions['Dry_Clover_g'] +
                    predictions['Dry_Dead_g'] +
                    predictions['Dry_Green_g'])

        constraint_violation = F.mse_loss(pred_total, pred_sum)
        losses['constraint_violation'] = constraint_violation.item()

        total_loss += self.constraint_weight * constraint_violation

        return total_loss, losses


def create_loss_function(config: Dict):
    """
    Create loss function based on configuration.

    Args:
        config: Configuration dict with loss settings

    Returns:
        Loss function module
    """
    loss_function = config.get('loss_function', 'mse')

    # Task weights
    task_weights = {
        'Dry_Clover_g': config.get('weight_clover', 1.0),
        'Dry_Dead_g': config.get('weight_dead', 1.0),
        'Dry_Green_g': config.get('weight_green', 1.0),
        'Dry_Total_g': config.get('weight_total', 1.0),
        'GDM_g': config.get('weight_gdm', 1.0),
    }

    if loss_function == 'mse':
        return MultiTaskMSELoss(task_weights=task_weights)

    elif loss_function == 'huber':
        delta = config.get('huber_delta', 1.0)
        return MultiTaskHuberLoss(delta=delta, task_weights=task_weights)

    elif loss_function == 'constraint_aware':
        constraint_weight = config.get('constraint_weight', 0.1)
        return ConstraintAwareLoss(
            constraint_weight=constraint_weight,
            task_weights=task_weights
        )

    else:
        raise ValueError(f"Unknown loss function: {loss_function}")
