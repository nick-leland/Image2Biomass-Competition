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
            - model_type: 'standard', 'rgbd', or 'rgb_depth_fusion'
            - backbone: timm model name
            - depth_model: depth estimation model (for rgbd/fusion)
            - fusion_type: 'concat', 'add', 'attention' (for fusion)
            - constraint_mode: 'none', 'hard', 'soft'
            - pretrained, dropout, head_hidden_dim: usual params

    Returns:
        PyTorch model
    """
    model_type = config.get('model_type', 'standard')
    constraint_mode = config.get('constraint_mode', 'none')

    if model_type == 'rgb_depth_fusion':
        # Dual encoder with RGB and depth features fused
        from src.models.depth_encoder import RGBDepthFusionEncoder
        model = RGBDepthFusionEncoder(
            rgb_backbone=config.get('backbone', 'efficientnetv2_rw_m'),
            depth_model=config.get('depth_model', 'depth_anything_v2_small'),
            fusion_type=config.get('fusion_type', 'concat'),
            dropout=config.get('dropout', 0.3),
            freeze_depth=config.get('freeze_depth', True),
            pretrained=config.get('pretrained', True)
        )

    elif model_type == 'rgbd':
        # 4-channel input (RGB + Depth as 4th channel)
        from src.models.depth_encoder import RGBDEncoder
        model = RGBDEncoder(
            backbone=config.get('backbone', 'efficientnetv2_rw_m'),
            depth_model=config.get('depth_model', 'depth_anything_v2_small'),
            dropout=config.get('dropout', 0.3),
            freeze_depth=config.get('freeze_depth', True),
            pretrained=config.get('pretrained', True)
        )

    elif constraint_mode == 'hard':
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
