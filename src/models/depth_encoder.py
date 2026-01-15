"""
Depth estimation and RGB+Depth fusion models for biomass prediction.

Uses pretrained monocular depth models to extract depth information,
which correlates with vegetation height and biomass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class DepthEstimator(nn.Module):
    """
    Wrapper for pretrained monocular depth estimation models.

    Supports:
    - Depth Anything v2 (recommended, best quality)
    - MiDaS (lighter, faster)
    """

    def __init__(
        self,
        model_type: str = 'depth_anything_v2_small',
        freeze: bool = True,
        output_size: Optional[Tuple[int, int]] = None
    ):
        """
        Args:
            model_type: Which depth model to use:
                - 'depth_anything_v2_small': Small, fast, good quality
                - 'depth_anything_v2_base': Better quality, slower
                - 'midas_small': MiDaS small model
            freeze: Whether to freeze depth model weights
            output_size: Optional (H, W) to resize depth output
        """
        super().__init__()
        self.model_type = model_type
        self.output_size = output_size
        self.freeze = freeze

        # Load the appropriate model
        if 'depth_anything' in model_type:
            self._load_depth_anything(model_type)
        elif 'midas' in model_type:
            self._load_midas(model_type)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        if freeze:
            self._freeze_weights()

    def _load_depth_anything(self, model_type: str):
        """Load Depth Anything v2 model from HuggingFace."""
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        model_id_map = {
            'depth_anything_v2_small': 'depth-anything/Depth-Anything-V2-Small-hf',
            'depth_anything_v2_base': 'depth-anything/Depth-Anything-V2-Base-hf',
        }

        model_id = model_id_map.get(model_type, model_id_map['depth_anything_v2_small'])

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
        self.model_family = 'depth_anything'

    def _load_midas(self, model_type: str):
        """Load MiDaS model from torch hub."""
        model_id_map = {
            'midas_small': 'MiDaS_small',
            'midas_large': 'DPT_Large',
        }

        model_id = model_id_map.get(model_type, 'MiDaS_small')

        self.model = torch.hub.load('intel-isl/MiDaS', model_id, trust_repo=True)
        self.transforms = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)

        if 'small' in model_type:
            self.midas_transform = self.transforms.small_transform
        else:
            self.midas_transform = self.transforms.dpt_transform

        self.model_family = 'midas'
        self.processor = None

    def _freeze_weights(self):
        """Freeze all model weights."""
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Estimate depth from RGB images.

        Args:
            images: Tensor of shape (B, 3, H, W), normalized to ImageNet stats

        Returns:
            depth: Tensor of shape (B, 1, H, W) with relative depth values
        """
        B, C, H, W = images.shape

        if self.model_family == 'depth_anything':
            # Depth Anything expects specific preprocessing
            # Since we already have normalized images, we need to handle this
            with torch.no_grad() if self.freeze else torch.enable_grad():
                outputs = self.model(images)
                depth = outputs.predicted_depth

                # Interpolate to original size
                depth = F.interpolate(
                    depth.unsqueeze(1),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                )
        else:
            # MiDaS
            with torch.no_grad() if self.freeze else torch.enable_grad():
                depth = self.model(images)
                depth = depth.unsqueeze(1)

                if depth.shape[-2:] != (H, W):
                    depth = F.interpolate(
                        depth,
                        size=(H, W),
                        mode='bilinear',
                        align_corners=False
                    )

        # Normalize depth to [0, 1] range per image
        depth = self._normalize_depth(depth)

        if self.output_size is not None:
            depth = F.interpolate(
                depth,
                size=self.output_size,
                mode='bilinear',
                align_corners=False
            )

        return depth

    def _normalize_depth(self, depth: torch.Tensor) -> torch.Tensor:
        """Normalize depth to [0, 1] range per image."""
        B = depth.shape[0]
        depth_flat = depth.view(B, -1)
        depth_min = depth_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        depth_max = depth_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        depth = (depth - depth_min) / (depth_max - depth_min + 1e-8)
        return depth


class RGBDepthFusionEncoder(nn.Module):
    """
    Dual-encoder model that fuses RGB and depth features for biomass prediction.

    Architecture:
        RGB Image ──> RGB Encoder ──────┐
                                        ├──> Fusion ──> Regression Heads
        Depth Map ──> Depth Encoder ────┘
    """

    def __init__(
        self,
        rgb_backbone: str = 'efficientnetv2_rw_m',
        depth_model: str = 'depth_anything_v2_small',
        fusion_type: str = 'concat',
        num_targets: int = 5,
        dropout: float = 0.3,
        freeze_depth: bool = True,
        pretrained: bool = True
    ):
        """
        Args:
            rgb_backbone: timm model name for RGB encoder
            depth_model: Depth estimation model type
            fusion_type: How to fuse features ('concat', 'add', 'attention')
            num_targets: Number of regression targets
            dropout: Dropout rate
            freeze_depth: Whether to freeze depth model
            pretrained: Use pretrained weights for RGB encoder
        """
        super().__init__()

        import timm

        self.fusion_type = fusion_type
        self.target_names = [
            'Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g'
        ]

        # RGB Encoder
        self.rgb_encoder = timm.create_model(
            rgb_backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool='avg'
        )
        rgb_features = self.rgb_encoder.num_features

        # Depth Estimator
        self.depth_estimator = DepthEstimator(
            model_type=depth_model,
            freeze=freeze_depth
        )

        # Depth Feature Encoder (lightweight CNN to encode depth map)
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        depth_features = 256

        # Fusion layer
        if fusion_type == 'concat':
            fused_features = rgb_features + depth_features
            self.fusion = nn.Identity()
        elif fusion_type == 'add':
            # Project both to same dimension then add
            self.rgb_proj = nn.Linear(rgb_features, 512)
            self.depth_proj = nn.Linear(depth_features, 512)
            fused_features = 512
            self.fusion = lambda rgb, depth: self.rgb_proj(rgb) + self.depth_proj(depth)
        elif fusion_type == 'attention':
            # Cross-attention fusion
            self.rgb_proj = nn.Linear(rgb_features, 512)
            self.depth_proj = nn.Linear(depth_features, 512)
            self.attention = nn.MultiheadAttention(512, num_heads=8, batch_first=True)
            fused_features = 512
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        # Task-specific regression heads
        self.heads = nn.ModuleDict()
        for target_name in self.target_names:
            self.heads[target_name] = nn.Sequential(
                nn.Linear(fused_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1)
            )

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with RGB and depth fusion.

        Args:
            images: RGB images tensor (B, 3, H, W)

        Returns:
            Dict mapping target names to predictions (B, 1)
        """
        # Extract RGB features
        rgb_features = self.rgb_encoder(images)

        # Estimate depth and extract depth features
        with torch.no_grad():
            depth_maps = self.depth_estimator(images)
        depth_features = self.depth_encoder(depth_maps)

        # Fuse features
        if self.fusion_type == 'concat':
            fused = torch.cat([rgb_features, depth_features], dim=1)
        elif self.fusion_type == 'add':
            fused = self.fusion(rgb_features, depth_features)
        elif self.fusion_type == 'attention':
            rgb_proj = self.rgb_proj(rgb_features).unsqueeze(1)
            depth_proj = self.depth_proj(depth_features).unsqueeze(1)
            combined = torch.cat([rgb_proj, depth_proj], dim=1)
            attended, _ = self.attention(combined, combined, combined)
            fused = attended.mean(dim=1)

        # Predict each target
        predictions = {}
        for target_name in self.target_names:
            predictions[target_name] = self.heads[target_name](fused).squeeze(-1)

        return predictions

    def get_depth_map(self, images: torch.Tensor) -> torch.Tensor:
        """Get depth maps for visualization."""
        with torch.no_grad():
            return self.depth_estimator(images)


class RGBDEncoder(nn.Module):
    """
    Simple 4-channel encoder (RGB + Depth as 4th channel).

    This is a simpler alternative to dual-encoder that just concatenates
    depth as a 4th input channel.
    """

    def __init__(
        self,
        backbone: str = 'efficientnetv2_rw_m',
        depth_model: str = 'depth_anything_v2_small',
        num_targets: int = 5,
        dropout: float = 0.3,
        freeze_depth: bool = True,
        pretrained: bool = True
    ):
        super().__init__()

        import timm

        self.target_names = [
            'Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g'
        ]

        # Depth Estimator
        self.depth_estimator = DepthEstimator(
            model_type=depth_model,
            freeze=freeze_depth
        )

        # Create backbone with modified first conv to accept 4 channels
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg',
            in_chans=4  # RGB + Depth
        )
        num_features = self.encoder.num_features

        # Regression heads
        self.heads = nn.ModuleDict()
        for target_name in self.target_names:
            self.heads[target_name] = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1)
            )

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with RGBD input.

        Args:
            images: RGB images tensor (B, 3, H, W)

        Returns:
            Dict mapping target names to predictions
        """
        # Estimate depth
        with torch.no_grad():
            depth = self.depth_estimator(images)

        # Concatenate RGB + Depth
        rgbd = torch.cat([images, depth], dim=1)

        # Extract features
        features = self.encoder(rgbd)

        # Predict each target
        predictions = {}
        for target_name in self.target_names:
            predictions[target_name] = self.heads[target_name](features).squeeze(-1)

        return predictions
