"""
Multi-task ResNet architecture for biomass regression.

Implements a shared ResNet backbone with separate prediction heads for each target.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import timm

from src.config import TARGET_NAMES, NUM_TARGETS


class MultiTaskResNet(nn.Module):
    """
    Multi-task model with shared backbone and separate heads for each target.

    Args:
        backbone: Backbone architecture (resnet18/34/50/101, efficientnet_b0/b1/b4,
                  vit_base_patch16_224/384, convnext_base/large, swin_base_patch4_window7_224,
                  tf_efficientnetv2_m/l, or any timm model)
        num_targets: Number of targets to predict (default: 5)
        pretrained: Use ImageNet pretrained weights
        dropout: Dropout probability in prediction heads
        head_hidden_dim: Hidden layer size in prediction heads
    """

    def __init__(
        self,
        backbone='resnet34',
        num_targets=NUM_TARGETS,
        pretrained=True,
        dropout=0.3,
        head_hidden_dim=256
    ):
        super(MultiTaskResNet, self).__init__()

        self.backbone_name = backbone
        self.num_targets = num_targets

        # Load backbone
        if backbone.startswith('resnet'):
            self.backbone, backbone_out_features = self._load_resnet(backbone, pretrained)
        elif backbone.startswith('efficientnet'):
            self.backbone, backbone_out_features = self._load_efficientnet(backbone, pretrained)
        else:
            # Use timm for all other models (ViT, ConvNeXt, Swin, etc.)
            self.backbone, backbone_out_features = self._load_timm_model(backbone, pretrained)

        self.backbone_out_features = backbone_out_features

        # Create separate prediction heads for each target
        self.heads = nn.ModuleDict({
            target_name: self._make_head(backbone_out_features, head_hidden_dim, dropout)
            for target_name in TARGET_NAMES
        })

    def _load_resnet(self, backbone, pretrained):
        """Load ResNet backbone."""
        if backbone == 'resnet18':
            model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            out_features = 512
        elif backbone == 'resnet34':
            model = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
            out_features = 512
        elif backbone == 'resnet50':
            model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            out_features = 2048
        elif backbone == 'resnet101':
            model = models.resnet101(weights='IMAGENET1K_V1' if pretrained else None)
            out_features = 2048
        else:
            raise ValueError(f"Unknown ResNet variant: {backbone}")

        # Remove final FC layer
        model.fc = nn.Identity()

        return model, out_features

    def _load_efficientnet(self, backbone, pretrained):
        """Load EfficientNet backbone from timm."""
        model = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        out_features = model.num_features
        return model, out_features

    def _load_timm_model(self, backbone, pretrained):
        """Load any timm model (ViT, ConvNeXt, Swin, etc.)."""
        model = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        out_features = model.num_features
        return model, out_features

    def _make_head(self, in_features, hidden_dim, dropout):
        """
        Create a prediction head with one hidden layer.

        Args:
            in_features: Input feature dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability

        Returns:
            Sequential module for prediction head
        """
        return nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch_size, 3, H, W]

        Returns:
            Dict of predictions for each target, shape [batch_size] for each
        """
        # Extract features from backbone
        features = self.backbone(x)  # [batch_size, backbone_out_features]

        # Pass through each prediction head
        outputs = {}
        for target_name, head in self.heads.items():
            output = head(features).squeeze(-1)  # [batch_size, 1] -> [batch_size]
            outputs[target_name] = output

        return outputs


class ConstraintAwareResNet(nn.Module):
    """
    Multi-task ResNet with hard constraint: Dry_Total = Dry_Clover + Dry_Dead + Dry_Green.

    Only predicts 4 targets (Clover, Dead, Green, GDM), derives Total from components.
    Applies ReLU to ensure non-negative predictions.

    Args:
        backbone: Backbone architecture
        pretrained: Use ImageNet pretrained weights
        dropout: Dropout probability in prediction heads
        head_hidden_dim: Hidden layer size in prediction heads
    """

    def __init__(
        self,
        backbone='resnet34',
        pretrained=True,
        dropout=0.3,
        head_hidden_dim=256
    ):
        super(ConstraintAwareResNet, self).__init__()

        self.backbone_name = backbone

        # Load backbone (same as MultiTaskResNet)
        if backbone.startswith('resnet'):
            self.backbone, backbone_out_features = self._load_resnet(backbone, pretrained)
        elif backbone.startswith('efficientnet'):
            self.backbone, backbone_out_features = self._load_efficientnet(backbone, pretrained)
        else:
            # Use timm for all other models (ViT, ConvNeXt, Swin, etc.)
            self.backbone, backbone_out_features = self._load_timm_model(backbone, pretrained)

        # Create prediction heads (only for 4 targets, not Total)
        self.heads = nn.ModuleDict({
            'Dry_Clover_g': self._make_head(backbone_out_features, head_hidden_dim, dropout),
            'Dry_Dead_g': self._make_head(backbone_out_features, head_hidden_dim, dropout),
            'Dry_Green_g': self._make_head(backbone_out_features, head_hidden_dim, dropout),
            'GDM_g': self._make_head(backbone_out_features, head_hidden_dim, dropout),
        })

    def _load_resnet(self, backbone, pretrained):
        """Load ResNet backbone (same as MultiTaskResNet)."""
        if backbone == 'resnet18':
            model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            out_features = 512
        elif backbone == 'resnet34':
            model = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
            out_features = 512
        elif backbone == 'resnet50':
            model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            out_features = 2048
        elif backbone == 'resnet101':
            model = models.resnet101(weights='IMAGENET1K_V1' if pretrained else None)
            out_features = 2048
        else:
            raise ValueError(f"Unknown ResNet variant: {backbone}")
        model.fc = nn.Identity()
        return model, out_features

    def _load_efficientnet(self, backbone, pretrained):
        """Load EfficientNet backbone from timm."""
        model = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        out_features = model.num_features
        return model, out_features

    def _load_timm_model(self, backbone, pretrained):
        """Load any timm model (ViT, ConvNeXt, Swin, etc.)."""
        model = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        out_features = model.num_features
        return model, out_features

    def _make_head(self, in_features, hidden_dim, dropout):
        """Create a prediction head with one hidden layer."""
        return nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        Forward pass with hard constraint enforcement.

        Args:
            x: Input tensor of shape [batch_size, 3, H, W]

        Returns:
            Dict of predictions for all 5 targets, with Total derived from components
        """
        # Extract features
        features = self.backbone(x)

        # Predict components and GDM
        clover = self.heads['Dry_Clover_g'](features).squeeze(-1)
        dead = self.heads['Dry_Dead_g'](features).squeeze(-1)
        green = self.heads['Dry_Green_g'](features).squeeze(-1)
        gdm = self.heads['GDM_g'](features).squeeze(-1)

        # Apply ReLU to ensure non-negative predictions
        clover = torch.relu(clover)
        dead = torch.relu(dead)
        green = torch.relu(green)
        gdm = torch.relu(gdm)

        # Hard constraint: Total = Clover + Dead + Green
        total = clover + dead + green

        return {
            'Dry_Clover_g': clover,
            'Dry_Dead_g': dead,
            'Dry_Green_g': green,
            'Dry_Total_g': total,
            'GDM_g': gdm,
        }


def count_parameters(model):
    """
    Count trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model(backbone='resnet34', batch_size=4, image_size=512):
    """
    Test model forward pass.

    Args:
        backbone: Backbone architecture
        batch_size: Batch size for dummy input
        image_size: Image size

    Returns:
        None (prints model info and output shapes)
    """
    print(f"\nTesting {backbone} model:")
    print("-" * 60)

    # Create model
    model = MultiTaskResNet(backbone=backbone, pretrained=False)
    model.eval()

    # Count parameters
    n_params = count_parameters(model)
    print(f"Trainable parameters: {n_params:,}")

    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, image_size, image_size)

    # Forward pass
    with torch.no_grad():
        outputs = model(dummy_input)

    # Print output shapes
    print("\nOutput shapes:")
    for target_name, output in outputs.items():
        print(f"  {target_name}: {output.shape}")

    print("\nModel test passed!")


if __name__ == '__main__':
    # Test different backbones
    for backbone in ['resnet18', 'resnet34', 'resnet50', 'efficientnet_b0']:
        test_model(backbone=backbone)
        print()
