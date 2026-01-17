"""
Vegetation indices computed from RGB images.

These indices help quantify vegetation presence and health from standard RGB images.
They're fast to compute and can improve biomass prediction.
"""

import numpy as np
import torch


def compute_vegetation_indices_np(image: np.ndarray) -> np.ndarray:
    """
    Compute vegetation indices from RGB image (numpy).

    Args:
        image: RGB image as numpy array, shape (H, W, 3), values in [0, 255] or [0, 1]

    Returns:
        Stacked indices as numpy array, shape (H, W, N_indices)
    """
    # Ensure float and normalize to 0-1 if needed
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    # Avoid division by zero
    eps = 1e-8

    # 1. Excess Green Index (ExG)
    # Highlights green vegetation
    ExG = 2 * G - R - B

    # 2. Excess Green minus Excess Red (ExGR)
    # Better discrimination between vegetation and soil
    ExR = 1.4 * R - G
    ExGR = ExG - ExR

    # 3. Green Ratio (simple but effective)
    total = R + G + B + eps
    green_ratio = G / total

    # 4. Normalized Green-Red Difference Index (NGRDI)
    # Similar to NDVI but for RGB
    NGRDI = (G - R) / (G + R + eps)

    # 5. Visible Atmospherically Resistant Index (VARI)
    # Estimates vegetation fraction
    VARI = (G - R) / (G + R - B + eps)
    # Clip extreme values
    VARI = np.clip(VARI, -1, 1)

    # 6. Green Leaf Index (GLI)
    GLI = (2 * G - R - B) / (2 * G + R + B + eps)

    # Stack all indices
    indices = np.stack([ExG, ExGR, green_ratio, NGRDI, VARI, GLI], axis=-1)

    return indices


def compute_vegetation_indices_torch(image: torch.Tensor) -> torch.Tensor:
    """
    Compute vegetation indices from RGB image (torch tensor).

    Args:
        image: RGB image as torch tensor, shape (3, H, W) or (B, 3, H, W)
            Assumed to be normalized with ImageNet stats or in [0, 1]

    Returns:
        Stacked indices as torch tensor, shape (N_indices, H, W) or (B, N_indices, H, W)
    """
    # Handle batch dimension
    if image.dim() == 3:
        image = image.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    # Denormalize if ImageNet normalized (approximate)
    # ImageNet mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
    img = image * std + mean
    img = torch.clamp(img, 0, 1)

    R = img[:, 0, :, :]
    G = img[:, 1, :, :]
    B = img[:, 2, :, :]

    eps = 1e-8

    # 1. Excess Green Index (ExG)
    ExG = 2 * G - R - B

    # 2. Excess Green minus Excess Red (ExGR)
    ExR = 1.4 * R - G
    ExGR = ExG - ExR

    # 3. Green Ratio
    total = R + G + B + eps
    green_ratio = G / total

    # 4. Normalized Green-Red Difference Index (NGRDI)
    NGRDI = (G - R) / (G + R + eps)

    # 5. VARI
    VARI = (G - R) / (G + R - B + eps)
    VARI = torch.clamp(VARI, -1, 1)

    # 6. Green Leaf Index (GLI)
    GLI = (2 * G - R - B) / (2 * G + R + B + eps)

    # Stack: (B, 6, H, W)
    indices = torch.stack([ExG, ExGR, green_ratio, NGRDI, VARI, GLI], dim=1)

    if squeeze_output:
        indices = indices.squeeze(0)

    return indices


# For easy import
VEGETATION_INDEX_NAMES = ['ExG', 'ExGR', 'green_ratio', 'NGRDI', 'VARI', 'GLI']
N_VEGETATION_INDICES = len(VEGETATION_INDEX_NAMES)


if __name__ == '__main__':
    # Test
    import cv2
    from pathlib import Path

    # Load a test image
    img_path = Path('/home/chaot/kaggle/Image2Biomass-Competition/train')
    img_files = list(img_path.glob('*.jpg'))

    if img_files:
        img = cv2.imread(str(img_files[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        indices = compute_vegetation_indices_np(img)
        print(f"Input shape: {img.shape}")
        print(f"Output shape: {indices.shape}")
        print(f"Index names: {VEGETATION_INDEX_NAMES}")

        for i, name in enumerate(VEGETATION_INDEX_NAMES):
            idx = indices[:, :, i]
            print(f"  {name}: min={idx.min():.3f}, max={idx.max():.3f}, mean={idx.mean():.3f}")
