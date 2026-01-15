"""
Data augmentation strategies using albumentations.

Provides three preset augmentation levels: conservative, moderate, and aggressive.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config import IMAGENET_MEAN, IMAGENET_STD


def get_train_transforms(level='moderate', image_size=512):
    """
    Get training transforms based on augmentation level.

    Args:
        level: Augmentation level ('conservative', 'moderate', 'aggressive')
        image_size: Target image size (will resize to square)

    Returns:
        Albumentations Compose object
    """
    if level == 'conservative':
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    elif level == 'moderate':
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    elif level == 'aggressive':
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=15,
                p=0.5
            ),
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
                p=0.6
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=5),
            ], p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.3
            ),
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                p=0.2
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    elif level == 'extreme':
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.2,
                scale_limit=0.3,
                rotate_limit=30,
                p=0.7
            ),
            A.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.15,
                p=0.7
            ),
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.GaussianBlur(blur_limit=(3, 9), p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
            ], p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.6
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.CoarseDropout(
                max_holes=12,
                max_height=48,
                max_width=48,
                p=0.3
            ),
            A.RandomShadow(p=0.2),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    else:
        raise ValueError(f"Unknown augmentation level: {level}. Choose 'conservative', 'moderate', 'aggressive', or 'extreme'")


def get_val_transforms(image_size=512):
    """
    Get validation/test transforms (no augmentation).

    Args:
        image_size: Target image size (will resize to square)

    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_tta_transforms(image_size=512, include_rotations=True):
    """
    Get test-time augmentation (TTA) transforms.

    Returns a list of transforms for TTA ensemble. Top Kaggle solutions use
    flips + rotations for best results.

    Args:
        image_size: Target image size
        include_rotations: If True, includes 90/180/270 degree rotations (8 total transforms)
                          If False, only uses flips (4 total transforms)

    Returns:
        List of Albumentations Compose objects
    """
    transforms = []

    # Define flip combinations
    flip_configs = [
        (False, False),  # Original
        (True, False),   # Horizontal flip
        (False, True),   # Vertical flip
        (True, True),    # Both flips
    ]

    # Define rotation angles (0, 90, 180, 270 degrees)
    # Note: 180 = hflip + vflip, so we only need 0 and 90 when combined with flips
    rotation_angles = [0, 90] if include_rotations else [0]

    for hflip, vflip in flip_configs:
        for angle in rotation_angles:
            transform_list = [A.Resize(image_size, image_size)]

            if hflip:
                transform_list.append(A.HorizontalFlip(p=1.0))
            if vflip:
                transform_list.append(A.VerticalFlip(p=1.0))
            if angle > 0:
                # Rotate by fixed angle
                transform_list.append(A.Rotate(limit=(angle, angle), p=1.0, border_mode=0))

            transform_list.extend([
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ])

            transforms.append(A.Compose(transform_list))

    return transforms


def visualize_augmentations(image, transform, n_samples=9):
    """
    Visualize augmentations on a single image.

    Args:
        image: NumPy array (H, W, C)
        transform: Albumentations transform
        n_samples: Number of augmented samples to show

    Returns:
        None (displays plot)
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(n_samples):
        augmented = transform(image=image)
        aug_image = augmented['image']

        # Convert tensor to numpy for display
        if len(aug_image.shape) == 3 and aug_image.shape[0] == 3:
            # Tensor format [C, H, W] -> [H, W, C]
            aug_image = aug_image.permute(1, 2, 0).numpy()
            # Denormalize
            aug_image = aug_image * IMAGENET_STD + IMAGENET_MEAN
            aug_image = np.clip(aug_image, 0, 1)

        axes[i].imshow(aug_image)
        axes[i].axis('off')
        axes[i].set_title(f'Sample {i+1}')

    plt.tight_layout()
    plt.show()
