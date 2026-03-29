"""
Shared augmentation and preprocessing transforms.
Provides train and eval transform factories based on config.
"""

from torchvision import transforms


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(
    img_size: int = 224,
    random_crop: bool = True,
    horizontal_flip: bool = True,
    rotation_degrees: float = 15.0,
    color_jitter: bool = True,
    color_jitter_strength: float = 0.2,
) -> transforms.Compose:
    """
    Build training transforms with data augmentation.
    Avoids aggressive distortions that could alter breed-specific features.
    """
    transform_list = []

    if random_crop:
        transform_list.append(transforms.RandomResizedCrop(
            img_size,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
        ))
    else:
        transform_list.append(transforms.Resize((img_size, img_size)))

    if horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    if rotation_degrees > 0:
        transform_list.append(transforms.RandomRotation(degrees=rotation_degrees))

    if color_jitter:
        transform_list.append(transforms.ColorJitter(
            brightness=color_jitter_strength,
            contrast=color_jitter_strength,
            saturation=color_jitter_strength,
            hue=color_jitter_strength * 0.5,
        ))

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return transforms.Compose(transform_list)


def get_eval_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Build evaluation/inference transforms (no augmentation).
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_denormalize_transform() -> transforms.Compose:
    """
    Inverse of ImageNet normalization for visualization.
    """
    return transforms.Compose([
        transforms.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0 / s for s in IMAGENET_STD],
        ),
        transforms.Normalize(
            mean=[-m for m in IMAGENET_MEAN],
            std=[1.0, 1.0, 1.0],
        ),
    ])
