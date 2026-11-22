
"""Image transformations for ReLoMAML few-shot classification experiments."""

from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    ToTensor,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transform(train=True):
    """Return the training augmentation or evaluation preprocessing pipeline."""
    transforms = []

    if train:
        transforms.extend([RandomHorizontalFlip(), RandomVerticalFlip()])

    transforms.extend([ToTensor(), Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return Compose(transforms)
