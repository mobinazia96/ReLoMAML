"""Few-shot dataset helpers for the ReLoMAML experiments.

Only miniImageNet and tieredImageNet are supported by this local helper module.
"""

import warnings

from torchmeta.datasets import MiniImagenet, TieredImagenet
from torchmeta.transforms import Categorical, ClassSplitter
from torchvision.transforms import Compose, Resize, ToTensor


__all__ = ["miniimagenet", "tieredimagenet"]


def helper_with_default(klass, folder, shots, ways, shuffle=True,
                        test_shots=None, seed=None, defaults=None, **kwargs):
    """Create an episodic dataset with per-class support/query splitting."""
    defaults = {} if defaults is None else defaults

    if "num_classes_per_task" in kwargs:
        warnings.warn(
            "Both `ways` and `num_classes_per_task` were specified; using "
            "`num_classes_per_task`.",
            stacklevel=2,
        )
        ways = kwargs.pop("num_classes_per_task")

    if "transform" not in kwargs:
        kwargs["transform"] = defaults.get("transform", ToTensor())
    if "target_transform" not in kwargs:
        kwargs["target_transform"] = defaults.get("target_transform", Categorical(ways))
    if "class_augmentations" not in kwargs:
        kwargs["class_augmentations"] = defaults.get("class_augmentations", None)
    if test_shots is None:
        test_shots = shots

    dataset = klass(folder, num_classes_per_task=ways, **kwargs)
    dataset = ClassSplitter(dataset, shuffle=shuffle,
                            num_train_per_class=shots, num_test_per_class=test_shots)
    dataset.seed(seed)
    return dataset


def miniimagenet(folder, shots, ways, shuffle=True, test_shots=None,
                 seed=None, **kwargs):
    """Create episodic miniImageNet tasks with 84-pixel default preprocessing."""
    defaults = {"transform": Compose([Resize(84), ToTensor()])}
    return helper_with_default(MiniImagenet, folder, shots, ways,
                               shuffle=shuffle, test_shots=test_shots,
                               seed=seed, defaults=defaults, **kwargs)


def tieredimagenet(folder, shots, ways, shuffle=True, test_shots=None,
                   seed=None, **kwargs):
    """Create episodic tieredImageNet tasks with 84-pixel default preprocessing."""
    defaults = {"transform": Compose([Resize(84), ToTensor()])}
    return helper_with_default(TieredImagenet, folder, shots, ways,
                               shuffle=shuffle, test_shots=test_shots,
                               seed=seed, defaults=defaults, **kwargs)
