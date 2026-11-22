
"""Few-shot dataset selection for ReLoMAML experiments."""

from datasets.helpers import miniimagenet, tieredimagenet


DATASET_HELPERS = {
    'miniImageNet': miniimagenet,
    'tieredImageNet': tieredimagenet,
}


def get_dataset(dataset, folder, N_ways, K_shots_for_support, K_shots_for_query, download, train_transform, test_transform):
    """Return meta-training, validation, and test datasets."""
    if dataset not in DATASET_HELPERS:
        raise ValueError('Unsupported dataset: {}. Choose from: {}.'.format(dataset, ', '.join(DATASET_HELPERS)))

    dataset_helper = DATASET_HELPERS[dataset]
    common_kwargs = dict(shots=K_shots_for_support, ways=N_ways, shuffle=True, test_shots=K_shots_for_query, download=download)

    meta_train_dataset = dataset_helper(folder, meta_train=True, transform=train_transform, **common_kwargs)
    meta_val_dataset = dataset_helper(folder, meta_val=True, transform=test_transform, **common_kwargs)
    meta_test_dataset = dataset_helper(folder, meta_test=True, transform=test_transform, **common_kwargs)

    return meta_train_dataset, meta_val_dataset, meta_test_dataset
