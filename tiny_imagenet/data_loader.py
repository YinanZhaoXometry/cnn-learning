import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

from ._prepare_tiny_imagenet_data import prepare_tiny_imagenet_data

_NORMALIZE = transforms.Normalize(
    mean=[0.4802, 0.4481, 0.3975],
    std=[0.2302, 0.2265, 0.2262],
)
_EVAL_TRANSFORM = transforms.Compose([transforms.ToTensor(), _NORMALIZE])


def _make_loader(dataset, batch_size, sampler=None, shuffle=False):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )


def get_train_valid_data_loader(
    batch_size,
    augment=True,
    random_seed=1,
    valid_size=0.1,
    shuffle=True,
    source="local",
    local_dir="./dataset-tiny-imagenet-200",
    data_dir="./data",
):
    root = prepare_tiny_imagenet_data(
        source=source, local_dir=local_dir, data_dir=data_dir
    )

    if augment:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                _NORMALIZE,
            ]
        )
    else:
        train_transform = _EVAL_TRANSFORM

    train_dataset = datasets.ImageFolder(root=root / "train", transform=train_transform)
    valid_dataset = datasets.ImageFolder(root=root / "train", transform=_EVAL_TRANSFORM)

    indices = list(range(len(train_dataset)))
    split = int(np.floor(valid_size * len(train_dataset)))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_loader = _make_loader(
        train_dataset, batch_size, sampler=SubsetRandomSampler(indices[split:])
    )
    valid_loader = _make_loader(
        valid_dataset, batch_size, sampler=SubsetRandomSampler(indices[:split])
    )
    return train_loader, valid_loader


def get_test_data_loader(
    batch_size,
    shuffle=False,
    source="local",
    local_dir="./dataset-tiny-imagenet-200",
    data_dir="./data",
):
    root = prepare_tiny_imagenet_data(source=source, local_dir=local_dir, data_dir=data_dir)
    dataset = datasets.ImageFolder(
        root=root / "val_by_class", transform=_EVAL_TRANSFORM
    )
    return _make_loader(dataset, batch_size, shuffle=shuffle)
