import json
import os
from pathlib import Path
from typing import List, Type, Dict, Callable
import requests
import numpy as np
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import torch
import torchvision


class BaseDataset(Dataset):
    def __init__(self) -> None:
        self.classes: List = None
        self.data = None
        self.targets = None
        self.data_transform = None
        self.target_transform = None

    def __getitem__(self, index):
        data, targets = self.data[index], self.targets[index]
        if self.data_transform is not None:
            data = self.data_transform(data)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        return data, targets

    def __len__(self):
        return len(self.targets)


class FEMNIST(BaseDataset):
    def __init__(
            self,
            root,
            args=None,
            general_data_transform=None,
            general_target_transform=None,
            train_data_transform=None,
            train_target_transform=None,
    ) -> None:
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        if not os.path.isfile(root / "data.npy") or not os.path.isfile(
                root / "targets.npy"
        ):
            raise RuntimeError(
                "run data/utils/run.py -d femnist for generating the data.npy and targets.npy first."
            )

        data = np.load(root / "data.npy")
        targets = np.load(root / "targets.npy")

        self.data = torch.from_numpy(data).float().reshape(-1, 1, 28, 28)
        self.targets = torch.from_numpy(targets).long()
        self.classes = list(range(62))
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


class CINIC10(datasets.ImageFolder):
    def __init__(
            self,
            root,
            which: str
    ):
        super().__init__(root=f"{root}/{which}", transform=DATA_TRANSFORMS['cinic10'][which])


class CIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, root, which):
        super().__init__(root=root, train=True if which == 'train' else False, download=True,
                         transform=DATA_TRANSFORMS['cifar100'][which])


def download_csv(url, local_filename):
    """
    Download a CSV from a URL and save it to a local file.

    Parameters:
    - url (str): The URL of the CSV file.
    - local_filename (str): The local path where the CSV should be saved.
    """
    response = requests.get(url)

    # Ensure the request was successful.
    response.raise_for_status()

    with open(local_filename, 'wb') as f:
        f.write(response.content)


DATASETS: Dict[str, Type[BaseDataset]] = {
    "femnist": FEMNIST,
    "cinic10": CINIC10,
    "cifar100": CIFAR100,
}

DATA_NUM_CLASSES_DICT: Dict[str, int] = {
    "femnist": 62,
    "cinic10": 10,
    "cifar100": 100,
}
DATA_TRANSFORMS = {
    "cinic10": {
        "train": transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835)),
        ]),
        "test": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835)),
        ])
    },
    "cifar100": {
        "train": transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        ])
    }
}
