import random
import itertools
from torch import Tensor
from torch.utils.data import Dataset, Sampler
import copy


class CustomDataset(Dataset):
    def __init__(self, samples, targets):
        self.samples = samples
        self.targets = targets

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        target = self.targets[index]

        # Convert the sample and target to PyTorch tensors if needed
        sample = Tensor(sample)
        target = Tensor(target)

        return sample, target


class CustomTextDataset(Dataset):
    def __init__(self, samples, targets):
        self.samples = samples
        self.targets = targets

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        target = self.targets[index]

        # Convert the sample and target to PyTorch tensors if needed
        # text type sample = Tensor(sample)
        target = Tensor(target)

        return sample, target


class CustomSampler(Sampler):
    def __init__(self, indices):
        super(CustomSampler, self).__init__()
        random.shuffle(indices)
        self.indices = indices

    def set_index(self, indices, epoch=1):
        train_index = []
        for _ in range(epoch):
            copyed_indices = copy.deepcopy(indices)
            random.shuffle(copyed_indices)
            train_index.extend(copyed_indices)
        self.indices = train_index

    def __iter__(self):
        # 返回索引的迭代器
        return iter(self.indices)

    def __len__(self):
        # 返回索引的数量
        return len(self.indices)
