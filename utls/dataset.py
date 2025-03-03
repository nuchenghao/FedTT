import random
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler, _DatasetKind
from torch.utils.data.dataloader import _BaseDataLoaderIter
import copy
import numpy as np
import warnings


class CustomSampler(Sampler):
    def __init__(self, indices):
        super(CustomSampler, self).__init__()
        assert isinstance(indices, torch.Tensor) == False , "the type of indices cannot be torch.Tensor"
        random.shuffle(indices)
        self.indices = indices

    def set_index(self, indices, epoch=1):
        assert isinstance(indices, torch.Tensor) == False , "the type of indices cannot be torch.Tensor"
        train_index = []
        for _ in range(epoch):
            copyed_indices = copy.deepcopy(indices)
            random.shuffle(copyed_indices)
            train_index.extend(copyed_indices)
        self.indices = train_index

    def __iter__(self):
        # 返回索引的迭代器；这里每次需要先shuffle一次；注意：random.shuffle只能作用在list和np.array中；不能作用在torch.tensor中，会出错
        assert isinstance(self.indices, torch.Tensor) == False , "the type of indices cannot be torch.Tensor"
        random.shuffle(self.indices)
        return iter(self.indices)

    def __len__(self):
        # 返回索引的数量
        return len(self.indices)

def NeedIndex_hack_indices(self):
    with torch.autograd.profiler.record_function(self._profile_name):
        if self._sampler_iter is None:
            self._reset()
        if isinstance(self._dataset, NeedIndexDataset):
            indices, data = self._next_data()
        else:
            data = self._next_data()
        self._num_yielded += 1
        if self._dataset_kind == _DatasetKind.Iterable and \
                self._IterableDataset_len_called is not None and \
                self._num_yielded > self._IterableDataset_len_called:
            warn_msg = ("Length of IterableDataset {} was reported to be {} (when accessing len(dataloader)), but {} "
                        "samples have been fetched. ").format(self._dataset, self._IterableDataset_len_called,
                                                              self._num_yielded)
            if self._num_workers > 0:
                warn_msg += ("For multiprocessing data-loading, this could be caused by not properly configuring the "
                             "IterableDataset replica at each worker. Please see "
                             "https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset for examples.")
            warnings.warn(warn_msg)
        if isinstance(self._dataset, NeedIndexDataset):
            self._dataset.set_current_indices(indices)
        return data


class NeedIndexDataset(Dataset):
    def __init__(self , dataset):
        self.dataset = dataset    
        self.value = torch.zeros(len(self.dataset),dtype=torch.float32)
        self.weight = torch.ones(len(self.dataset),dtype=torch.float32)
        self.cur_batch_index = None
        _BaseDataLoaderIter.__next__ = NeedIndex_hack_indices # 在类内重载dataloader的__next__方法
    
    
    def set_current_indices(self, cur_batch_indices: torch.Tensor):
        self.cur_batch_index = cur_batch_indices
    
    def get_value(self,index : np.array):
        return self.value[index].numpy()
    
    def get_min_max_value(self,index:np.array):
        _ = self.value[index].numpy()
        return sum(_),np.min(_),_[np.abs(_ - np.percentile(_,80)).argmin()]


    def update(self , values , loss_ = True):
        assert len(self.cur_batch_index) == batch_size and isinstance(values, torch.Tensor)
        batch_size = values.shape[0]
        weight = self.weight[self.cur_batch_index].to("cuda:0")
        value_val = values.detach().clone()
        self.value[self.cur_batch_index] = value_val.cpu()
        if loss_:
            values.mul_(weight)
            return values.mean()
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return index, self.dataset[index]  

    @property
    def sampler(self):
        return NeedIndexSampler(self , np.arange(len(self.dataset))) 

    def reset_weight(self,index,value):
        self.weight[index] = value


class NeedIndexSampler(Sampler):
    def __init__(self, dataset : NeedIndexDataset , sample_indices : np.array):
        super().__init__()
        self.dataset = dataset
        assert isinstance(sample_indices, torch.Tensor) == False , "the type of sample_indices cannot be torch.Tensor"
        self.sample_indices = sample_indices # 存储当前样本的索引，值为索引
        self.iter_obj = None # 用于迭代样本索引的迭代器
        self.reset()  # 在初始化时重置采样器的状态
    
    def __getitem__(self, idx):
        return self.sample_indices[idx]

    def reset(self):
        np.random.shuffle(self.sample_indices) # 将索引打散，返回一个全新的数组
        self.iter_obj = iter(self.sample_indices)  # 将 sample_indices 转换为迭代器 iter_obj。

    def set_index(self , sample_indices):
        assert isinstance(sample_indices, torch.Tensor) == False , "the type of sample_indices cannot be torch.Tensor"
        self.sample_indices = copy.deepcopy(sample_indices)
        self.reset()

    def __next__(self):
        return next(self.iter_obj)
    
    def __len__(self):
        return len(self.sample_indices)
    
    def __iter__(self):
        self.reset()
        return self