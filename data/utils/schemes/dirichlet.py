from collections import Counter
from typing import Dict, List, Tuple
import numpy as np
from torch.utils.data import Dataset

def dirichlet(
        dataset: Dataset, client_num: int, alpha: float, least_samples: int
) -> Tuple[List[List[int]], Dict]:
    label_num = len(dataset.classes)
    min_size = 0
    stats = {}
    partition = {"separation": None, "data_indices": None}
    targets_numpy = np.array(dataset.targets, dtype=np.int32)
    data_idx_for_each_label = [
        np.where(targets_numpy == i)[0] for i in range(label_num)
    ]
    while min_size < least_samples:
        data_indices = [[] for _ in range(client_num)]
        for k in range(label_num):
            np.random.shuffle(data_idx_for_each_label[k])
            distrib = np.random.dirichlet(np.repeat(alpha, client_num))
            distrib = (np.cumsum(distrib) * len(data_idx_for_each_label[k])).astype(int)[:-1]
            data_indices = [
                np.concatenate((idx_j, idx.tolist())).astype(np.int64)
                for idx_j, idx in zip(
                    data_indices, np.split(data_idx_for_each_label[k], distrib)
                )
            ]
            min_size = min([len(idx_j) for idx_j in data_indices])
    for i in range(client_num):
        stats[i] = {"total": None, "distribution": None}
        stats[i]["total"] = len(targets_numpy[data_indices[i]])
        stats[i]["distribution"] = Counter(targets_numpy[data_indices[i]].tolist())
    num_samples = np.array(list(map(lambda stat_i: stat_i["total"], stats.values())))
    print(f"mean : {num_samples.mean()}")
    print(f"std: {num_samples.std()}")
    partition["data_indices"] = data_indices
    return partition, stats
