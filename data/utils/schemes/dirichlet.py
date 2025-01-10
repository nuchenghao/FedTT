from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import Dataset


def dirichlet(
        dataset: Dataset, client_num: int, alpha: float, least_samples: int
) -> Tuple[List[List[int]], Dict]:
    """
    参数：
        dataset: 一个 PyTorch 的 Dataset 对象，包含数据和目标标签。
        client_num: 要划分的数据集客户端数量。
        alpha: Dirichlet 分布的浓度参数。较高的值会导致更均匀的分布。
        least_samples: 每个客户端应至少拥有的样本数量。
    返回值：一个元组，包含：
        每个客户端的数据索引列表。
        一个包含样本分布信息的统计字典。
    """
    label_num = len(dataset.classes)  # 数据集中类别的数量
    min_size = 0  # 跟踪分配给任何客户端的最小样本数量。
    stats = {}  # 用于存储每个客户端数据分布统计信息的字典。
    partition = {"separation": None, "data_indices": None}  # 用于存储数据分割信息的字典

    targets_numpy = np.array(dataset.targets, dtype=np.int32)  # 将目标标签转为 NumPy 数组。
    data_idx_for_each_label = [
        np.where(targets_numpy == i)[0] for i in range(label_num)
    ]  # 为每个标签创建一个列表，包含该标签对应的数据索引。

    while min_size < least_samples:  # 分配样本直到满足最小样本数量
        data_indices = [[] for _ in range(client_num)]
        for k in range(label_num):
            np.random.shuffle(data_idx_for_each_label[k])  # 对每个标签，随机打乱索引，并根据 Dirichlet 分布生成样本分配。
            # 生成一个长度为 client_num 的数组，其中每个元素都是基于 Dirichlet 分布的随机值，这些值的和为 1。
            # 这些随机值表示在 client_num 个客户端之间的样本分配比例，浓度参数为 alpha，可以控制分布的均匀性：
            # 较小的 alpha 值会导致更不均匀的分布（某些客户端可能会得到更多样本）
            # 较大的 alpha 值会导致更均匀的分布（样本在客户端之间更平衡）
            distrib = np.random.dirichlet(np.repeat(alpha, client_num))
            # # distrib = np.array(
            # #     [
            # #         p * (len(idx_j) < len(targets_numpy) / client_num)
            # #         for p, idx_j in zip(distrib, data_indices)
            # #     ]
            # # )
            # distrib = distrib / distrib.sum()
            # 按照比例将该类别的样本进行索引划分(要去掉最后一个分割点
            distrib = (np.cumsum(distrib) * len(data_idx_for_each_label[k])).astype(int)[:-1]
            # 保存每个客户的索引点
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
    # stats["sample per client"] = {
    #     "std": num_samples.mean(),
    #     "stddev": num_samples.std(),
    # }

    partition["data_indices"] = data_indices

    return partition, stats
