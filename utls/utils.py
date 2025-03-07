import os
import random
from copy import deepcopy
from collections import Counter, OrderedDict
from typing import List, Optional, Tuple, Union
from collections import Counter
from argparse import ArgumentParser, Namespace
import torch
import numpy as np
from torch.utils.data import DataLoader
import time
from pathlib import Path
from typing import Union
from rich.console import Console
from torcheval.metrics.functional import multiclass_f1_score

from data.utils.datasets import BaseDataset
from utls.language import process_x

PROJECT_DIR = Path(__file__).parent.parent.absolute()
TRAIN_LOG = PROJECT_DIR / "trainlog"
TEMP_DIR = PROJECT_DIR / "temp"


class Timer:  # @save
    """记录多次运行时间"""

    def __init__(self):
        self.times = []

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)


def fix_random_seed(seed: int) -> None:
    """Fix the random seed of FL training.

    Args:
        seed (int): Any number you like as the random seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def NN_state_load(
        src: Union[OrderedDict[str, torch.Tensor], torch.nn.Module],
        detach=False,
        requires_name=False,
) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[str]]]:
    """Collect all parameters in `src` that `.requires_grad = True` into a list and return it.
        这个函数的作用相当于 .parameters()： trainable_params(model) == list(model.parameters()) : True
    Args:
        src (Union[OrderedDict[str, torch.Tensor], torch.nn.Module]): 可以是一个有序字典（OrderedDict）或一个 PyTorch 模型（torch.nn.Module），包含要提取的参数。
        requires_name (bool, optional): 布尔值，决定是否返回参数名称，默认为 False。
        detach (bool, optional): If set to `True`, the list would contain `param.detach().clone()` rather than `param`. Defaults to False.布尔值，决定是否返回参数的克隆版本（detach()），默认为 False。

    Returns:
        如果 requires_name 为 True，返回一个元组，包含参数列表和参数名称列表；否则，只返回参数列表。
        Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[str]]]: List of parameters, [List of names of parameters].
    """
    func = (lambda x: x.detach().clone()) if detach else (lambda x: x)
    parameters = []
    keys = []
    if isinstance(src, OrderedDict):
        for name, param in src.items():
            if param.requires_grad:
                parameters.append(func(param))
                keys.append(name)
    elif isinstance(src, torch.nn.Module):
        # state_dict() 返回一个字典，其中包含了模型的所有可学习参数（如权重和偏置）以及其他状态信息（如 BatchNorm 的均值和方差等）。
        # keep_vars=True: 当设置为 True 时，返回的状态字典中的参数将保持为原始的 torch.Tensor 对象，而不是将其从计算图中断开。这意味着返回的字典中的每个张量仍然与模型的计算图相连，可以用于后续的计算。
        for name, param in src.state_dict(keep_vars=True).items():
            if param.requires_grad:
                parameters.append(func(param))
                keys.append(name)
            if param.persistent:
                parameters.append(func(param))
                keys.append(name)
    if requires_name:
        return parameters, keys
    else:
        return parameters


def vectorize(
        src: Union[OrderedDict[str, torch.Tensor], List[torch.Tensor]], detach=True
) -> torch.Tensor:
    """Vectorize and concatenate all tensors in `src`.

    Args:
        src (Union[OrderedDict[str, torch.Tensor]List[torch.Tensor]]): The source of tensors.
        detach (bool, optional): Set to `True`, return the `.detach().clone()`. Defaults to True.

    Returns:
        torch.Tensor: The vectorized tensor.
    """
    func = (lambda x: x.detach().clone()) if detach else (lambda x: x)
    if isinstance(src, list):
        return torch.cat([func(param).flatten() for param in src])
    elif isinstance(src, OrderedDict):
        return torch.cat([func(param).flatten() for param in src.values()])


@torch.no_grad()
def evaluate(
        device: torch.device,
        model: torch.nn.Module,
        dataloader: DataLoader,
) -> Tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    total_loss = 0.0
    for inputs, targets in dataloader:
        if isinstance(inputs,torch.Tensor):
            inputs = inputs.to(device, non_blocking=True)
        else:
            inputs = [tensor.to(device, non_blocking=True) for tensor in inputs]
        targets = targets.to(device,non_blocking=True)
        outputs = model(inputs)
        total_loss += criterion(outputs, targets).sum().item()
        pred = torch.argmax(outputs, -1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)
    accuracy = 100. * correct / total
    return accuracy , total_loss


def count_labels(
        dataset: BaseDataset, indices: List[int] = None, min_value=0
) -> List[int]:
    """For counting number of labels in `dataset.targets`.

    Args:
        dataset (BaseDataset): Target dataset.
        indices (List[int]): the subset indices. Defaults to all indices of `dataset` if not specified.
        min_value (int, optional): The minimum value for each label. Defaults to 0.

    Returns:
        List[int]: The number of each label.
    """
    if indices is None:
        indices = list(range(len(dataset.targets)))
    counter = Counter(dataset.targets[indices].tolist())
    return [counter.get(i, min_value) for i in range(len(dataset.classes))]


class Logger:
    def __init__(
            self, stdout: Console, enable_log: bool, logfile_path: Union[Path, str]
    ):
        """This class is for solving the incompatibility between the progress bar and log function in library `rich`.

        Args:
            stdout (Console): The `rich.console.Console` for printing info onto stdout (终端中).
            enable_log (bool): Flag indicates whether log function is actived.
            logfile_path (Union[Path, str]): The path of log file.
        """
        self.stdout = stdout
        self.logfile_stream = None
        self.enable_log = enable_log
        if self.enable_log:
            self.logfile_stream = open(logfile_path, "w")
            self.logger = Console(
                file=self.logfile_stream, record=True, log_path=False, log_time=True
            )

    def log(self, *args, **kwargs):
        self.stdout.log(*args, **kwargs)
        if self.enable_log:
            self.logger.log(*args, **kwargs)

    def close(self):
        if self.logfile_stream:
            self.logfile_stream.close()


def get_argparser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./config/fedavg.yaml")
    parser.add_argument("--name", type=str,default="")
    return parser
