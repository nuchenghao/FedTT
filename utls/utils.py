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
class Timer:
    def __init__(self):
        self.times = []
    def start(self):
        self.tik = time.time()
    def stop(self):
        self.times.append(time.time() - self.tik)
    def sum(self):
        return sum(self.times)
def fix_random_seed(seed: int) -> None:
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
        src,
        detach=False,
        requires_name=False,
) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[str]]]:
    func = (lambda x: x.detach().clone()) if detach else (lambda x: x)
    parameters = []
    keys = []
    if isinstance(src, OrderedDict):
        for name, param in src.items():
            if param.requires_grad:
                parameters.append(func(param))
                keys.append(name)
    elif isinstance(src, torch.nn.Module):
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
        src, detach=True
) -> torch.Tensor:
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
    torch.cuda.empty_cache()
    return accuracy , total_loss
def count_labels(
        dataset: BaseDataset, indices: List[int] = None, min_value=0
) -> List[int]:
    if indices is None:
        indices = list(range(len(dataset.targets)))
    counter = Counter(dataset.targets[indices].tolist())
    return [counter.get(i, min_value) for i in range(len(dataset.classes))]
class Logger:
    def __init__(
            self, stdout: Console, enable_log: bool, logfile_path: Union[Path, str]
    ):
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
