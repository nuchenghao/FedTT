import pickle
import sys
import json
import os
import random
from pathlib import Path
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List
import wandb
import torch
import yaml
from rich.console import Console
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from collections import Counter
import queue
from tqdm import tqdm
PROJECT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(PROJECT_DIR.as_posix())
sys.path.append(PROJECT_DIR.joinpath("src").as_posix())

from utls.utils import (
    TRAIN_LOG,
    Logger,
    fix_random_seed,
    NN_state_load,
    get_argparser,
    evaluate
)
from utls.models import MODEL_DICT
from data.utils.datasets import DATA_NUM_CLASSES_DICT, DATASETS , DATASETS_COLLATE_FN
from utls.dataset import CustomSampler


class FedAvgServerOnDevice:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda'
        self.current_time = 0  # 全局时间
        

        self.client_sample_stream = [
            random.sample(
                self.train_client_ids, max(1, int(self.client_num * self.args["client_join_ratio"]))
            )
            for _ in range(self.args["global_epoch"])
        ]
        self.current_selected_client_ids: List[int] = []

        self.data_num_classes = DATA_NUM_CLASSES_DICT[self.args['dataset']]
        self.model = MODEL_DICT[self.args["model"]](self.data_num_classes).to(self.device)

        self.testset = DATASETS[self.args['dataset']](PROJECT_DIR / "data" / args["dataset"], "test")
        self.testloader = DataLoader(Subset(self.testset, list(range(len(self.testset)))), batch_size=self.args['t_batch_size'],
                                     shuffle=False, pin_memory=True, num_workers=4,collate_fn = DATASETS_COLLATE_FN[self.args['dataset']],
                                     persistent_workers=True, pin_memory_device='cuda:0',prefetch_factor = 8)

    def select_clients(self, global_epoch):
        self.current_selected_client_ids = self.client_sample_stream[global_epoch]


    def get_model_dict(self):
        return {key: value for key, value in self.model.state_dict().items()}