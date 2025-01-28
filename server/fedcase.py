import pickle
import sys
import json
from tqdm import tqdm
import random
from pathlib import Path
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List
import numpy as np
import wandb
import torch
import yaml
from rich.console import Console
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from collections import Counter
import queue
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
from client.fedcase import FedCaSeTrainer, FedCaSeClient
from server.fedavg import FedAvgServer
from utls.models import MODEL_DICT
from data.utils.datasets import DATA_NUM_CLASSES_DICT, DATASETS
from utls.dataset import NeedIndexDataset

class FedCaseDataset(NeedIndexDataset):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.frequency = torch.zeros(len(self.dataset),dtype=torch.int32)
        self.experience = torch.zeros(len(self.dataset),dtype=torch.float32)
    
    def update(self,values):
        assert isinstance(values, torch.Tensor)
        batch_size = values.shape[0]
        assert len(self.cur_batch_index) == batch_size, 'not enough index'
        value_val = values.detach().clone()
        self.value[self.cur_batch_index.long()] = value_val.cpu()
        self.frequency[self.cur_batch_index.long()] += 1 # 频率添加一次
        return values.mean()
    
    def get_value(self,index : np.array):
        M_u_ = torch.max(self.value[index]) - torch.min(self.value[index])
        M_u_ = M_u_ if M_u_ !=0 else 1
        T_u_ = torch.max(self.frequency[index]) - torch.min(self.frequency[index])
        T_u_ = T_u_ if T_u_ !=0 else 1
        M_u = (self.value[index] - torch.min(self.value[index])) / M_u_
        T_u =(self.frequency[index] - torch.min(self.frequency[index])) / T_u_
        self.experience[index] = M_u + T_u
        return self.experience[index]


class FedCaSeServer(FedAvgServer):
    def __init__(self, args = None, trainer_type=FedCaSeTrainer, client_type=FedCaSeClient):
        super().__init__(args, trainer_type, client_type)

        # 初始化 R_S
        for client_instance in self.client_instances:
            client_instance.R_S = client_instance.R_S * self.args['R_S']

if __name__ == "__main__":
    parser = get_argparser().parse_args()
    with open(parser.config_path, 'r') as file:
        args = yaml.safe_load(file)
    if args["set_seed"]:
        fix_random_seed(args["seed"])
    server = FedCaSeServer(args=args)
    server.train()