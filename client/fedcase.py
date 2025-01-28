import json
import torch
import random
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Tuple, Union
from pathlib import Path
import torch
from sympy.stats.rv import probability
from torch.utils.data import DataLoader, Subset
import copy
from collections import Counter
from torch.func import grad, vmap

PROJECT_DIR = Path(__file__).parent.parent.absolute()
from utls.utils import NN_state_load, evaluate
from data.utils.datasets import DATASETS
from utls.utils import Timer
from collections import defaultdict
from client.fedavg import BaseClient,FedAvgTrainer
compute_heterogeneity = [1.0, 1.46, 1.89, 2.0]
probability = [0.45, 0.25, 0.2, 0.1]

with open(PROJECT_DIR / "utls" / "network_distribution.json", 'r') as f:
    network_distribution = json.load(f)

class FedCaSeClient(BaseClient):
    def __init__(self, client_id, train_index, batch_size):
        super().__init__(client_id, train_index, batch_size)

        self.R_S = self.train_set_len
        self.num_cached = int(self.train_set_len * 0.1)


class FedCaSeTrainer(FedAvgTrainer):
    def __init__(self, device, model, trainloader, testloader, args):
        super().__init__(device, model, trainloader, testloader, args)

    def client_data_sampling(self):
        experience:torch.tensor = self.trainloader.dataset.get_value(self.current_client.train_set_index)


    def start(self,
              client,
              optimizer_state_dict: OrderedDict[str, torch.Tensor],
              trainer_synchronization
              ):
        self.timer.start()
        self.current_client = client
        self.set_parameters(optimizer_state_dict, trainer_synchronization)  # 设置参数
        self.client_data_sampling()