import json
import pickle
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

class ODEClient(BaseClient):
    def __init__(self, client_id, train_index, batch_size):
        super().__init__(client_id, train_index, batch_size)
        self.label_distribution = {}
        self.buffer_size = 100
        self.distributed_labels = {}


class ODETrainer(FedAvgTrainer):
    def __init__(self, device, model, trainloader, testloader, args):
        super().__init__(device, model, trainloader, testloader, args)