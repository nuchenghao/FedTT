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
import copy
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
from client.fedavg import  BaseClient
from utls.models import MODEL_DICT
from data.utils.datasets import DATA_NUM_CLASSES_DICT, DATASETS , DATASETS_COLLATE_FN
from utls.dataset import CustomSampler
from server.fedavg import FedAvgServer
from client.fednova import FedNovaTrainer,FedNovaClient


class FedNovaServer(FedAvgServer):
    def __init__(self, args = None, trainer_type=FedNovaTrainer, client_type=FedNovaClient):
        super().__init__(args, trainer_type, client_type)
        self.client_coeff = {}
        self.client_norm_grad = {}


    def aggregate(
            self,
            client_model_cache,
            weight_cache,
    ):
        model_state = self.model.state_dict()
        nova_model_state = copy.deepcopy(model_state)
        coeff = 0.0
        for i,client_id in enumerate(self.current_selected_client_ids):
            client_instance = self.client_instances[client_id]
            coeff += client_instance.coeff * client_instance.train_set_len / sum(weight_cache)
            val = torch.tensor(client_instance.train_set_len / sum(weight_cache),dtype=torch.float,device=self.device)
            for key in client_instance.norm_grad:
                if i == 0:
                    nova_model_state[key] = client_instance.norm_grad[key] * val
                else:
                    nova_model_state[key] = nova_model_state[key] + client_instance.norm_grad[key] * val
        for key in model_state:
            model_state[key] = model_state[key] - coeff * nova_model_state[key]
        self.model.load_state_dict(model_state)








if __name__ == "__main__":
    parser = get_argparser().parse_args()
    with open(parser.config_path, 'r') as file:
        args = yaml.safe_load(file)
    if args["set_seed"]:
        fix_random_seed(args["seed"])
    server = FedNovaServer(args=args)
    server.train()
