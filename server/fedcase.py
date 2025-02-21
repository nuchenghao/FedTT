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
from data.utils.datasets import DATASETS_COLLATE_FN
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
        assert len(self.experience[index]) == len(M_u + T_u)
        self.experience[index] = M_u + T_u
        return self.experience[index]


class FedCaSeServer(FedAvgServer):
    def __init__(self, args = None, trainer_type=FedCaSeTrainer, client_type=FedCaSeClient):
        super().__init__(args, trainer_type, client_type)

        # 初始化 
        for client_instance in self.client_instances:
            client_instance.R_S = int(client_instance.R_S * self.args['R_S'])
            client_instance.train_set_index = np.array(client_instance.train_set_index)

        self.trainset = FedCaseDataset(self.trainset)
        self.train_sampler = self.trainset.sampler
        self.trainloader = DataLoader(self.trainset, batch_size=self.args["batch_size"],shuffle = False,
                                      pin_memory=True, num_workers=4 , collate_fn = DATASETS_COLLATE_FN[self.args['dataset']] , persistent_workers=True,
                                      sampler=self.train_sampler, pin_memory_device='cuda:0',prefetch_factor = 8)
        self.cuda_0_trainer.trainloader = self.trainloader

    def train_one_round(self,global_round):
        client_model_cache = []  # 缓存梯度
        weight_cache = []  # 缓存梯度对应的权重
        client_training_time = []
        trainer_synchronization = {"round":global_round , "alpha":self.args["alpha"]}
        for client_id in self.current_selected_client_ids:
            assert self.client_instances[client_id].client_id == client_id
            self.client_instances[client_id].model_dict = deepcopy(self.model.state_dict())
        for client_id in self.current_selected_client_ids:
            modified_client_instance = self.cuda_0_trainer.start(
                self.client_instances[client_id],
                self.optimizer.state_dict(),
                trainer_synchronization
            )
            assert modified_client_instance.client_id == client_id
            self.logger.log(
                f"client {client_id} has finished and has participate {modified_client_instance.participation_times}. The local train set size is {modified_client_instance.train_set_len}. ",
                f"The pretrained acc is {modified_client_instance.pretrained_accuracy:.3f}%. The local accuracy is {modified_client_instance.accuracy:.3f}%.",
                f"The time is {modified_client_instance.training_time}. Scaled time is {round(modified_client_instance.training_time * 10.0)}.")
            self.client_to_server.put(modified_client_instance)
        assert self.client_to_server.qsize() == len(self.current_selected_client_ids)
        while not self.client_to_server.empty():
            modified_client_instance = self.client_to_server.get()
            assert modified_client_instance.client_id in self.current_selected_client_ids
            client_model = {key: value for key, value in modified_client_instance.model_dict.items()}
            client_model_cache.append(client_model)
            del modified_client_instance.model_dict
            weight_cache.append(modified_client_instance.train_set_len)
            client_training_time.append(round(modified_client_instance.training_time * 10.0))
            self.client_instances[modified_client_instance.client_id] = modified_client_instance  # 更新client信息

        # 聚合并更新参数
        self.aggregate(client_model_cache, weight_cache)  # 聚合梯度
        return max(client_training_time)

if __name__ == "__main__":
    parser = get_argparser().parse_args()
    with open(parser.config_path, 'r') as file:
        args = yaml.safe_load(file)
    if args["set_seed"]:
        fix_random_seed(args["seed"])
    server = FedCaSeServer(args=args)
    server.train()