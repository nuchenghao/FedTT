import json
import torch
import math
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

        self.selected_samples_index = None


class FedCaSeTrainer(FedAvgTrainer):
    def __init__(self, device, model, trainloader, testloader, args):
        super().__init__(device, model, trainloader, testloader, args)
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none').to(self.device)

    def client_data_sampling(self):
        experience:torch.tensor = self.trainloader.dataset.get_value(self.current_client.train_set_index)
        index_sorted_experience = torch.argsort(experience)
        M_S_index = index_sorted_experience[-self.current_client.num_cached:]
        r = max(math.ceil(self.synchronization['alpha'] * (self.current_client.R_S / len(M_S_index))),1)
        rep_samples = M_S_index.repeat(r)
        len_flash = self.current_client.R_S - len(rep_samples)
        flash_index = index_sorted_experience[-(self.current_client.num_cached+len_flash):-self.current_client.num_cached]
        self.current_client.selected_samples_index = self.current_client.train_set_index[torch.cat((rep_samples,flash_index),dim=0).numpy()]
        self.current_client.train_set_len = len(self.current_client.selected_samples_index) # 更新，后续用于聚合时的参数

    def load_dataset(self):
        if self.current_client.participation_times > 2:
            self.client_data_sampling()
            self.trainloader.sampler.set_index(self.current_client.selected_samples_index)  # 在里面实现了深拷贝
        else:
            self.trainloader.sampler.set_index(self.current_client.train_set_index)
        self.trainloader.batch_sampler.batch_size = self.current_client.batch_size

    def start(self,
              client,
              optimizer_state_dict: OrderedDict[str, torch.Tensor],
              trainer_synchronization
              ):
        self.timer.start()
        self.current_client = client
        self.set_parameters(optimizer_state_dict, trainer_synchronization)  # 设置参数
        self.load_dataset()
        
        if self.args['client_eval']:
            self.current_client.pretrained_accuracy = evaluate(self.device, self.model, self.testloader)
        else:
            self.current_client.pretrained_accuracy = 0.0

        self.local_train() # 本地训练

        if self.args['client_eval']:
            self.current_client.accuracy = evaluate(self.device, self.model, self.testloader)
        else:
            self.current_client.accuracy = 0.0
        self.current_client.model_dict = deepcopy(self.model.state_dict())  # 一定要深拷贝
        self.timer.stop() # 里面的一些操作带来的开销就权当是网络传输的时间了
        self.current_client.training_time = self.timer.times[-1]
        self.current_client.participate_once()
        self.current_client.training_time_record[self.synchronization['round']] = round(self.current_client.training_time * 10.0) # 记录时间
        torch.cuda.empty_cache() # 释放缓存 
        return self.current_client
    
    def full_set(self):
        self.model.train()
        for _ in range(self.local_epoch):
            for inputs, targets in self.trainloader:
                if isinstance(inputs,torch.Tensor):
                    inputs = inputs.to(self.device, non_blocking=True)
                else:
                    inputs = [tensor.to(self.device, non_blocking=True) for tensor in inputs]
                targets = targets.to(self.device,non_blocking=True)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss = self.trainloader.dataset.update(loss,loss,self.device)
                loss.backward()
                self.optimizer.step()
        torch.cuda.synchronize()