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
from client.fedavg import BaseClient , FedAvgTrainer
compute_heterogeneity = [1.0, 1.46, 1.89, 2.0]
probability = [0.45, 0.25, 0.2, 0.1]

with open(PROJECT_DIR / "utls" / "network_distribution.json", 'r') as f:
    network_distribution = json.load(f)

class fedbalancerClient(BaseClient):
    def __init__(self, client_id, train_index, batch_size):
        super().__init__(client_id, train_index, batch_size)
        self.batch_training_time = 0.0 # 单位为s
        self.train_set_index = np.array(self.train_set_index)
        self.len_OT = self.train_set_len
        self.selected_data_index = None
        self.metadata = {}



class fedbalancerTrainer(FedAvgTrainer):
    def __init__(self, device, model, trainloader, testloader, args):
        super().__init__(device, model, trainloader, testloader, args)
        self.p = 1
        # 修改评判器
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none').to(self.device)
    
    def load_dataset(self):
        S = int(max(self.synchronization['deadline'] / self.local_epoch / self.current_client.batch_training_time,1.0) * self.current_client.batch_size)
        if self.current_client.participation_times == 0 or S >= self.current_client.train_set_len:
            self.trainloader.sampler.set_index(self.current_client.train_set_index)  # 在里面实现了深拷贝
            self.trainloader.batch_sampler.batch_size = self.current_client.batch_size
            self.current_client.len_OT = self.current_client.train_set_len
            self.current_client.selected_data_index = self.current_client.train_set_index
        else:
            under_threshold = []
            over_threshold = []
            loss_value : np.array = self.trainloader.dataset.get_value(self.current_client.train_set_index)
            under_threshold = self.current_client.train_set_index[loss_value < self.synchronization['loss_threshold']]
            over_threshold = self.current_client.train_set_index[loss_value >= self.synchronization['loss_threshold']]
            L = max(S,len(over_threshold))
            D_ = np.random.choice(over_threshold, size = min(int(L * self.p) , len(over_threshold)), replace=False) 
            D__ = np.random.choice(under_threshold , size = min(max(0 , L - len(D_)),len(under_threshold)) , replace = False)
            selected_index=np.concatenate((D_ , D__))
            self.trainloader.sampler.set_index(selected_index)  # 在里面实现了深拷贝
            self.trainloader.batch_sampler.batch_size = self.current_client.batch_size
            self.current_client.selected_data_index = selected_index
            self.current_client.len_OT = len(over_threshold)



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
        loss_sum , min_loss , max_loss_80 = self.trainloader.dataset.get_min_max_value(self.current_client.selected_data_index) # 注意，是selected_data_index
        self.current_client.metadata['lsum'] = loss_sum
        self.current_client.metadata['llow'] = min_loss
        self.current_client.metadata['lhigh'] = max_loss_80
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
                loss = self.trainloader.dataset.update(loss)
                loss.backward()
                self.optimizer.step()
        torch.cuda.synchronize()