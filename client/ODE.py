import json
import queue
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
        self.train_set_index = np.array(self.train_set_index)
        self.train_set_label = None

        self.label_distribution = {}
        self.buffer = {} # 拥有的每个标签对应一个优先队列
        self.buffer_idx = [] # 缓存的训练标签数量
        self.buffer_size = 100 # 后面会变成一个dict：每个标签缓存的数量
        self.new_num_train_samples = 0.0
        self.distributed_labels = {} #dict{label(int):float}



class ODETrainer(FedAvgTrainer):
    def __init__(self, device, model, trainloader, testloader, args):
        super().__init__(device, model, trainloader, testloader, args)
        # 修改评判器
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none').to(self.device)

    def get_client_grad(self , client_instance):
        self.current_client = client_instance
        self.model.load_state_dict(self.current_client.model_dict)
        self.model = self.model.to(self.device)
        self.trainloader.sampler.set_index(self.current_client.train_set_index)  # 在里面实现了深拷贝
        self.trainloader.batch_sampler.batch_size = self.current_client.batch_size

        self.model.train()
        trainable_parameters = [p for p in self.model.parameters()]# 获得可学习的参数，构成一个list 
        gradient_list= [] # list[(torch.tensor, ... ,torch.tensor)]
        self.optimizer.zero_grad() # 优化器并没有使用任何的状态存储信息，因此可以复用
        for inputs, targets in self.trainloader:
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device,non_blocking=True)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets).sum() # 求和并累计梯度
            # torch.autograd.grad() 是 PyTorch 中用于计算梯度的函数，它允许用户手动计算某个标量张量相对于其他张量的梯度。
            # 与 backward() 不同，grad() 不会累积梯度到 .grad 属性中，而是直接返回梯度值。
            # 返回一个元组，包含 inputs 中每个张量的梯度。
            # 如果 inputs 是单个张量，返回值仍为元组，需通过索引 [0] 获取梯度。
            gradient_list.append(torch.autograd.grad(loss,trainable_parameters))
        sum_gradient = [torch.sum(torch.stack(grad,dim=-1),dim=-1) for grad in zip(*gradient_list)]
        
        return sum_gradient

     
    def update_client_buffer(self,client_instance):
        for label in client_instance.distributed_labels.keys():
            _buffer=queue.PriorityQueue()
            while client_instance.buffer[label].qsize() != 0:
                (value,index,index_in_dataset) = client_instance.buffer[label].get()
                grad 

