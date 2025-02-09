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

compute_heterogeneity = [1.0, 1.46, 1.89, 2.0]
probability = [0.45, 0.25, 0.2, 0.1]

with open(PROJECT_DIR / "utls" / "network_distribution.json", 'r') as f:
    network_distribution = json.load(f)


class BaseClient:
    def __init__(self, client_id, train_index, batch_size):
        self.client_id = client_id
        self.train_set_index = train_index
        self.train_set_len = len(train_index)
        # self.compute_ability = np.random.choice(compute_heterogeneity, p=probability)
        self.communicate_ability = random.sample(network_distribution, 1)[0]
        self.download_speed_u = self.communicate_ability['down_u']
        self.download_speed_sigma = self.communicate_ability['down_sigma']
        self.upload_speed_u = self.communicate_ability["up_u"]
        self.upload_speed_sigma = self.communicate_ability["up_sigma"]
        self.participation_times = 0

        self.batch_size = batch_size

        self.model_dict = None  # 存当前的全局模型状态
        self.training_time = 0
        self.pretrained_accuracy = 0
        self.accuracy = 0
        self.loss = 0.0
        self.grad = None #存梯度值
        self.buffer = None # 存persistent_buffers

        self.training_time_record = {}

    def participate_once(self):
        self.participation_times += 1


class FedAvgTrainer:
    def __init__(
            self,
            device,
            model,
            trainloader,
            testloader,
            args: dict,
    ):
        self.args = args
        self.device = device
        self.model = model.to(self.device)
        self.model_size = sum(p.numel() * p.element_size() for p in self.model.parameters())  # 字节数量

        self.current_client = None

        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epoch = self.args["local_epoch"]
        # TODO:---------------- 实现自己的方法时，这里需要加上reduction='none'-----------------------
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1).to(self.device)
        self._criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none').to(self.device)
        self.optimizer = torch.optim.SGD(
            params=self.model.parameters(),
            lr=self.args["lr"],
            momentum=self.args["momentum"],
            weight_decay=self.args["weight_decay"],
        )
        # TODO -------------------其他参数----------------------------
        self.timer = Timer()  # 训练计时器
        self.synchronization = {}

    def get_upload_time(self):
        model_size = self.model_size / 1024  # change to KB because speed data use 'KB/s'
        upload_speed = np.random.normal(self.current_client.upload_speed_u, self.current_client.upload_speed_sigma)
        while upload_speed < 0:
            upload_speed = np.random.normal(self.current_client.upload_speed_u, self.current_client.upload_speed_sigma)
        upload_time = model_size / upload_speed
        return float(upload_time)

    def get_download_time(self):
        model_size = self.model_size / 1024  # change to KB because speed data use 'KB/s'
        download_speed = np.random.normal(self.current_client.download_speed_u,
                                          self.current_client.download_speed_sigma)
        while download_speed < 0:
            download_speed = np.random.normal(self.current_client.download_speed_u,
                                              self.current_client.download_speed_sigma)
        download_time = model_size / download_speed
        return float(download_time)

    def load_dataset(self):
        self.trainloader.sampler.set_index(self.current_client.train_set_index)  # 在里面实现了深拷贝
        self.trainloader.batch_sampler.batch_size = self.current_client.batch_size

    def set_parameters(self, optimizer_state_dict, trainer_synchronization):
        self.optimizer.load_state_dict(optimizer_state_dict)  # 加载全局优化器
        self.model.load_state_dict(self.current_client.model_dict)
        self.synchronization = trainer_synchronization

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
                loss = self._criterion(outputs, targets).mean() # 两者会有稍许误差
                # loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
        torch.cuda.synchronize()

    def local_train(self):
        """
        本地训练函数，后面实现的算法要重写这个函数.
        可以确定的是：在训练前，模型已经在设备上了
        """
        self.full_set()
