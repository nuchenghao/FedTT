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
from utls.utils import  evaluate
from utls.utils import Timer

class BaseClient:
    def __init__(self, client_id, train_index, batch_size):
        self.client_id = client_id
        self.train_set_index = np.array(train_index)
        self.train_set_len = len(train_index)
        self.participation_times = 0
        self.batch_size = batch_size
        self.model_dict = None
        self.training_time = 0
        self.pretrained_accuracy = 0
        self.accuracy = 0
        self.loss = 0.0
        self.grad = None
        self.buffer = None
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
        self.model_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        self.current_client = None
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epoch = self.args["local_epoch"]
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1).to(self.device)
        self._criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none').to(self.device)
        self.optimizer = torch.optim.SGD(
            params=self.model.parameters(),
            lr=self.args["lr"],
            momentum=self.args["momentum"],
            weight_decay=self.args["weight_decay"],
        )
        self.timer = Timer()  
        self.synchronization = {}

    def load_dataset(self):
        self.trainloader.sampler.set_index(self.current_client.train_set_index)
        self.trainloader.batch_sampler.batch_size = self.current_client.batch_size

    def set_parameters(self, optimizer_state_dict, trainer_synchronization):
        self.optimizer.load_state_dict(optimizer_state_dict)
        self.model.load_state_dict(self.current_client.model_dict)
        self.synchronization = trainer_synchronization

    def start(self,
              client,
              optimizer_state_dict: OrderedDict[str, torch.Tensor],
              trainer_synchronization
              ):
        self.timer.start()
        self.current_client = client
        self.set_parameters(optimizer_state_dict, trainer_synchronization)
        self.load_dataset()
        if self.args['client_eval']:
            self.current_client.pretrained_accuracy = evaluate(self.device, self.model, self.testloader)[0]
        else:
            self.current_client.pretrained_accuracy = 0.0
        self.local_train()
        if self.args['client_eval']:
            self.current_client.accuracy = evaluate(self.device, self.model, self.testloader)[0]
        else:
            self.current_client.accuracy = 0.0
        self.current_client.model_dict = deepcopy(self.model.state_dict())
        self.timer.stop()
        self.current_client.training_time = self.timer.times[-1]
        self.current_client.participate_once()
        self.current_client.training_time_record[self.synchronization['round']] = round(self.current_client.training_time * 10.0)
        torch.cuda.empty_cache()
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
                loss = self._criterion(outputs, targets).mean()
                loss.backward()
                self.optimizer.step()
        torch.cuda.synchronize()
        
    def local_train(self):
        self.full_set()
