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
class ODEClient(BaseClient):
    def __init__(self, client_id, train_index, batch_size):
        super().__init__(client_id, train_index, batch_size)
        self.train_set_label = None
        self.label_num_distribution = {}
        self.buffer = {}
        self.buffer_idx = []
        self.buffer_size = 100
        self.new_num_train_samples = 0.0
        self.label_weight = {}
        self.weights_tensor = None
        self.new_weight4aggregation = 0.0
class ODETrainer(FedAvgTrainer):
    def __init__(self, device, model, trainloader, testloader, args):
        super().__init__(device, model, trainloader, testloader, args)
        self.criterion_ = torch.nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none').to(self.device)
    def get_client_grad(self , client_instance , factor):
        self.current_client = client_instance
        self.model.load_state_dict(self.current_client.model_dict)
        self.trainloader.sampler.set_index(self.current_client.train_set_index)
        self.trainloader.batch_sampler.batch_size = self.current_client.batch_size if self.args['model'] != "vit" else int(self.current_client.batch_size // 2)
        self.model.train()
        __NormBase = torch.nn.BatchNorm2d.__mro__[2]
        for module in self.model.modules():
            if isinstance(module,__NormBase):
                module.track_running_stats = False
        if self.args['model'] != "vit":
            trainable_parameters = [p for p in self.model.parameters()]
        else:
            trainable_parameters = []
            for name, param in self.model.named_parameters():
                if 'head' in name or 'lora' in name or 'Prompt' in name:
                    trainable_parameters.append(param)
        cnt = 0
        gradient = []
        self.optimizer.zero_grad()
        for inputs, targets in self.trainloader:
            if isinstance(inputs,torch.Tensor):
                inputs = inputs.to(self.device, non_blocking=True)
            else:
                inputs = [tensor.to(self.device, non_blocking=True) for tensor in inputs]
            targets = targets.to(self.device,non_blocking=True)
            outputs = self.model(inputs)
            loss = self.criterion_(outputs, targets).sum()
            if cnt == 0:
                gradient = torch.autograd.grad(loss,trainable_parameters)
            else:
                gradient =[torch.sum(torch.stack(grad,dim=-1),dim=-1) for grad in zip(gradient, torch.autograd.grad(loss,trainable_parameters))]
            cnt += 1
        __NormBase = torch.nn.BatchNorm2d.__mro__[2]
        for module in self.model.modules():
            if isinstance(module,__NormBase):
                module.track_running_stats = True
        return [factor * _grad for _grad in gradient]
    def loss_fn(self,params, data, target):
        output = torch.func.functional_call(self.model, params, (data.unsqueeze(0),))
        loss = self.criterion(output, target.unsqueeze(0))
        return loss
    def update_client_buffer(self,client_instance,global_gradient,batch_size):
        self.current_client = client_instance
        self.model.load_state_dict(self.current_client.model_dict)
        self.model.train()
        self.trainloader.sampler.set_index(self.current_client.train_set_index)
        self.trainloader.batch_sampler.batch_size = batch_size
        __NormBase = torch.nn.BatchNorm2d.__mro__[2]
        for module in self.model.modules():
            if isinstance(module,__NormBase):
                module.track_running_stats = False
        params = dict(self.model.named_parameters())
        if self.args['model'] != "vit":
            trainable_parameters = [p for p in self.model.parameters()]
        else:
            trainable_parameters = []
            for name, param in self.model.named_parameters():
                if 'head' in name or 'lora' in name or 'Prompt' in name:
                    trainable_parameters.append(param)
        for inputs,targets in self.trainloader:
            result=torch.zeros(len(targets),dtype=torch.float32,device=self.device)
            if isinstance(inputs,torch.Tensor) and self.args['model'] != "vit":
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device,non_blocking=True)
                gradients: dict[str:torch.tensor] = vmap(grad(self.loss_fn), in_dims=(None, 0, 0),randomness='same')(params, inputs, targets)
                for local_grad,global_grad in zip(gradients.values(),global_gradient):
                    result += torch.sum(local_grad * global_grad,dim=tuple(range(1,local_grad.ndim)))
            else:
                self.optimizer.zero_grad()
                if isinstance(inputs,torch.Tensor):
                    inputs = inputs.to(self.device, non_blocking=True)
                else:
                    inputs = [tensor.to(self.device, non_blocking=True) for tensor in inputs]
                targets = targets.to(self.device,non_blocking=True)
                outputs = self.model(inputs)
                loss = self.criterion_(outputs, targets).mean()
                _grad = torch.autograd.grad(loss,trainable_parameters)
                self.optimizer.zero_grad()
                for local_grad,global_grad in zip(_grad,global_gradient):
                    result += torch.sum(local_grad * global_grad)
            self.trainloader.dataset.update(result,result,self.device)
        del trainable_parameters
        value:np.array = self.trainloader.dataset.get_value(self.current_client.train_set_index)
        self.current_client.buffer_idx = []
        for label in self.current_client.label_weight.keys():
            indices=np.where(self.current_client.train_set_label == label)[0]
            value_at_indices = value[indices]
            sorted_indices=indices[np.argsort(value_at_indices)]
            self.current_client.buffer_idx.extend(self.current_client.train_set_index[sorted_indices[-self.current_client.buffer_size[label]:]].tolist())
        for module in self.model.modules():
            if isinstance(module,__NormBase):
                module.track_running_stats = True
        self.current_client.train_set_len = len(self.current_client.buffer_idx)
    def load_dataset(self):
        self.trainloader.sampler.set_index(self.current_client.buffer_idx)
        self.trainloader.batch_sampler.batch_size = self.current_client.batch_size
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
            self.current_client.pretrained_accuracy = evaluate(self.device, self.model, self.testloader)
        else:
            self.current_client.pretrained_accuracy = 0.0
        self.local_train() 
        if self.args['client_eval']:
            self.current_client.accuracy = evaluate(self.device, self.model, self.testloader)
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
                loss = self.criterion_(outputs, targets)
                weights = self.current_client.weights_tensor[targets]
                loss = (loss * weights / torch.sum(weights)).sum()
                loss.backward()
                self.optimizer.step()
        torch.cuda.synchronize()
