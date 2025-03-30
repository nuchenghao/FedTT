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
import threading
from data.utils.datasets import DATASETS_SIZE
import queue

PROJECT_DIR = Path(__file__).parent.parent.absolute()
from utls.utils import NN_state_load, evaluate
from data.utils.datasets import DATASETS
from utls.utils import Timer
from collections import defaultdict
from client.fedavg import BaseClient,FedAvgTrainer

class FedProxTrainer(FedAvgTrainer):
    def __init__(self, device, model, trainloader, testloader, args):
        super().__init__(device, model, trainloader, testloader, args)
        self.global_model = None
        self.inference_net = deepcopy(model).to(self.device)
        self.train_stream = torch.cuda.Stream()
        self.inference_stream = torch.cuda.Stream()
        self.inference_event = torch.cuda.Event()
        self.train_event = torch.cuda.Event()
        self.inputs = [None, None]
        self.inputs_b = [torch.zeros((self.args['batch_size'],*DATASETS_SIZE[self.args['dataset']]),dtype=torch.float32,device=self.device), torch.zeros((self.args['batch_size'],*DATASETS_SIZE[self.args['dataset']]),dtype=torch.float32,device=self.device)]
        self.targets = [None, None]
        self.targets_b = [torch.zeros((self.args['batch_size'],),dtype=torch.int64,device=self.device), torch.zeros((self.args['batch_size'],),dtype=torch.int64,device=self.device)]
        self.weights = [None, None]
        self.inference_to_train = queue.Queue()
        self.barrier = threading.Barrier(2)
        self.finish_one_epoch = threading.Event()
        self.r = self.args['r']
        # 修改评判器
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none').to(self.device)


    
    def set_parameters(self, optimizer_state_dict, trainer_synchronization):
        self.optimizer.load_state_dict(optimizer_state_dict)  # 加载全局优化器
        self.model.load_state_dict(self.current_client.model_dict)
        self.synchronization = trainer_synchronization
        self.global_model = self.current_client.model_dict

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
                loss = self.criterion(outputs, targets).mean() # 两者会有稍许误差
                proximal_term = 0.0
                for name, param in self.model.named_parameters():
                    proximal_term += (param - self.global_model[name]).norm(2)
                loss += self.args['mu'] / 2 * proximal_term
                loss.backward()
                self.optimizer.step()
        torch.cuda.synchronize()
    
    def train(self):
        cnt = 1
        while True:
            iteration_number = self.inference_to_train.get()
            if iteration_number == 0:
                break
            else:
                with torch.cuda.stream(self.train_stream):
                    self.model.train()
                    for _ in range(iteration_number):
                        self.barrier.wait()
                        cnt ^= 1
                        self.inference_event.wait()
                        self.optimizer.zero_grad()
                        outputs = self.model(self.inputs[cnt])
                        loss = self.criterion(outputs, self.targets[cnt])
                        loss = (loss * self.weights[cnt]).mean()
                        proximal_term = 0.0
                        for name, param in self.model.named_parameters():
                            proximal_term += (param - self.global_model[name]).norm(2)
                        loss += self.args['mu'] / 2 * proximal_term
                        loss.backward()
                        self.optimizer.step()
                        self.train_event.record()
        torch.cuda.synchronize()
    
    def my(self):
        train_thread = threading.Thread(target=self.train, args=())
        train_thread.start()
        # gpu_utilization = []
        self.train_event.record()
        cnt = 0
        for epoch in range(self.local_epoch):
            # torch.cuda.reset_peak_memory_stats() # 重置显存峰值统计
            # gpu_utilization = []
            itertrainloader = iter(self.trainloader)  # 创建trainloader的迭代器
            self.inference_to_train.put(len(itertrainloader))  # 训练线程预加载,这里的值时batch_size会load的次数
            inputs_raw, targets_raw = next(itertrainloader)
            with torch.cuda.stream(self.inference_stream):
                if isinstance(inputs_raw,torch.Tensor):
                    self.inputs_b[cnt][:len(targets_raw), ...] = inputs_raw.to(self.device, non_blocking=True)
                    self.inputs[cnt] = self.inputs_b[cnt][:len(targets_raw)]
                else:
                    self.inputs[cnt] = [tensor.to(self.device, non_blocking=True) for tensor in inputs_raw]
                    self.inputs_b[cnt] = self.inputs[cnt]
                self.targets_b[cnt][:len(targets_raw), ...] = targets_raw.to(self.device,non_blocking=True)
                self.targets[cnt] = self.targets_b[cnt][:len(targets_raw), ...]
                self.train_event.wait()
                self.inference_net.load_state_dict(self.model.state_dict())
                self.inference_net.eval()
                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=True):
                    with torch.no_grad():
                        outputs = self.inference_net(self.inputs[cnt])
                        _, predicted = outputs.max(1)  # 返回这个batch中，值和索引
                        well_classified = self.targets[cnt] == predicted
                        mis_classified = ~well_classified
                        num_well_classified = well_classified.sum()
                        num_mis_classified = mis_classified.sum()
                        num_select_well = torch.ceil(num_well_classified * self.r).int()  # 这里要注意
                        self.weights[cnt] = torch.cat((torch.ones(num_mis_classified, dtype=torch.float32, device=self.device),
                                torch.full((num_select_well,), 1 / self.r, device=self.device)))
                        if isinstance(inputs_raw,torch.Tensor):
                            self.inputs_b[cnt][:num_mis_classified + num_select_well] = torch.cat((self.inputs[cnt][mis_classified], self.inputs[cnt][well_classified][:num_select_well]),dim=0)
                            self.inputs[cnt] = self.inputs_b[cnt][:num_mis_classified + num_select_well]
                        else:
                            self.inputs[cnt][0],self.inputs[cnt][2] = self.inputs[cnt][0].permute(1, 0, 2),self.inputs[cnt][2].permute(1, 0, 2)
                            self.inputs[cnt] = [torch.cat((tensor[mis_classified],tensor[well_classified][:num_select_well]),dim=0) for tensor in self.inputs[cnt]]
                            self.inputs[cnt][0],self.inputs[cnt][2]=self.inputs[cnt][0].permute(1, 0, 2),self.inputs[cnt][2].permute(1, 0, 2)
                        self.targets_b[cnt][:num_mis_classified + num_select_well] = torch.cat((self.targets[cnt][mis_classified], self.targets[cnt][well_classified][:num_select_well]), dim=0)
                        self.targets[cnt] = self.targets_b[cnt][:num_mis_classified + num_select_well]

                self.inference_event.record()
                self.barrier.wait()
                cnt ^= 1

                for inputs_raw, targets_raw in itertrainloader:
                    if isinstance(inputs_raw,torch.Tensor):
                        self.inputs_b[cnt][:len(targets_raw), ...] = inputs_raw.to(self.device, non_blocking=True)
                        self.inputs[cnt] = self.inputs_b[cnt][:len(targets_raw), ...]
                    else:
                        self.inputs[cnt] = [tensor.to(self.device, non_blocking=True) for tensor in inputs_raw]
                        self.inputs_b[cnt] = self.inputs[cnt]
                    self.targets_b[cnt][:len(targets_raw), ...] = targets_raw.to(self.device,non_blocking=True)
                    self.targets[cnt] = self.targets_b[cnt][:len(targets_raw), ...]
                    self.train_event.wait()
                    self.inference_net.load_state_dict(self.model.state_dict())
                    self.inference_net.eval()
                    with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=True):
                        with torch.no_grad():
                            outputs = self.inference_net(self.inputs[cnt])
                            _, predicted = outputs.max(1)  # 返回这个batch中，值和索引
                            well_classified = self.targets[cnt] == predicted
                            mis_classified = ~well_classified
                            num_well_classified = well_classified.sum()
                            num_mis_classified = mis_classified.sum()
                            num_select_well = torch.ceil(num_well_classified * self.r).int()  # 这里要注意
                            # gpu_utilization.append(nvml.nvmlDeviceGetUtilizationRates(self.handle).gpu)
                            self.weights[cnt] = torch.cat((torch.ones(num_mis_classified, dtype=torch.float32, device=self.device),
                                torch.full((num_select_well,), 1 / self.r, device=self.device)))
                            if isinstance(inputs_raw,torch.Tensor):
                                self.inputs_b[cnt][:num_mis_classified + num_select_well] = torch.cat((self.inputs[cnt][mis_classified], self.inputs[cnt][well_classified][:num_select_well]),dim=0)
                                self.inputs[cnt] = self.inputs_b[cnt][:num_mis_classified + num_select_well]
                            else:
                                self.inputs[cnt][0],self.inputs[cnt][2] = self.inputs[cnt][0].permute(1, 0, 2),self.inputs[cnt][2].permute(1, 0, 2)
                                self.inputs[cnt] = [torch.cat((tensor[mis_classified],tensor[well_classified][:num_select_well]),dim=0) for tensor in self.inputs[cnt]]
                                self.inputs[cnt][0],self.inputs[cnt][2]=self.inputs[cnt][0].permute(1, 0, 2),self.inputs[cnt][2].permute(1, 0, 2)
                            self.targets_b[cnt][:num_mis_classified + num_select_well] = torch.cat((self.targets[cnt][mis_classified], self.targets[cnt][well_classified][:num_select_well]), dim=0)
                            self.targets[cnt] = self.targets_b[cnt][:num_mis_classified + num_select_well]

                    self.inference_event.record()
                    self.barrier.wait()
                    cnt ^= 1
            # if sum(gpu_utilization) / len(gpu_utilization) < 95.0 and (torch.cuda.max_memory_reserved() < int(self.max_gpu_memory_GB * (1024 ** 3))):
            #     self.trainloader.batch_sampler.batch_size = self.trainloader.batch_sampler.batch_size + 8 
        torch.cuda.synchronize()
        self.inference_to_train.put(0)
        train_thread.join()



    def local_train(self):
        if self.args['algorithm'] == 'fedprox' or self.current_client.participation_times == 0:
            self.full_set()
        else:
            self.my()
        