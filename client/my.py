from copy import deepcopy
import torch
import queue
import threading
from py3nvml import py3nvml as nvml
from fedavg import FedAvgTrainer
import numpy as np
from data.utils.datasets import DATASETS_SIZE
nvml.nvmlInit()
import random
import math

class myFed(FedAvgTrainer):
    def __init__(self,
                 device,
                 model,
                 trainloader,
                 testloader,
                 args: dict, ):
        super().__init__(device, model, trainloader, testloader, args)
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
        self.max_gpu_memory_GB = self.args['max_gpu_memory_GB']
        # 修改评判器
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none').to(self.device)

        self.func = {
            "loss_fixed_batch_global_loss": self.loss_fixed_batch_global_loss,
            "loss_dynamic_batch_global_loss": self.loss_dynamic_batch_global_loss,
            "classify_fixed_batch": self.classify_fixed_batch,
            "classify_dynamic_batch": self.classify_dynamic_batch,
            "classify_dynamic_batch_wo_weights": self.classify_dynamic_batch_wo_weights,
            "loss_dynamic_batch_global_loss_wo_weights": self.loss_dynamic_batch_global_loss_wo_weights,
            "woparallel":self.woparallel,
            "random_select": self.random_select
        }

        nvml.nvmlInit()
        self.handle = nvml.nvmlDeviceGetHandleByIndex(0) 


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
                loss = self.criterion(outputs, targets).mean()
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
                        loss.backward()
                        self.optimizer.step()
                        self.train_event.record()
        torch.cuda.synchronize()
    

    def loss_fixed_batch_global_loss(self):
        train_thread = threading.Thread(target=self.train, args=())
        train_thread.start()
        global_loss_threshold = self.current_client.loss # 上一轮参与聚合的loss
        loss_global = torch.zeros(self.current_client.train_set_len, device=self.device, dtype=torch.float)
        self.train_event.record()
        cnt = 0
        for epoch in range(self.local_epoch):
            total_correct = 0
            itertrainloader = iter(self.trainloader)  # 创建trainloader的迭代器,这里会进行非常多的初始化(多进程的)！！！
            self.inference_to_train.put(len(itertrainloader))  # 训练线程预加载
            inputs_raw, targets_raw = next(itertrainloader)
            with torch.cuda.stream(self.inference_stream):
                self.inputs[cnt], self.targets[cnt] = inputs_raw.to(self.device, non_blocking=True), targets_raw.to(self.device, non_blocking=True)
                self.train_event.wait() # 开始/ 等待上一轮训练流结束
                self.inference_net.load_state_dict(self.model.state_dict())
                self.inference_net.eval()
                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=True):
                    with torch.no_grad():
                        outputs = self.inference_net(self.inputs[cnt])
                        loss_global[total_correct:total_correct + len(targets_raw)] = self.criterion(outputs,self.targets[cnt])
                        well_classified = loss_global[total_correct:total_correct + len(targets_raw)] < global_loss_threshold
                        mis_classified = ~well_classified
                        num_well_classified = well_classified.sum()
                        num_mis_classified = mis_classified.sum()
                        num_select_well = torch.ceil(num_well_classified * self.r).int()  # 这里要注意
                        total_correct += len(targets_raw)
                        self.weights[cnt] = torch.cat((torch.ones(num_mis_classified, dtype=torch.float32, device=self.device),
                                torch.full((num_select_well,), 1 / self.r, device=self.device)))
                        self.inputs[cnt] = torch.cat((self.inputs[cnt][mis_classified], self.inputs[cnt][well_classified][:num_select_well]),dim=0)
                        self.targets[cnt] = torch.cat((self.targets[cnt][mis_classified], self.targets[cnt][well_classified][:num_select_well]), dim=0)
                self.inference_event.record() # 让训练流开始
                self.barrier.wait() # cpu开始发射训练流
                cnt ^= 1

                for inputs_raw, targets_raw in itertrainloader:
                    self.inputs[cnt], self.targets[cnt] = inputs_raw.to(self.device, non_blocking=True), targets_raw.to(self.device,non_blocking=True)
                    self.train_event.wait()
                    self.inference_net.load_state_dict(self.model.state_dict())
                    self.inference_net.eval()
                    with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=True):
                        with torch.no_grad():
                            outputs = self.inference_net(self.inputs[cnt])
                            loss_global[total_correct:total_correct + len(targets_raw)] = self.criterion(outputs,self.targets[cnt])
                            well_classified = loss_global[total_correct:total_correct + len(targets_raw)] < global_loss_threshold
                            mis_classified = ~well_classified
                            num_well_classified = well_classified.sum()
                            num_mis_classified = mis_classified.sum()
                            num_select_well = torch.ceil(num_well_classified * self.r).int()  # 这里要注意
                            total_correct += len(targets_raw)
                            self.weights[cnt] = torch.cat((torch.ones(num_mis_classified, dtype=torch.float32, device=self.device),
                                                        torch.full((num_select_well,), 1 / self.r, device=self.device)))
                            self.inputs[cnt] = torch.cat((self.inputs[cnt][mis_classified], self.inputs[cnt][well_classified][:num_select_well]), dim=0)
                            self.targets[cnt] = torch.cat((self.targets[cnt][mis_classified], self.targets[cnt][well_classified][:num_select_well]),dim=0)
                    self.inference_event.record()
                    cnt ^= 1
                    self.barrier.wait()
            global_loss_threshold = loss_global.mean()
            new_batch_size = min(int(
                (((self.args["batch_size"] / ((1.0 - self.r) * (
                    (loss_global > global_loss_threshold).sum().item()) / self.current_client.train_set_len + self.r)) // 32)) * 32) , 256)
            self.trainloader.batch_sampler.batch_size = new_batch_size
        self.inference_to_train.put(0)
        train_thread.join()
        self.current_client.loss = global_loss_threshold.item()
    

    def loss_dynamic_batch_global_loss(self):
        # gpu_utilization = []
        train_thread = threading.Thread(target=self.train, args=())
        train_thread.start()
        global_loss_threshold = self.current_client.loss
        loss_global = torch.zeros(self.current_client.train_set_len, device=self.device, dtype=torch.float)
        self.train_event.record()
        cnt = 0
        for epoch in range(self.local_epoch):
            # torch.cuda.reset_peak_memory_stats() # 重置显存峰值统计
            # gpu_utilization = []
            total_correct = 0
            itertrainloader = iter(self.trainloader)  # 创建trainloader的迭代器
            self.inference_to_train.put(len(itertrainloader))  # 训练线程预加载
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
                self.train_event.wait() # 开始/ 等待上一轮训练流结束
                self.inference_net.load_state_dict(self.model.state_dict())
                self.inference_net.eval()
                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=True):
                    with torch.no_grad():
                        outputs = self.inference_net(self.inputs[cnt])
                        loss_global[total_correct:total_correct + len(targets_raw)] = self.criterion(outputs,self.targets[cnt])
                        well_classified = loss_global[total_correct:total_correct + len(targets_raw)] < global_loss_threshold
                        mis_classified = ~well_classified
                        num_well_classified = well_classified.sum()
                        num_mis_classified = mis_classified.sum()
                        num_select_well = torch.ceil(num_well_classified * self.r).int()  # 这里要注意
                        total_correct += len(targets_raw)
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
                            loss_global[total_correct:total_correct + len(targets_raw)] = self.criterion(outputs , self.targets[cnt])
                            well_classified = loss_global[total_correct:total_correct + len(targets_raw)] < global_loss_threshold
                            mis_classified = ~well_classified
                            num_well_classified = well_classified.sum()
                            num_mis_classified = mis_classified.sum()
                            num_select_well = torch.ceil(num_well_classified * self.r).int()  # 这里要注意
                            total_correct += len(targets_raw)
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
                    cnt ^= 1
                    self.barrier.wait()
            global_loss_threshold = loss_global.mean()
            # if sum(gpu_utilization) / len(gpu_utilization) < 95.0 and (torch.cuda.max_memory_reserved() < int(self.max_gpu_memory_GB * (1024 ** 3))):
            #     self.trainloader.batch_sampler.batch_size = self.trainloader.batch_sampler.batch_size + 16
        self.inference_to_train.put(0)
        train_thread.join()
        self.current_client.loss = global_loss_threshold.item()
    

    def classify_fixed_batch(self):
        train_thread = threading.Thread(target=self.train, args=())
        train_thread.start()
        total_dataset = self.current_client.train_set_len
        self.train_event.record()
        cnt = 0
        for epoch in range(self.local_epoch):
            total_correct = 0
            itertrainloader = iter(self.trainloader)  # 创建trainloader的迭代器
            self.inference_to_train.put(len(itertrainloader))  # 训练线程预加载
            inputs_raw, targets_raw = next(itertrainloader)
            with torch.cuda.stream(self.inference_stream):
                self.inputs[cnt], self.targets[cnt] = inputs_raw.to(self.device, non_blocking=True), targets_raw.to(self.device, non_blocking=True)
                self.train_event.wait() # 开始/ 等待上一轮训练流结束
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
                        total_correct += num_well_classified
                        num_select_well = torch.ceil(num_well_classified * self.r).int()  # 这里要注意
                        self.weights[cnt] = torch.cat((torch.ones(num_mis_classified, dtype=torch.float32, device=self.device),
                                torch.full((num_select_well,), 1 / self.r, device=self.device)))
                        self.inputs[cnt] = torch.cat((self.inputs[cnt][mis_classified], self.inputs[cnt][well_classified][:num_select_well]),dim=0)
                        self.targets[cnt] = torch.cat((self.targets[cnt][mis_classified], self.targets[cnt][well_classified][:num_select_well]),dim=0)

                self.inference_event.record()
                self.barrier.wait()
                cnt ^= 1

                for inputs_raw, targets_raw in itertrainloader:
                    self.inputs[cnt], self.targets[cnt] = inputs_raw.to(self.device, non_blocking=True), targets_raw.to(self.device, non_blocking=True)
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
                            total_correct += num_well_classified
                            num_select_well = torch.ceil(num_well_classified * self.r).int()  # 这里要注意
                            self.weights[cnt] = torch.cat((torch.ones(num_mis_classified, dtype=torch.float32, device=self.device),
                                    torch.full((num_select_well,), 1 / self.r, device=self.device)))
                            self.inputs[cnt] = torch.cat((self.inputs[cnt][mis_classified], self.inputs[cnt][well_classified][:num_select_well]),dim=0)
                            self.targets[cnt] = torch.cat((self.targets[cnt][mis_classified],self.targets[cnt][well_classified][:num_select_well]), dim=0)

                    self.inference_event.record()
                    cnt ^= 1
                    self.barrier.wait()
            new_batch_size = min(int((((self.args["batch_size"] / (1 + (self.r - 1) * (total_correct / total_dataset))) // 32)) * 32) , 256)
            self.trainloader.batch_sampler.batch_size = new_batch_size
        self.inference_to_train.put(0)
        train_thread.join()


    def classify_dynamic_batch(self):
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
                self.train_event.wait()
                if isinstance(inputs_raw,torch.Tensor):
                    self.inputs_b[cnt][:len(targets_raw), ...] = inputs_raw.to(self.device, non_blocking=True)
                    self.inputs[cnt] = self.inputs_b[cnt][:len(targets_raw)]
                else:
                    self.inputs[cnt] = [tensor.to(self.device, non_blocking=True) for tensor in inputs_raw]
                    self.inputs_b[cnt] = self.inputs[cnt]
                self.targets_b[cnt][:len(targets_raw), ...] = targets_raw.to(self.device,non_blocking=True)
                self.targets[cnt] = self.targets_b[cnt][:len(targets_raw), ...]
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
                    self.train_event.wait()
                    if isinstance(inputs_raw,torch.Tensor):
                        self.inputs_b[cnt][:len(targets_raw), ...] = inputs_raw.to(self.device, non_blocking=True)
                        self.inputs[cnt] = self.inputs_b[cnt][:len(targets_raw), ...]
                    else:
                        self.inputs[cnt] = [tensor.to(self.device, non_blocking=True) for tensor in inputs_raw]
                        self.inputs_b[cnt] = self.inputs[cnt]
                    self.targets_b[cnt][:len(targets_raw), ...] = targets_raw.to(self.device,non_blocking=True)
                    self.targets[cnt] = self.targets_b[cnt][:len(targets_raw), ...]
                    
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

    def classify_dynamic_batch_wo_weights(self):
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
                        self.weights[cnt] = torch.ones(num_mis_classified + num_select_well, dtype=torch.float32, device=self.device)
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
                            self.weights[cnt] = torch.ones(num_mis_classified + num_select_well, dtype=torch.float32, device=self.device)
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
        torch.cuda.synchronize()
        self.inference_to_train.put(0)
        train_thread.join()


    def loss_dynamic_batch_global_loss_wo_weights(self):
        # gpu_utilization = []
        train_thread = threading.Thread(target=self.train, args=())
        train_thread.start()
        global_loss_threshold = self.current_client.loss
        loss_global = torch.zeros(self.current_client.train_set_len, device=self.device, dtype=torch.float)
        self.train_event.record()
        cnt = 0
        for epoch in range(self.local_epoch):
            torch.cuda.reset_peak_memory_stats() # 重置显存峰值统计
            # gpu_utilization = []
            total_correct = 0
            itertrainloader = iter(self.trainloader)  # 创建trainloader的迭代器
            self.inference_to_train.put(len(itertrainloader))  # 训练线程预加载
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
                self.train_event.wait() # 开始/ 等待上一轮训练流结束
                self.inference_net.load_state_dict(self.model.state_dict())
                self.inference_net.eval()
                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=True):
                    with torch.no_grad():
                        outputs = self.inference_net(self.inputs[cnt])
                        loss_global[total_correct:total_correct + len(targets_raw)] = self.criterion(outputs,self.targets[cnt])
                        well_classified = loss_global[total_correct:total_correct + len(targets_raw)] < global_loss_threshold
                        mis_classified = ~well_classified
                        num_well_classified = well_classified.sum()
                        num_mis_classified = mis_classified.sum()
                        num_select_well = torch.ceil(num_well_classified * self.r).int()  # 这里要注意
                        total_correct += len(targets_raw)
                        self.weights[cnt] = torch.ones(num_mis_classified + num_select_well, dtype=torch.float32, device=self.device)
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
                            loss_global[total_correct:total_correct + len(targets_raw)] = self.criterion(outputs , self.targets[cnt])
                            well_classified = loss_global[total_correct:total_correct + len(targets_raw)] < global_loss_threshold
                            mis_classified = ~well_classified
                            num_well_classified = well_classified.sum()
                            num_mis_classified = mis_classified.sum()
                            num_select_well = torch.ceil(num_well_classified * self.r).int()  # 这里要注意
                            total_correct += len(targets_raw)
                            # gpu_utilization.append(nvml.nvmlDeviceGetUtilizationRates(self.handle).gpu)
                            self.weights[cnt] = torch.ones(num_mis_classified + num_select_well, dtype=torch.float32, device=self.device)
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
                    cnt ^= 1
                    self.barrier.wait()
            global_loss_threshold = loss_global.mean()
        self.inference_to_train.put(0)
        train_thread.join()
        self.current_client.loss = global_loss_threshold.item()

    def woparallel(self):
        for _ in range(self.local_epoch):
            for inputs, targets in self.trainloader:
                if isinstance(inputs,torch.Tensor):
                    inputs = inputs.to(self.device, non_blocking=True)
                else:
                    inputs = [tensor.to(self.device, non_blocking=True) for tensor in inputs]
                targets = targets.to(self.device,non_blocking=True)
                self.model.eval()
                with torch.no_grad():
                    # with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=True):
                        outputs = self.model(inputs)
                        _, predicted = outputs.max(1)  # 返回这个batch中，值和索引
                        well_classified = targets == predicted
                        mis_classified = ~well_classified
                        num_well_classified = well_classified.sum()
                        num_mis_classified = mis_classified.sum()
                        num_select_well = torch.ceil(num_well_classified * self.r).int()  # 这里要注意
                        weights = torch.cat((torch.ones(num_mis_classified, dtype=torch.float32, device=self.device),
                                    torch.full((num_select_well,), 1 / self.r, device=self.device)))
                        inputs = torch.cat((inputs[mis_classified], inputs[well_classified][:num_select_well]),dim=0)
                        targets = torch.cat((targets[mis_classified], targets[well_classified][:num_select_well]), dim=0)
                self.model.train()
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss = (loss * weights).mean()
                loss.backward()
                self.optimizer.step()
        torch.cuda.synchronize()

    def random_select(self):
        self.model.train()
        for _ in range(self.local_epoch):
            for inputs, targets in self.trainloader:
                upper = math.ceil(len(targets) - (1-self.r) * len(targets) * self.synchronization['accuracy'])
                lower = math.ceil(len(targets) * self.r)
                if isinstance(inputs,torch.Tensor):
                    inputs = inputs.to(self.device, non_blocking=True)
                else:
                    inputs = [tensor.to(self.device, non_blocking=True) for tensor in inputs]
                targets = targets.to(self.device,non_blocking=True)
                random_num = random.choice(range(lower, upper + 1))  # +1 确保包含 upper
                index = random.sample(range(len(targets)), random_num)
                inputs = inputs[index].contiguous()
                targets = targets[index].contiguous()
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets).mean()
                loss.backward()
                self.optimizer.step()
        torch.cuda.synchronize()


    def local_train(self):
        if self.args["algorithm"] == "woparallel" or self.args["algorithm"] == "random_select":
            self.func[self.args["algorithm"]]()
        elif self.synchronization['prune'] and self.current_client.participation_times > 0:
            self.func[self.args["algorithm"]]()
            self.current_client.batch_size = self.trainloader.batch_sampler.batch_size  # 记录当前client的batch——size
        else:
            self.full_set()
            self.current_client.batch_size = self.trainloader.batch_sampler.batch_size  # 记录当前client的batch——size
