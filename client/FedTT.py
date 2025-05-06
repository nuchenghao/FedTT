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

class FedTTClient(FedAvgTrainer):
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
        self.r = self.args['r']
        self.max_gpu_memory_GB = self.args['max_gpu_memory_GB']
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none').to(self.device)
        self.func = {
            "FedTT": self.FedTT,
            "FedTT_wo_gc": self.FedTT_wo_gc,
            "FedTT_w_ogc": self.FedTT_w_ogc,
            "FedTT_wo_pst":self.FedTT_wo_pst,
            "IBRS": self.IBRS,
            "FedTT_loss": self.FedTT_loss,
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

    def FedTT_loss(self):
        train_thread = threading.Thread(target=self.train, args=())
        train_thread.start()
        global_loss_threshold = self.current_client.loss
        loss_global = torch.zeros(self.current_client.train_set_len, device=self.device, dtype=torch.float)
        self.train_event.record()
        cnt = 0
        for epoch in range(self.local_epoch):
            total_correct = 0
            itertrainloader = iter(self.trainloader)
            self.inference_to_train.put(len(itertrainloader))
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
                        loss_global[total_correct:total_correct + len(targets_raw)] = self.criterion(outputs,self.targets[cnt])
                        well_classified = loss_global[total_correct:total_correct + len(targets_raw)] < global_loss_threshold
                        mis_classified = ~well_classified
                        num_well_classified = well_classified.sum()
                        num_mis_classified = mis_classified.sum()
                        num_select_well = torch.ceil(num_well_classified * self.r).int()
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
                            num_select_well = torch.ceil(num_well_classified * self.r).int()
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
                    cnt ^= 1
                    self.barrier.wait()
            global_loss_threshold = loss_global.mean()
        self.inference_to_train.put(0)
        train_thread.join()
        self.current_client.loss = global_loss_threshold.item()


    def FedTT(self):
        train_thread = threading.Thread(target=self.train, args=())
        train_thread.start()
        self.train_event.record()
        cnt = 0
        for epoch in range(self.local_epoch):
            itertrainloader = iter(self.trainloader)
            self.inference_to_train.put(len(itertrainloader))
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
                        _, predicted = outputs.max(1)
                        well_classified = self.targets[cnt] == predicted
                        mis_classified = ~well_classified
                        num_well_classified = well_classified.sum()
                        num_mis_classified = mis_classified.sum()
                        num_select_well = torch.ceil(num_well_classified * self.r).int()
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
                            _, predicted = outputs.max(1)
                            well_classified = self.targets[cnt] == predicted
                            mis_classified = ~well_classified
                            num_well_classified = well_classified.sum()
                            num_mis_classified = mis_classified.sum()
                            num_select_well = torch.ceil(num_well_classified * self.r).int()
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
        torch.cuda.synchronize()
        self.inference_to_train.put(0)
        train_thread.join()


    def FedTT_wo_gc(self):
        train_thread = threading.Thread(target=self.train, args=())
        train_thread.start()
        self.train_event.record()
        cnt = 0
        for epoch in range(self.local_epoch):
            itertrainloader = iter(self.trainloader)
            self.inference_to_train.put(len(itertrainloader))
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
                        _, predicted = outputs.max(1)
                        well_classified = self.targets[cnt] == predicted
                        mis_classified = ~well_classified
                        num_well_classified = well_classified.sum()
                        num_mis_classified = mis_classified.sum()
                        num_select_well = torch.ceil(num_well_classified * self.r).int()
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
                            _, predicted = outputs.max(1)
                            well_classified = self.targets[cnt] == predicted
                            mis_classified = ~well_classified
                            num_well_classified = well_classified.sum()
                            num_mis_classified = mis_classified.sum()
                            num_select_well = torch.ceil(num_well_classified * self.r).int()
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

    def FedTT_w_ogc(self):
        train_thread = threading.Thread(target=self.train, args=())
        train_thread.start()
        self.train_event.record()
        cnt = 0
        for epoch in range(self.local_epoch):
            itertrainloader = iter(self.trainloader)
            self.inference_to_train.put(len(itertrainloader))
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
                        _, predicted = outputs.max(1)
                        well_classified = self.targets[cnt] == predicted
                        mis_classified = ~well_classified
                        num_well_classified = well_classified.sum()
                        num_mis_classified = mis_classified.sum()
                        num_select_well = torch.ceil(num_well_classified * self.r).int()
                        self.weights[cnt] = torch.cat((torch.full((num_mis_classified,), (num_mis_classified + num_select_well) / len(targets_raw), dtype=torch.float32, device=self.device),
                                torch.full((num_select_well,), 1 / self.r * (num_mis_classified + num_select_well) / len(targets_raw), device=self.device)))
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
                            _, predicted = outputs.max(1)
                            well_classified = self.targets[cnt] == predicted
                            mis_classified = ~well_classified
                            num_well_classified = well_classified.sum()
                            num_mis_classified = mis_classified.sum()
                            num_select_well = torch.ceil(num_well_classified * self.r).int()
                            self.weights[cnt] = torch.cat((torch.full((num_mis_classified,), (num_mis_classified + num_select_well) / len(targets_raw), dtype=torch.float32, device=self.device),
                                torch.full((num_select_well,), 1 / self.r * (num_mis_classified + num_select_well) / len(targets_raw), device=self.device)))
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


    def FedTT_wo_pst(self):
        for _ in range(self.local_epoch):
            for inputs, targets in self.trainloader:
                if isinstance(inputs,torch.Tensor):
                    inputs = inputs.to(self.device, non_blocking=True)
                else:
                    inputs = [tensor.to(self.device, non_blocking=True) for tensor in inputs]
                targets = targets.to(self.device,non_blocking=True)
                self.model.eval()
                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=True):
                    with torch.no_grad():
                            outputs = self.model(inputs)
                            _, predicted = outputs.max(1)
                            well_classified = targets == predicted
                            mis_classified = ~well_classified
                            num_well_classified = well_classified.sum()
                            num_mis_classified = mis_classified.sum()
                            num_select_well = torch.ceil(num_well_classified * self.r).int()
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


    def IBRS(self):
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
                random_num = random.choice(range(lower, upper + 1))
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
        if (self.args["algorithm"] == "FedTT_wo_pst" and self.current_client.participation_times > 0) or self.args["algorithm"] == "IBRS":
            self.func[self.args["algorithm"]]()
        elif self.synchronization['prune'] and self.current_client.participation_times > 0:
            self.func[self.args["algorithm"]]()
            self.current_client.batch_size = self.trainloader.batch_sampler.batch_size
        else:
            self.full_set()
            self.current_client.batch_size = self.trainloader.batch_sampler.batch_size
