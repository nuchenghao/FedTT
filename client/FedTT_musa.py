import torch
import torch_musa
from copy import deepcopy
import queue
import threading
from fedavg import FedAvgTrainer
import numpy as np
from data.utils.datasets import DATASETS_SIZE
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
        self.inference_net = deepcopy(model).to(self.device) # To prevent race conditions, a deep copy of the model is required here
        self.train_stream = torch_musa.Stream()
        self.inference_stream = torch_musa.Stream()
        self.inference_event = torch_musa.Event()
        self.train_event = torch_musa.Event()
        # Buffer and indicators
        self.inputs = [None, None]
        self.inputs_b = [torch.zeros((self.args['batch_size'],*DATASETS_SIZE[self.args['dataset']]),dtype=torch.float32,device=self.device), torch.zeros((self.args['batch_size'],*DATASETS_SIZE[self.args['dataset']]),dtype=torch.float32,device=self.device)]
        self.targets = [None, None]
        self.targets_b = [torch.zeros((self.args['batch_size'],),dtype=torch.int64,device=self.device), torch.zeros((self.args['batch_size'],),dtype=torch.int64,device=self.device)]
        self.weights = [None, None]

        self.inference_to_train = queue.Queue() # The queue for exchanging information between the sampling thread and the training thread.
        self.barrier = threading.Barrier(2) # Barrier on the host
        self.r = self.args['r']
        self.max_gpu_memory_GB = self.args['max_gpu_memory_GB']
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none').to(self.device)
        self.func = {
            "FedTT": self.FedTT,
        }


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
        torch_musa.synchronize()
        
    def train(self):
        cnt = 1
        while True:
            iteration_number = self.inference_to_train.get()
            if iteration_number == 0:
                break
            else:
                with torch_musa.stream(self.train_stream):
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
        torch_musa.synchronize()

    def FedTT(self):
        train_thread = threading.Thread(target=self.train, args=())
        train_thread.start() # Start the training thread.
        self.train_event.record()
        cnt = 0
        for epoch in range(self.local_epoch):
            itertrainloader = iter(self.trainloader)
            self.inference_to_train.put(len(itertrainloader))
            inputs_raw, targets_raw = next(itertrainloader)
            with torch_musa.stream(self.inference_stream):
                self.train_event.wait()
                if isinstance(inputs_raw,torch.Tensor):
                    self.inputs_b[cnt][:len(targets_raw), ...] = inputs_raw.to(self.device, non_blocking=True)
                    self.inputs[cnt] = self.inputs_b[cnt][:len(targets_raw)]
                else:
                    self.inputs[cnt] = [tensor.to(self.device, non_blocking=True) for tensor in inputs_raw]
                    self.inputs_b[cnt] = self.inputs[cnt]
                self.targets_b[cnt][:len(targets_raw), ...] = targets_raw.to(self.device,non_blocking=True)
                self.targets[cnt] = self.targets_b[cnt][:len(targets_raw), ...]
                self.inference_net.load_state_dict(self.model.state_dict()) # Deepcopy the model
                self.inference_net.eval()
                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=True): # Mixed precision
                    with torch.no_grad():
                        outputs = self.inference_net(self.inputs[cnt])
                        _, predicted = outputs.max(1)
                        well_classified = self.targets[cnt] == predicted
                        mis_classified = ~well_classified
                        num_well_classified = well_classified.nonzero().shape[0]
                        num_mis_classified = mis_classified.nonzero().shape[0]
                        # The unimportant samples have a retention probability of r, which is equivalent to selecting r samples from the unimportant ones, and the sampler ensures randomness. Therefore, here we directly select the top r.
                        num_select_well = math.ceil(num_well_classified * self.r)
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
                            num_well_classified = well_classified.nonzero().shape[0]
                            num_mis_classified = mis_classified.nonzero().shape[0]
                            num_select_well = math.ceil(num_well_classified * self.r)
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
        torch_musa.synchronize()
        self.inference_to_train.put(0)
        train_thread.join()

 
    def local_train(self):
        if (self.args["algorithm"] == "FedTT_wo_pst" and self.current_client.participation_times > 0) or self.args["algorithm"] == "IBRS":
            self.func[self.args["algorithm"]]()
        elif self.synchronization['prune'] and self.current_client.participation_times > 0:
            self.func[self.args["algorithm"]]()
            self.current_client.batch_size = self.trainloader.batch_sampler.batch_size
        else:
            self.full_set() # The first time training with complete local data.
            self.current_client.batch_size = self.trainloader.batch_sampler.batch_size
