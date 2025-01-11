from copy import deepcopy
import torch
import queue
import threading
from py3nvml import py3nvml as nvml
from fedavg import FedAvgTrainer

nvml.nvmlInit()


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
        self.targets = [None, None]
        self.weights = [None, None]
        self.inference_to_train = queue.Queue()
        self.barrier = threading.Barrier(2)
        self.finish_one_epoch = threading.Event()
        self.r = self.args['r']
        # 修改评判器
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none').to(self.device)

        self.func = {
            "loss_fixed_batch_global_loss": self.loss_fixed_batch_global_loss,
            "loss_dynamic_batch_global_loss": self.loss_dynamic_batch_global_loss,
            "classify_fixed_batch": self.classify_fixed_batch,
            "classify_dynamic_batch": self.classify_dynamic_batch
        }

        nvml.nvmlInit()
        self.handle = nvml.nvmlDeviceGetHandleByIndex(
            0) if self.device == 'cuda:0' else nvml.nvmlDeviceGetHandleByIndex(1)


    def full_set(self):
        self.model.train()
        for _ in range(self.local_epoch):
            for inputs, targets in self.trainloader:
                inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device,
                                                                                        non_blocking=True)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets).mean()
                loss.backward()
                self.optimizer.step()
        torch.cuda.synchronize()

    def train(self):
        while True:
            iteration_number = self.inference_to_train.get()
            if iteration_number == 0:
                break
            else:
                cnt = 1
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
                    self.finish_one_epoch.set()
    

    @torch.no_grad
    def loss_fixed_batch_global_loss(self):
        train_thread = threading.Thread(target=self.train, args=())
        train_thread.start()
        global_loss_threshold = self.current_client.loss # 上一轮参与聚合的loss
        loss_global = torch.zeros(self.current_client.train_set_len, device=self.device, dtype=torch.float)
        self.train_event.record()
        for epoch in range(self.local_epoch):
            cnt = 0
            total_correct = 0
            itertrainloader = iter(self.trainloader)  # 创建trainloader的迭代器,这里会进行非常多的初始化(多进程的)！！！
            self.inference_to_train.put(len(itertrainloader))  # 训练线程预加载
            inputs_raw, targets_raw = next(itertrainloader)
            with torch.cuda.stream(self.inference_stream):
                self.train_event.wait() # 开始/ 等待上一轮训练流结束
                self.inputs[cnt], self.targets[cnt] = inputs_raw.to(self.device, non_blocking=True), targets_raw.to(self.device, non_blocking=True)
                self.inference_net.load_state_dict(self.model.state_dict())
                self.inference_net.eval()
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                    outputs = self.inference_net(self.inputs[cnt])
                    loss_global[total_correct:total_correct + len(targets_raw)] = self.criterion(outputs,self.targets[cnt])
                    well_classified = loss_global[total_correct:total_correct + len(targets_raw)] < global_loss_threshold
                    mis_classified = ~well_classified
                    num_well_classified = well_classified.sum()
                    num_mis_classified = mis_classified.sum()
                    num_select_well = torch.ceil(num_well_classified * self.r).int()  # 这里要注意
                    total_correct += len(targets_raw)
                    select_well_classified_inputs = self.inputs[cnt][well_classified][:num_select_well]
                    select_well_classified_targets = self.targets[cnt][well_classified][:num_select_well]
                    self.weights[cnt] = torch.cat((torch.ones(num_mis_classified, dtype=torch.float32, device=self.device),
                            torch.full((num_select_well,), 1 / self.r, device=self.device)))
                    self.inputs[cnt] = torch.cat((self.inputs[cnt][mis_classified], select_well_classified_inputs),dim=0)
                    self.targets[cnt] = torch.cat((self.targets[cnt][mis_classified], select_well_classified_targets), dim=0)
                self.inference_event.record() # 让训练流开始
                self.train_event.record() # 让自己的筛选流开始
                self.barrier.wait() # cpu开始发射训练流
                cnt ^= 1

                for inputs_raw, targets_raw in itertrainloader:
                    self.train_event.wait()
                    self.inputs[cnt], self.targets[cnt] = inputs_raw.to(self.device, non_blocking=True), targets_raw.to(self.device,non_blocking=True)
                    self.inference_net.load_state_dict(self.model.state_dict())
                    self.inference_net.eval()
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                        outputs = self.inference_net(self.inputs[cnt])
                        loss_global[total_correct:total_correct + len(targets_raw)] = self.criterion(outputs,self.targets[cnt])
                        well_classified = loss_global[total_correct:total_correct + len(targets_raw)] < global_loss_threshold
                        mis_classified = ~well_classified
                        num_well_classified = well_classified.sum()
                        num_mis_classified = mis_classified.sum()
                        num_select_well = torch.ceil(num_well_classified * self.r).int()  # 这里要注意
                        total_correct += len(targets_raw)
                        select_well_classified_inputs = self.inputs[cnt][well_classified][:num_select_well]
                        select_well_classified_targets = self.targets[cnt][well_classified][:num_select_well]
                        self.weights[cnt] = torch.cat((torch.ones(num_mis_classified, dtype=torch.float32, device=self.device),
                                                       torch.full((num_select_well,), 1 / self.r, device=self.device)))
                        self.inputs[cnt] = torch.cat((self.inputs[cnt][mis_classified], select_well_classified_inputs), dim=0)
                        self.targets[cnt] = torch.cat((self.targets[cnt][mis_classified], select_well_classified_targets),dim=0)
                    self.inference_event.record()
                    cnt ^= 1
                    self.barrier.wait()
            global_loss_threshold = loss_global.mean()
            new_batch_size = min(int(
                (((self.args["batch_size"] / ((1.0 - self.r) * (
                    (loss_global > global_loss_threshold).sum().item()) / self.current_client.train_set_len + self.r)) // 32)+1) * 32) , 256)
            self.trainloader.batch_sampler.batch_size = new_batch_size
            self.finish_one_epoch.wait()
            self.finish_one_epoch.clear()
        torch.cuda.synchronize()
        self.inference_to_train.put(0)
        train_thread.join()
        self.current_client.loss = global_loss_threshold.item()
        return 100.0
    
    @torch.no_grad
    def loss_dynamic_batch_global_loss(self):
        gpu_utilization = []
        train_thread = threading.Thread(target=self.train, args=())
        train_thread.start()
        global_loss_threshold = self.current_client.loss
        loss_global = torch.zeros(self.current_client.train_set_len, device=self.device, dtype=torch.float)
        self.train_event.record()
        for epoch in range(self.local_epoch):
            cnt = 0
            total_correct = 0
            itertrainloader = iter(self.trainloader)  # 创建trainloader的迭代器
            self.inference_to_train.put((len(itertrainloader)))  # 训练线程预加载
            inputs_raw, targets_raw = next(itertrainloader)
            with torch.cuda.stream(self.inference_stream):
                self.inputs[cnt], self.targets[cnt] = inputs_raw.to(self.device, non_blocking=True), targets_raw.to(self.device, non_blocking=True)
                self.inference_net.load_state_dict(self.model.state_dict())
                self.inference_net.eval()
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                    outputs = self.inference_net(self.inputs[cnt])
                    loss_global[total_correct:total_correct + len(targets_raw)] = self.criterion(outputs,self.targets[cnt])
                    well_classified = loss_global[total_correct:total_correct + len(targets_raw)] < global_loss_threshold
                    mis_classified = ~well_classified
                    num_well_classified = well_classified.sum()
                    num_mis_classified = mis_classified.sum()
                    num_select_well = torch.ceil(num_well_classified * self.r).int()  # 这里要注意
                    total_correct += len(targets_raw)
                    select_well_classified_inputs = self.inputs[cnt][well_classified][:num_select_well]
                    select_well_classified_targets = self.targets[cnt][well_classified][:num_select_well]
                    self.weights[cnt] = torch.cat((torch.ones(num_mis_classified, dtype=torch.float32, device=self.device),
                            torch.full((num_select_well,), 1 / self.r, device=self.device)))
                    self.inputs[cnt] = torch.cat((self.inputs[cnt][mis_classified], select_well_classified_inputs),dim=0)
                    self.targets[cnt] = torch.cat((self.targets[cnt][mis_classified], select_well_classified_targets), dim=0)
                self.inference_event.record()
                self.train_event.record()
                self.barrier.wait()
                cnt ^= 1

                for inputs_raw, targets_raw in itertrainloader:
                    self.train_event.wait()
                    self.inputs[cnt], self.targets[cnt] = inputs_raw.to(self.device, non_blocking=True), targets_raw.to(self.device,non_blocking=True)
                    self.inference_net.load_state_dict(self.model.state_dict())
                    self.inference_net.eval()
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                        outputs = self.inference_net(self.inputs[cnt])
                        loss_global[total_correct:total_correct + len(targets_raw)] = self.criterion(outputs , self.targets[cnt])
                        well_classified = loss_global[total_correct:total_correct + len(targets_raw)] < global_loss_threshold
                        mis_classified = ~well_classified
                        num_well_classified = well_classified.sum()
                        num_mis_classified = mis_classified.sum()
                        num_select_well = torch.ceil(num_well_classified * self.r).int()  # 这里要注意
                        total_correct += len(targets_raw)
                        gpu_utilization.append(nvml.nvmlDeviceGetUtilizationRates(self.handle).gpu)
                        select_well_classified_inputs = self.inputs[cnt][well_classified][:num_select_well]
                        select_well_classified_targets = self.targets[cnt][well_classified][:num_select_well]
                        self.weights[cnt] = torch.cat((torch.ones(num_mis_classified, dtype=torch.float32, device=self.device),
                                torch.full((num_select_well,), 1 / self.r, device=self.device)))
                        self.inputs[cnt] = torch.cat((self.inputs[cnt][mis_classified], select_well_classified_inputs), dim=0)
                        self.targets[cnt] = torch.cat((self.targets[cnt][mis_classified], select_well_classified_targets),dim=0)
                    self.inference_event.record()
                    cnt ^= 1
                    self.barrier.wait()
            global_loss_threshold = loss_global.mean()
            self.finish_one_epoch.wait()
            self.finish_one_epoch.clear()
        torch.cuda.synchronize()
        self.inference_to_train.put(0)
        train_thread.join()
        self.current_client.loss = global_loss_threshold.item()
        return sum(gpu_utilization) / len(gpu_utilization)
    
    @torch.no_grad
    def classify_fixed_batch(self):
        train_thread = threading.Thread(target=self.train, args=())
        train_thread.start()
        total_dataset = self.current_client.train_set_len
        self.train_event.record()
        for epoch in range(self.local_epoch):
            cnt = 0
            total_correct = 0
            itertrainloader = iter(self.trainloader)  # 创建trainloader的迭代器
            self.inference_to_train.put((len(itertrainloader)))  # 训练线程预加载
            inputs_raw, targets_raw = next(itertrainloader)
            with torch.cuda.stream(self.inference_stream):
                self.train_event.wait() # 开始/ 等待上一轮训练流结束
                self.inputs[cnt], self.targets[cnt] = inputs_raw.to(self.device, non_blocking=True), targets_raw.to(self.device, non_blocking=True)
                self.inference_net.load_state_dict(self.model.state_dict())
                self.inference_net.eval()
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
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
                self.train_event.record()
                self.barrier.wait()
                cnt ^= 1

                for inputs_raw, targets_raw in itertrainloader:
                    self.train_event.wait()
                    self.inputs[cnt], self.targets[cnt] = inputs_raw.to(self.device, non_blocking=True), targets_raw.to(self.device, non_blocking=True)
                    self.inference_net.load_state_dict(self.model.state_dict())
                    self.inference_net.eval()
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
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
            new_batch_size = int(
                (((self.args["batch_size"] / (1 + (self.r - 1) * (total_correct / total_dataset))) // 32)) * 32)
            self.trainloader.batch_sampler.batch_size = new_batch_size
            self.finish_one_epoch.wait()
            self.finish_one_epoch.clear()
        torch.cuda.synchronize()
        self.inference_to_train.put(0)
        train_thread.join()
        return 100.0

    @torch.no_grad
    def classify_dynamic_batch(self):
        train_thread = threading.Thread(target=self.train, args=())
        train_thread.start()
        gpu_utilization = []
        self.train_event.record()
        for epoch in range(self.local_epoch):
            cnt = 0
            itertrainloader = iter(self.trainloader)  # 创建trainloader的迭代器
            self.inference_to_train.put((len(itertrainloader)))  # 训练线程预加载,这里的值时batch_size会load的次数
            inputs_raw, targets_raw = next(itertrainloader)
            with torch.cuda.stream(self.inference_stream):
                self.train_event.wait()
                self.inputs[cnt], self.targets[cnt] = inputs_raw.to(self.device, non_blocking=True), targets_raw.to(self.device, non_blocking=True)
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                    self.inference_net.load_state_dict(self.model.state_dict())
                    self.inference_net.eval()
                    outputs = self.inference_net(self.inputs[cnt])
                    _, predicted = outputs.max(1)  # 返回这个batch中，值和索引
                    well_classified = self.targets[cnt] == predicted
                    mis_classified = ~well_classified
                    num_well_classified = well_classified.sum()
                    num_mis_classified = mis_classified.sum()
                    num_select_well = torch.ceil(num_well_classified * self.r).int()  # 这里要注意
                    self.weights[cnt] = torch.cat((torch.ones(num_mis_classified, dtype=torch.float32, device=self.device),
                            torch.full((num_select_well,), 1 / self.r, device=self.device)))
                    self.inputs[cnt] = torch.cat((self.inputs[cnt][mis_classified], self.inputs[cnt][well_classified][:num_select_well]),dim=0)
                    self.targets[cnt] = torch.cat((self.targets[cnt][mis_classified], self.targets[cnt][well_classified][:num_select_well]),dim=0)

                self.inference_event.record()
                self.train_event.record()
                self.barrier.wait()
                cnt ^= 1

                for inputs_raw, targets_raw in itertrainloader:
                    self.train_event.wait()
                    self.inputs[cnt], self.targets[cnt] = inputs_raw.to(self.device, non_blocking=True), targets_raw.to(self.device, non_blocking=True)
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                        self.inference_net.load_state_dict(self.model.state_dict())
                        self.inference_net.eval()
                        outputs = self.inference_net(self.inputs[cnt])
                        _, predicted = outputs.max(1)  # 返回这个batch中，值和索引
                        well_classified = self.targets[cnt] == predicted
                        mis_classified = ~well_classified
                        num_well_classified = well_classified.sum()
                        num_mis_classified = mis_classified.sum()
                        num_select_well = torch.ceil(num_well_classified * self.r).int()  # 这里要注意
                        gpu_utilization.append(nvml.nvmlDeviceGetUtilizationRates(self.handle).gpu)
                        self.weights[cnt] = torch.cat((torch.ones(num_mis_classified, dtype=torch.float32, device=self.device),
                            torch.full((num_select_well,), 1 / self.r, device=self.device)))
                        self.inputs[cnt] = torch.cat((self.inputs[cnt][mis_classified], self.inputs[cnt][well_classified][:num_select_well]),dim=0)
                        self.targets[cnt] = torch.cat((self.targets[cnt][mis_classified],self.targets[cnt][well_classified][:num_select_well]), dim=0)

                    self.inference_event.record()
                    self.barrier.wait()
                    cnt ^= 1
            self.finish_one_epoch.wait()
            self.finish_one_epoch.clear()
        torch.cuda.synchronize()
        self.inference_to_train.put(0)
        train_thread.join()
        return sum(gpu_utilization) / len(gpu_utilization)

    def local_train(self):
        if self.synchronization['prune'] and self.current_client.participation_times > 0:
            gpu_utilization = self.func[self.args["algorithm"]]()
            if gpu_utilization < 98.0:
                self.current_client.batch_size = min(self.trainloader.batch_sampler.batch_size + 16 , 256)  # 记录当前client的batch——size
            else:
                self.current_client.batch_size = self.trainloader.batch_sampler.batch_size  # 记录当前client的batch——size
        else:
            self.full_set()
            self.current_client.batch_size = self.trainloader.batch_sampler.batch_size  # 记录当前client的batch——size
