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
from utls.models import MODEL_DICT
from data.utils.datasets import DATA_NUM_CLASSES_DICT, DATASETS , DATASETS_COLLATE_FN
from utls.dataset import CustomSampler

class BaseClient:
    def __init__(self, client_id, train_index, batch_size):
        self.client_id = client_id
        self.train_set_index = np.array(train_index)
        self.train_set_len = len(train_index)
        self.participation_times = 0

        self.batch_size = batch_size

        self.training_time = 0
        self.pretrained_accuracy = 0
        self.accuracy = 0
        self.loss = 0.0
        self.grad = None #存梯度值
        self.buffer = None # 存persistent_buffers

        self.training_time_record = {}

    def participate_once(self):
        self.participation_times += 1

class FedAvgTrainerOnDevice:
    def __init__(
            self,
            args
    ):
        self.args = args
        self.device ="cuda"
        self.data_num_classes = DATA_NUM_CLASSES_DICT[self.args['dataset']]
        self.model = MODEL_DICT[self.args["model"]](self.data_num_classes) # 暂时放置在CPU上
        self.current_client_instance = None

        self.trainset = DATASETS[self.args['dataset']](PROJECT_DIR / "data" / self.args["dataset"], "train")
        self.train_sampler = CustomSampler(list(range(len(self.trainset))))
        self.trainloader = DataLoader(Subset(self.trainset, list(range(len(self.trainset)))), self.args["batch_size"], num_workers=2,collate_fn = DATASETS_COLLATE_FN[self.args['dataset']], persistent_workers=True,
                                      sampler=self.train_sampler,)
        self.local_epoch = self.args["local_epoch"]
        self._criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none').to(self.device)
        self.optimizer = None
        self.timer = Timer()  # 训练计时器
    

    def load_dataset(self):
        self.trainloader.sampler.set_index(self.current_client_instance.train_set_index)  # 在里面实现了深拷贝
        self.trainloader.batch_sampler.batch_size = self.current_client_instance.batch_size
    
    def set_parameters(self,model_parameters):
        self.model.load_state_dict(model_parameters)
        self.model = self.model.to(self.device) # 转移到gpu上
        self.optimizer = torch.optim.SGD(params=self.model.parameters(),lr=self.args["lr"],momentum=self.args["momentum"],weight_decay=self.args["weight_decay"],)

    def start(self,global_epoch, client_instance, model_parameters):
        self.timer.start()
        self.current_client_instance = client_instance
        self.set_parameters(model_parameters)  # 设置参数
        self.load_dataset()

        self.local_train() # 本地训练
        self.model = self.model.to("cpu") # 训练完成后放置到CPU上

        # 拷贝模型参数
        current_client_instance_model_dict = {key: copy.deepcopy(value) for key, value in self.model.state_dict().items()}  # 一定要深拷贝
        self.timer.stop()

        self.current_client_instance.training_time_record[global_epoch] = self.timer.times[-1]
        self.current_client_instance.participate_once()

        # 返回训练后的模型参数，训练时间
        return current_client_instance_model_dict, self.current_client_instance.training_time_record[global_epoch]

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
                loss.backward()
                self.optimizer.step()
        torch.cuda.synchronize()
    
    def local_train(self):
        """
        本地训练函数，后面实现的算法要重写这个函数.
        可以确定的是：在训练前，模型已经在设备上了
        """
        self.full_set()