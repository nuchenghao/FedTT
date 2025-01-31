import pickle
import sys
import json
from tqdm import tqdm
import random
from pathlib import Path
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List
import numpy as np
import wandb
import torch
import yaml
from rich.console import Console
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from collections import Counter
import queue
PROJECT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(PROJECT_DIR.as_posix())
sys.path.append(PROJECT_DIR.joinpath("src").as_posix())

from utls.utils import (
    TRAIN_LOG,
    Logger,
    fix_random_seed,
    NN_state_load,
    get_argparser,
    evaluate
)
from client.ODE import ODETrainer, ODEClient
from server.fedavg import FedAvgServer
from utls.models import MODEL_DICT
from data.utils.datasets import DATA_NUM_CLASSES_DICT, DATASETS
from utls.dataset import NeedIndexDataset


class ODEServer(FedAvgServer):
    def __init__(self, args = None, trainer_type=ODETrainer, client_type=ODEClient):
        super().__init__(args, trainer_type, client_type)
        self.train_set_len = len(self.trainset)

        # 重新设置数据集
        self.trainset = NeedIndexDataset(self.trainset)
        self.train_sampler = self.trainset.sampler
        self.trainloader = DataLoader(self.trainset, batch_size=self.args["batch_size"],shuffle = False,
                                      pin_memory=True, num_workers=8, persistent_workers=True,
                                      sampler=self.train_sampler, pin_memory_device='cuda:0')
        self.cuda_0_trainer.trainloader = self.trainloader


        # 读取标签分布
        distribution_path = PROJECT_DIR / "data" / self.args["dataset"] / "all_stats.json"
        with open(distribution_path, "rb") as f:
            label_distribution = json.load(f)
        for client_instance in self.client_instances:
            client_instance.label_num_distribution = label_distribution[str(client_instance.client_id)]["distribution"] # {'label':int}
            client_instance.buffer_size = min(client_instance.label_num_distribution)

        self.set_client_label() # 设定client的训练索引对应的标签

        self.clients_per_label = self.client_num * self.args['labels_per_client'] # 每个标签都会分配给所有客户
        self.labels_per_client = self.data_num_classes * self.args['labels_per_client'] # 每个client拥有的类别数

        self.coordinate()
        self.init_client_buffer()
        self.global_gradient:list[torch.tensor] = None
    
    def set_client_label(self):
        for client_instance in self.client_instances:
            label = []
            for index in client_instance.train_set_index:
                data = self.trainset[index]
                assert index == data[0]
                label.append(data[1][1]) # 注意这里的dataset是特别修改版
            client_instance.train_set_label = np.array(label) 

            # check :看数据集加载得一不一致
            assert len(client_instance.label_num_distribution.keys()) == len(np.unique(client_instance.train_set_label))
            for lable_ in client_instance.label_num_distribution.keys():
                assert client_instance.label_num_distribution[lable_] == len(np.where(client_instance.train_set_label == int(lable_))[0])


    def coordinate(self):
        info_classes = {i:[] for i in range(self.data_num_classes)}
        size_classes = {i:0 for i in range(self.data_num_classes)}
        
        for client_instance in self.client_instances:
            for label,value in client_instance.label_num_distribution.items():# 在label_distribution是{str:int}类型，后面需要细致处理一下
                info_classes[int(label)].append((value,client_instance.client_id)) # 对每一个类别，存入(样本数，客户id)
                size_classes[int(label)] += value
        assert sum(size_classes.values()) == self.train_set_len
        sorted_classes = sorted(info_classes.items(),key=lambda i: len(i[1])) # 按照客户数排序，从小到大(即[]中元素的个数排序);返回值是一个列表[(label,[(value,client id)])]
        sorted_classes = [int(i[0]) for i in sorted_classes] #提取出标签
        for _class in sorted_classes:
            info_classes[_class]=sorted(info_classes[_class],key=lambda j :j[0] ,reverse=True)# 按照样本数排序，从大到小
        coordination = {idx: [] for idx in range(self.client_num)} # 每个客户需要缓存的label有哪些
        for _class in sorted_classes:
            cnt = 0 
            for (num,client_id) in info_classes[_class]: # 优先给样本数多的客户
                if num==0:
                    break
                if cnt >= self.clients_per_label:
                    break
                if len(coordination[client_id]) == self.labels_per_client:
                    continue
                cnt+=1
                coordination[client_id].append(_class)
        stored_num = {y: 0 for y in range(self.data_num_classes)}
        real_num = size_classes
        for client_instance in self.client_instances:
            tmp_size = 0
            coordination[client_instance.client_id]=sorted(coordination[client_instance.client_id],key=lambda y:client_instance.label_num_distribution[str(y)], reverse=True)

            for label in coordination[client_instance.client_id]:
                stored_num[label] += int(client_instance.buffer_size / len(coordination[client_instance.client_id]))
                tmp_size += int(client_instance.buffer_size / len(coordination[client_instance.client_id]))

            for label in coordination[client_instance.client_id]:
                if tmp_size == client_instance.buffer_size:
                    break
                else : # 把多余的分给类别数量多的
                    tmp_size+=1
                    stored_num[label]+=1
        stored_tot = sum(stored_num.values())
        real_tot = sum(real_num.values())
        weight = {}
        for label in range(self.data_num_classes):
            if stored_num[label]==0:
                continue
            weight[label] = (real_num[label]/real_tot)/(stored_num[label]/stored_tot)
        for client_instance in self.client_instances:
            client_instance.label_weight = {label:weight[label] for label in coordination[client_instance.client_id] }
            client_instance.weights_tensor = torch.zeros(self.data_num_classes,dtype=torch.float32,device=self.device)
            for label,value in client_instance.label_weight.items():
                client_instance.weights_tensor[label] = value
    
    def init_client_buffer(self):
        self.logger.log("===============start initializing client buffer================")
        for client_instance in tqdm(self.client_instances):
            buffer_per_label = {label:int(client_instance.buffer_size / len(client_instance.label_weight.keys())) for label in client_instance.label_weight.keys()}
            for label in client_instance.label_weight.keys():
                if sum(buffer_per_label.values()) == client_instance.buffer_size:
                    break
                else:
                    buffer_per_label[label] += 1
            client_instance.buffer_size = buffer_per_label
            new_num_train_samples = 0.0
            for label in client_instance.label_weight.keys():
                new_num_train_samples += client_instance.label_weight[label] * client_instance.buffer_size[label]
            client_instance.new_weight4aggregation = new_num_train_samples
        

    def calculate_global_grad(self):
        self.logger.log("===============start calculating global grad================")
        for client_instance in self.client_instances:
            client_instance.model_dict = deepcopy(self.model.state_dict())
        client_local_gradient :list[torch.tensor] = []
        for client_instance in tqdm(self.client_instances):
            client_grad = self.cuda_0_trainer.get_client_grad(client_instance)
            client_local_gradient.append(client_grad)
        self.global_gradient = [1 / self.train_set_len * torch.sum(torch.stack(client_grad,dim=-1),dim=-1) for client_grad in zip(*client_local_gradient)]
    

    def update_client_buffer(self):
        for client_id in tqdm(self.current_selected_client_ids):
            self.client_instances[client_id].model_dict = deepcopy(self.model.state_dict())
            self.cuda_0_trainer.update_client_buffer(self.client_instances[client_id],self.global_gradient,self.args["update_batch_size"])
            # self.logger.log(self.client_instances[client_id].train_set_len,len(self.client_instances[client_id].train_set_index))
            

    def train_one_round(self,global_round):
        self.calculate_global_grad() # 计算全局梯度
        self.update_client_buffer()
        client_model_cache = []  # 缓存梯度
        weight_cache = []  # 缓存梯度对应的权重
        client_training_time = []
        trainer_synchronization = {"round":global_round}
        for client_id in self.current_selected_client_ids:
            assert self.client_instances[client_id].client_id == client_id
            self.client_instances[client_id].model_dict = deepcopy(self.model.state_dict())
        for client_id in self.current_selected_client_ids:
            modified_client_instance = self.cuda_0_trainer.start(
                self.client_instances[client_id],
                self.optimizer.state_dict(),
                trainer_synchronization
            )
            assert modified_client_instance.client_id == client_id
            self.logger.log(
                f"client {client_id} has finished and has participate {modified_client_instance.participation_times}. The local train set size is {modified_client_instance.train_set_len} from {len(modified_client_instance.train_set_index)}. ",
                f"The pretrained acc is {modified_client_instance.pretrained_accuracy:.3f}%. The local accuracy is {modified_client_instance.accuracy:.3f}%.",
                f"The time is {modified_client_instance.training_time}. Scaled time is {round(modified_client_instance.training_time * 10.0)}.")
            self.client_to_server.put(modified_client_instance)
        assert self.client_to_server.qsize() == len(self.current_selected_client_ids)
        while not self.client_to_server.empty():
            modified_client_instance = self.client_to_server.get()
            assert modified_client_instance.client_id in self.current_selected_client_ids
            client_model = {key: value for key, value in modified_client_instance.model_dict.items()}
            client_model_cache.append(client_model)
            weight_cache.append(modified_client_instance.new_weight4aggregation)

            client_training_time.append(round(modified_client_instance.training_time * 10.0))
            self.client_instances[modified_client_instance.client_id] = modified_client_instance  # 更新client信息

        # 聚合并更新参数
        self.aggregate(client_model_cache, weight_cache)  # 聚合梯度
        return max(client_training_time)


if __name__ == "__main__":
    parser = get_argparser().parse_args()
    with open(parser.config_path, 'r') as file:
        args = yaml.safe_load(file)
    if args["set_seed"]:
        fix_random_seed(args["seed"])
    server = ODEServer(args=args)
    server.train()