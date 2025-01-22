import pickle
import sys
import json
import os
import random
from pathlib import Path
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List
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
from utls.dataset import CustomSampler


class ODEServer(FedAvgServer):
    def __init__(self, args = None, trainer_type=ODETrainer, client_type=ODEClient):
        super().__init__(args, trainer_type, client_type)

        distribution_path = PROJECT_DIR / "data" / self.args["dataset"] / "all_stats.json"
        with open(distribution_path, "rb") as f:
            label_distribution = pickle.load(f)
        for client_instance in self.client_instances:
            client_instance.label_distribution = label_distribution[str(client_instance.client_id)]["distribution"] # {'label':int}
            client_instance.buffer_size = client_instance.train_set_len * self.args['buffer_ratio']


        self.clients_per_label = self.client_num # 每个标签都会分配给所有客户
        self.labels_per_client = self.data_num_classes # 每个client拥有的类别数

        self.coordinate()
    
    def coordinate(self):
        info_classes = {i:[] for i in range(self.data_num_classes)}
        size_classes = {i:0 for i in range(self.data_num_classes)}
        
        for client_instance in self.client_instances:
            for label,value in client_instance.label_distribution:
                info_classes[label].append((value,client_instance.client_id)) # 对每一个类别，存入(样本数，客户id)
                size_classes[label] += value
        sorted_classes = sorted(info_classes.items(),key=lambda i: len(i[1])) # 按照客户数排序，从小到大(即[]中元素的个数排序);返回值是一个列表[(label)]
        sorted_classes = [i[0] for i in sorted_classes]
        for _class in sorted_classes:
            info_classes[_class]=sorted(info_classes[_class],key=lambda j :j[0] ,reverse=True)# 按照样本数排序，从大到小
        coordination = {idx: [] for idx in range(self.client_num)}
        for _class in sorted_classes:
            cnt = 0 
            for (num,client_id) in info_classes[_class]:
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
            coordination[client_instance.client_id]=sorted(coordination[client_instance.client_id],key=lambda y:client_instance.label_distribution[y], reverse=True)

            for label in coordination[client_instance.client_id]:
                stored_num[label] += int(client_instance.buffer_size / len(coordination[client_instance.client_id]))
                tmp_size += int(client_instance.buffer_size / len(coordination[client_instance.client_id]))

            for label in coordination[client_instance.client_id]:
                if tmp_size == client_instance.buffer_size:
                    break
                else :
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
            client_instance.distributed_labels = {label:weight[label] for label in coordination[client_instance.client_id] }
        
        for client_instance in self.client_instances: 
            print(client_instance.client_id,client_instance.distributed_labels)



if __name__ == "__main__":
    parser = get_argparser().parse_args()
    with open(parser.config_path, 'r') as file:
        args = yaml.safe_load(file)
    if args["set_seed"]:
        fix_random_seed(args["seed"])
    server = ODEServer(args=args)
    # server.train()