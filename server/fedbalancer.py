import pickle
import sys
import json
import os
import numpy as np
from pathlib import Path
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List
import wandb
import torch
import yaml
from rich.console import Console
from torch.utils.data import DataLoader
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
from client.fedavg import FedAvgTrainer, BaseClient
from server.fedavg import FedAvgServer
from client.fedbalancer import fedbalancerClient,fedbalancerTrainer
from utls.models import MODEL_DICT
from data.utils.datasets import DATA_NUM_CLASSES_DICT, DATASETS
from utls.dataset import NeedIndexDataset

class fedbalancer(FedAvgServer):
    def __init__(self, args = None, trainer_type=fedbalancerTrainer, client_type=fedbalancerClient):
        super().__init__(args, trainer_type, client_type)
        # -------------------- 设置batch的训练时间 --------------------------------
        self.batch_training_time = self.batch_training_time()
        for client_instance in self.client_instances:
            client_instance.batch_training_time = self.batch_training_time
        self.logger.log(f"========= batch_training_time: {self.batch_training_time}===========")

        # -------------------- 设置训练集和数据加载器--------------------------
        self.trainset = NeedIndexDataset(self.trainset)
        self.train_sampler = self.trainset.sampler
        self.trainloader = DataLoader(self.trainset, batch_size=self.args["batch_size"],shuffle = False,
                                      pin_memory=True, num_workers=8, persistent_workers=True,
                                      sampler=self.train_sampler, pin_memory_device='cuda:0')
        self.cuda_0_trainer.trainloader = self.trainloader

        self.current_global_epoch = 0 # 已完成的次数

        self.ltr = 0.0
        self.ddlr = 1.0 
        self.ddl_R = 0.0
        self.lt = 0.0
        self.U = []
        self.w = 10
        self.lss =0.05
        self.dss = 0.05

    
    def batch_training_time(self):
        model = MODEL_DICT[self.args["model"]](DATA_NUM_CLASSES_DICT[self.args['dataset']]).to(self.device)
        model.train()
        criterion = torch.nn.CrossEntropyLoss(label_smoothing = 0.1).to(self.device)  # label_smoothing的默认值为0.1
        optimizer = torch.optim.SGD(model.parameters(),
                          lr=0.1,
                          momentum=0.0,
                          weight_decay=0.0)
        start=torch.cuda.Event(enable_timing=True)
        end=torch.cuda.Event(enable_timing=True)
        train_time = []
        for _ in range(5):
            start.record()
            for inputs, targets in self.trainloader:
                inputs, targets = inputs.to(self.device,non_blocking=True), targets.to(self.device,non_blocking=True)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            end.record()
            torch.cuda.synchronize()
            train_time.append(start.elapsed_time(end)/len(self.trainloader)/1000.0)# 每个batch的时间，单位为s
        return sum(train_time[1:]) / len(train_time[1:])

    def lt_selection_next_round(self,LLow,LHigh):
        #  𝑙𝑡 selection for next (𝑅 + 1)-th round  (Algorithm 2)
        ll = min(LLow)
        lh = sum(LHigh) / len(LHigh)
        self.lt = ll + (lh - ll) * self.ltr

    def ltr_ddlr_control(self,LSum_R,L_R):
        self.U.append( LSum_R / (L_R * self.ddl_R))
        if self.current_global_epoch % self.w == 0 :
            if len(self.U) >= 2 * self.w and sum(self.U[self.current_global_epoch - 2 * self.w:self.current_global_epoch - self.w]) > sum(self.U[self.current_global_epoch - self.w:]):
                self.ltr = min(self.ltr + self.lss , 1.0)
                self.ddlr = max(self.ddlr - self.dss,0.0) 
            else:
                self.ltr = max(self.ltr - self.lss , 0.0)
                self.ddlr = min(self.ddlr + self.dss , 1.0)
        
    def select_deadline(self,E):
        def find_peak_ddl_E(epoch):
            completeTime = []
            ddl_E = []
            t = self.batch_training_time
            for client_id in self.current_selected_client_ids:
                completeTime.append(
                    (self.client_instances[client_id].len_OT + self.client_instances[client_id].batch_size - 1 ) / self.client_instances[client_id].batch_size * self.client_instances[client_id].batch_training_time * epoch 
                )
            completeTime =  np.array(completeTime)
            count = 0
            while count != len(self.current_selected_client_ids):
                count = np.sum(completeTime < t)
                ddl_E.append((count / t , t))
                t += self.batch_training_time
            return max(ddl_E, key=lambda x: (x[0], x[1]))[1]
        dl = find_peak_ddl_E(1)
        dh = find_peak_ddl_E(E)
        self.ddl_R = dl + (dh - dl) * self.ddlr

            

    def train_one_round(self,global_round):
        client_model_cache = []  # 缓存梯度
        weight_cache = []  # 缓存梯度对应的权重
        client_training_time = []
        LLow = []
        LHigh = []
        Lsum = []
        self.select_deadline(self.args['local_epoch'])
        trainer_synchronization = {"round":global_round,"deadline" : self.ddl_R , "loss_threshold" : self.lt}
        self.logger.log(trainer_synchronization,f"ddlr : {self.ddlr} ; ltr : {self.ltr}")
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
                f"client {client_id} has finished and has participate {modified_client_instance.participation_times}. The local train set size is {modified_client_instance.train_set_len} and used {len(modified_client_instance.selected_data_index)}. ",
                f"The pretrained acc is {modified_client_instance.pretrained_accuracy:.3f}%. The local accuracy is {modified_client_instance.accuracy:.3f}%.",
                f"The time is {modified_client_instance.training_time}. Scaled time is {round(modified_client_instance.training_time * 10.0)}.")
            self.client_to_server.put(modified_client_instance)
        assert self.client_to_server.qsize() == len(self.current_selected_client_ids)
        while not self.client_to_server.empty():
            modified_client_instance = self.client_to_server.get()
            assert modified_client_instance.client_id in self.current_selected_client_ids
            client_model = {key: value for key, value in modified_client_instance.model_dict.items()}
            client_model_cache.append(client_model)
            weight_cache.append(len(modified_client_instance.selected_data_index))
            LLow.append(modified_client_instance.metadata['llow'])
            LHigh.append(modified_client_instance.metadata['lhigh'])
            Lsum.append(modified_client_instance.metadata['lsum'])
            client_training_time.append(round(modified_client_instance.training_time * 10.0))
            self.client_instances[modified_client_instance.client_id] = modified_client_instance  # 更新client信息
        L_R = sum(weight_cache)
        # 聚合并更新参数
        self.current_global_epoch += 1 
        self.aggregate(client_model_cache, weight_cache)  # 聚合梯度
        self.lt_selection_next_round(LLow,LHigh)
        self.ltr_ddlr_control(sum(Lsum),L_R)
        
        
        return max(client_training_time)



if __name__ == "__main__":
    parser = get_argparser().parse_args()
    with open(parser.config_path, 'r') as file:
        args = yaml.safe_load(file)
    if args["set_seed"]:
        fix_random_seed(args["seed"])
    server = fedbalancer(args=args)
    server.train()