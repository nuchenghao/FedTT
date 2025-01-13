from copy import deepcopy
import torch
import queue
import threading
from py3nvml import py3nvml as nvml
from fedavg import FedAvgTrainer,BaseClient
from utls.utils import NN_state_load, evaluate
import numpy as np

class Estimator:
    def __init__(self,alpha,dataset_len):
        self.M = dataset_len # size threshold
        self.alpha = alpha # 公式(1)中的alpha
        self.real_response = dataset_len
    
    def query(self):
        fake_response = np.random.randint(1,self.M) #也就是随机取
        choice = np.random.binomial(n=1,p=self.alpha)# n=1 意味着每次只进行一次试验;p=self.alpha: 这是成功的概率
        # 函数返回一个整数，表示在 n 次试验中成功的次数。由于 n=1，返回值将是 0 或 1;0：表示失败。
        response = choice*self.real_response + (1-choice)*fake_response
        return response


class fedsamplingClient(BaseClient):
    def __init__(self, client_id, train_index, batch_size):
        super().__init__(client_id, train_index, batch_size)
        self.estimator = Estimator(0.5,self.train_set_len)



class fedsamplingTrainer(FedAvgTrainer):
    def __init__(self, device, model, trainloader, testloader, args):
        super().__init__(device, model, trainloader, testloader, args)
    
    def load_dataset(self):
        # 对于每个 candidate_sample，生成一个独立的随机结果，表示在一次试验中成功的概率为 KN
        choosed = np.random.binomial(size=(self.current_client.train_set_len,), n=1, p=self.synchronization['KN'])
        candidate_set_index=np.array(self.current_client.train_set_index)
        participate_set_index=candidate_set_index[choosed==1].tolist()
        self.trainloader.sampler.set_index(participate_set_index)  # 在里面实现了深拷贝
        self.trainloader.batch_sampler.batch_size = self.current_client.batch_size
