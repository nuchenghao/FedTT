from copy import deepcopy
import torch
import queue
import json
from py3nvml import py3nvml as nvml
from fedavg import FedAvgTrainer,BaseClient
from utls.utils import NN_state_load, evaluate
import numpy as np
from collections import OrderedDict
class Estimator:
    def __init__(self,M,alpha,dataset_len):
        self.M = M # size threshold；为了便于实现，这里设置为所有客户数据量的最大值+1
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
        
    def set_estimator(self , M):
        self.estimator = Estimator(M,0.5,self.train_set_len)




class fedsamplingTrainer(FedAvgTrainer):
    def __init__(self, device, model, trainloader, testloader, args):
        super().__init__(device, model, trainloader, testloader, args)  
        self.selected_data_num = 0
    
    def load_dataset(self):
        # 对于每个 candidate_sample，生成一个独立的随机结果，表示在一次试验中成功的概率为 KN
        choosed = np.random.binomial(size=(self.current_client.train_set_len,), n=1, p=self.synchronization['KN'])
        candidate_set_index=np.array(self.current_client.train_set_index)
        participate_set_index = candidate_set_index[choosed == 1].tolist()
        self.current_client.selected_data_num=len(participate_set_index)
        self.trainloader.sampler.set_index(participate_set_index)  # 在里面实现了深拷贝
        self.trainloader.batch_sampler.batch_size = self.current_client.batch_size
    
    def start(self,
              client,
              optimizer_state_dict,
              trainer_synchronization
              ):
        self.timer.start()
        self.current_client = client
        self.set_parameters(optimizer_state_dict, trainer_synchronization)  # 设置参数
        self.load_dataset()

        if self.args['client_eval']:
            self.current_client.pretrained_accuracy = evaluate(self.device, self.model, self.testloader)
        else:
            self.current_client.pretrained_accuracy = 0.0
        
        self.local_train()

        if self.args['client_eval']:
            self.current_client.accuracy = evaluate(self.device, self.model, self.testloader)
        else:
            self.current_client.accuracy = 0.0
        self.model = self.model.to("cpu")  # 训练&验证 结束后将模型转移到cpu上
        delta = OrderedDict()
        for name , param in self.model.named_parameters():
            delta[name] = self.current_client.selected_data_num * (self.current_client.model_dict[name] - param.data)
        buffer = OrderedDict()
        for name , param in self.model.named_buffers():
            buffer[name] = self.current_client.selected_data_num * deepcopy(param.data)
        self.timer.stop() # 里面的一些操作带来的开销就权当是网络传输的时间了
        self.current_client.training_time = self.timer.times[-1]
        self.current_client.participate_once()
        self.current_client.training_time_record[self.synchronization['round']] = round(self.current_client.training_time * 10.0) # 记录时间
        self.current_client.grad = delta
        self.current_client.buffer = buffer
        torch.cuda.empty_cache() # 释放缓存 
        return self.current_client