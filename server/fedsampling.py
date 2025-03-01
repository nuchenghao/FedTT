from copy import deepcopy
from pathlib import Path
import sys
import yaml
from fedavg import FedAvgServer
from utls.utils import get_argparser, fix_random_seed
from client.fedsampling import fedsamplingClient,fedsamplingTrainer
from rich.console import Console
import torch
import math
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # debug

PROJECT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(PROJECT_DIR.as_posix())
sys.path.append(PROJECT_DIR.joinpath("src").as_posix())



def calculate_alpha(M: int , epsilon: float = 3) -> float:
    exp_epsilon = math.exp(epsilon)  # 计算 e^ε
    numerator = exp_epsilon - 1      # 分子
    denominator = exp_epsilon + M - 2  # 分母

    return numerator / denominator


class fedsampling(FedAvgServer):
    def __init__(self, args, trainer_type=fedsamplingTrainer, client_type=fedsamplingClient):
        super().__init__(args, trainer_type, client_type)

        
        self.M = max(len(client_data_indices) for client_data_indices in self.data_indices) + 1
        self.K = sum([len(client_data_indices) for client_data_indices in self.data_indices]) * self.args['KN']
        self.alpha = 0.5
        print(f"the value of self.M is {self.M} ; the value of self.alpha is {self.alpha} ; the value of self.K is {self.K}")
        for client_instance in self.client_instances:
            assert type(client_instance) == fedsamplingClient
            client_instance.set_estimator(self.M , self.alpha) # 设置为最大值+1，因为fake无法取到上界
    
    def train_one_round(self,global_round):
        client_grad = []
        client_buffer = []
        client_training_time = []
        response = []

        # for client_id in range(self.client_num):
        #     response.append(self.client_instances[client_id].estimator.query())
        # R = sum(response)
        # hat_N = max(self.client_num,(R-self.client_num*(1-self.alpha)*self.M/2)/self.alpha)*self.args['client_join_ratio']


        hat_N = 0 
        for client_id in self.current_selected_client_ids:
            hat_N += self.client_instances[client_id].train_set_len
        self.K = hat_N * self.args["KN"]

        trainer_synchronization = {"round" : global_round ,'KN': self.K / hat_N , "K" : self.K}
        print(f" K:{self.K} N_hat:{hat_N} the ratio is {self.K / hat_N}")
        for client_id in self.current_selected_client_ids:
            self.client_instances[client_id].model_dict = self.model.state_dict()
        for client_id in self.current_selected_client_ids:
            modified_client_instance = self.cuda_0_trainer.start(
                self.client_instances[client_id],
                self.optimizer.state_dict(),
                trainer_synchronization
            )
            assert modified_client_instance.client_id == client_id
            self.logger.log(
                f"client {client_id} has finished and has participate {modified_client_instance.participation_times}. The local train set size is {modified_client_instance.train_set_len} and used {modified_client_instance.selected_data_num}.",
                f"The pretrained acc is {modified_client_instance.pretrained_accuracy:.3f}%.The local accuracy is {modified_client_instance.accuracy:.3f}%.",
                f"The time is {modified_client_instance.training_time}. Scaled time is {round(modified_client_instance.training_time * 10.0)}.")
            self.client_to_server.put(modified_client_instance)
        assert self.client_to_server.qsize() == len(self.current_selected_client_ids)
        while not self.client_to_server.empty():
            modified_client_instance = self.client_to_server.get()
            assert modified_client_instance.client_id in self.current_selected_client_ids
            client_grad.append(modified_client_instance.grad) # 每个元素都是一个dict
            client_buffer.append(modified_client_instance.buffer) # 每个元素都是一个dict
            del modified_client_instance.grad,modified_client_instance.buffer
            client_training_time.append(round(modified_client_instance.training_time * 10.0))
            self.client_instances[modified_client_instance.client_id] = modified_client_instance  # 更新client信息
        # 聚合并更新参数
        self.optimizer.zero_grad()
        for name , param in self.model.named_parameters():
            cache = []
            for grad in client_grad:
                cache.append(grad[name])
            agg_grad = (1 / self.K) * torch.sum(torch.stack(cache , dim=-1) , dim=-1)
            param.data -= agg_grad
        for name , param in self.model.named_buffers():
            cache = []
            for buffer in client_buffer:
                cache.append(buffer[name])
            agg_buffer =(1 / self.K) * torch.sum(torch.stack(cache,dim=-1) , dim=-1)
            param.data = agg_buffer
        
        return max(client_training_time)


if __name__ == '__main__':
    parser = get_argparser().parse_args()
    with open(parser.config_path, 'r') as file:
        args = yaml.safe_load(file)
    if args["set_seed"]:
        fix_random_seed(args["seed"])
    server = fedsampling(args=args)
    server.train()
