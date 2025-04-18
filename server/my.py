from copy import deepcopy
from pathlib import Path
import sys
import yaml
from fedavg import FedAvgServer
from utls.utils import get_argparser, fix_random_seed
from client.my import myFed
from rich.console import Console
import os
from utls.dataset import NeedIndexDataset
from torch.utils.data import DataLoader
from data.utils.datasets import DATASETS_COLLATE_FN
from utls.dataset import NeedIndexDataset

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # debug

PROJECT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(PROJECT_DIR.as_posix())
sys.path.append(PROJECT_DIR.joinpath("src").as_posix())



class myFedServer(FedAvgServer):
    def __init__(self, args):
        super().__init__(args=args, trainer_type=myFed)
        self.current_global_epoch = 0
        self.need_to_keep = int(self.args['global_epoch'] * self.args['start'])


    def train_one_round(self,global_round):
        client_model_cache = []  # 缓存梯度
        weight_cache = []  # 缓存梯度对应的权重
        client_training_time = []
        trainer_synchronization = {"round":global_round,'prune': False,"accuracy":self.accuracy / 100.}
        if self.current_global_epoch >= self.need_to_keep:
            trainer_synchronization['prune'] = True
        for client_id in self.current_selected_client_ids:
            assert self.client_instances[client_id].client_id == client_id
            self.client_instances[client_id].model_dict = self.model.state_dict()
        for client_id in self.current_selected_client_ids:
            modified_client_instance = self.cuda_0_trainer.start(
                self.client_instances[client_id],
                self.optimizer.state_dict(),
                trainer_synchronization
            )
            assert modified_client_instance.client_id == client_id
            self.logger.log(
                f"client {client_id} has finished and has participate {modified_client_instance.participation_times}. The local train set size is {modified_client_instance.train_set_len}. ",
                f"The pretrained acc is {modified_client_instance.pretrained_accuracy:.3f}%.The local accuracy is {modified_client_instance.accuracy:.3f}%.",
                f"The time is {modified_client_instance.training_time}. Scaled time is {round(modified_client_instance.training_time * 10.0)}.")
            self.client_to_server.put(modified_client_instance)
        assert self.client_to_server.qsize() == len(self.current_selected_client_ids)
        while not self.client_to_server.empty():
            modified_client_instance = self.client_to_server.get()
            assert modified_client_instance.client_id in self.current_selected_client_ids
            client_model = {key: value for key, value in modified_client_instance.model_dict.items()}
            del modified_client_instance.model_dict
            client_model_cache.append(client_model)
            weight_cache.append(modified_client_instance.train_set_len)
            client_training_time.append(round(modified_client_instance.training_time * 10.0))
            self.client_instances[modified_client_instance.client_id] = modified_client_instance  # 更新client信息
        # 聚合并更新参数
        self.aggregate(client_model_cache, weight_cache)  # 聚合梯度
        self.current_global_epoch += 1
        return max(client_training_time)


if __name__ == '__main__':
    parser = get_argparser().parse_args()
    with open(parser.config_path, 'r') as file:
        args = yaml.safe_load(file)
    if args["set_seed"]:
        fix_random_seed(args["seed"])
    server = myFedServer(args=args)
    server.train()
