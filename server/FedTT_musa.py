from copy import deepcopy
from pathlib import Path
import sys
import yaml
from fedavg_musa import FedAvgServer
from utls.utils import get_argparser, fix_random_seed_musa
from client.FedTT_musa import FedTTClient
from rich.console import Console
import os
from utls.dataset import NeedIndexDataset
from torch.utils.data import DataLoader
from data.utils.datasets import DATASETS_COLLATE_FN
from utls.dataset import NeedIndexDataset
PROJECT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(PROJECT_DIR.as_posix())
sys.path.append(PROJECT_DIR.joinpath("src").as_posix())


class FedTTServer(FedAvgServer):
    def __init__(self, args):
        super().__init__(args=args, trainer_type=FedTTClient)
        self.current_global_epoch = 0

    def train_one_round(self,global_round):
        client_model_cache = []
        weight_cache = []
        client_training_time = []
        trainer_synchronization = {"round":global_round, 'prune': True,"accuracy": self.accuracy / 100.}

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
            self.client_instances[modified_client_instance.client_id] = modified_client_instance
        self.aggregate(client_model_cache, weight_cache)
        self.current_global_epoch += 1
        return max(client_training_time)
    
if __name__ == '__main__':
    parser = get_argparser().parse_args()
    with open(parser.config_path, 'r') as file:
        args = yaml.safe_load(file)
    if args["set_seed"]:
        fix_random_seed_musa(args["seed"])
    server = FedTTServer(args=args)
    server.train()
