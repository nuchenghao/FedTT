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
from client.fedavg import FedAvgTrainer, BaseClient
from utls.models import MODEL_DICT
from data.utils.datasets import DATA_NUM_CLASSES_DICT, DATASETS , DATASETS_COLLATE_FN
from utls.dataset import CustomSampler



class FedAvgServer:
    def __init__(
            self,
            args: dict = None,
            trainer_type=FedAvgTrainer,
            client_type=BaseClient
    ):
        self.args = args  # 配置文件
        self.algorithm = args["algorithm"]
        self.current_time = 0  # 全局时间
        with open(PROJECT_DIR / "data" / self.args["dataset"] / "args.json", "r") as f:
            self.args["dataset_args"] = json.load(f)

        # TODO----------------------------------wandb-------------------------
        if self.args["wandb"]:
            log_dir = f"{PROJECT_DIR}/WANDB_LOG_DIR"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            self.experiment = wandb.init(
                project=f"{self.args['project']}",
                config=self.args,
                dir=log_dir,
                reinit=True,
            )
            # the name of the experiment run for Weights & Biases (W&B)
            self.experiment.name = self.args["experiment_name"]
            self.experiment.log({"acc": 0.0}, step=0)
            wandb.run.save()

        # TODO-------------------------------logging------------------------------------
        # TRAIN_LOG = PROJECT_DIR / "trainlog"
        # 在终端和文件中分别输出
        if not os.path.isdir(TRAIN_LOG / self.algorithm) and (
                self.args["save_log"]
        ):
            os.makedirs(TRAIN_LOG / self.algorithm, exist_ok=True)

        stdout = Console(log_path=False, log_time=True)  # 输出时会添加当前的时间戳，但不添加是哪个文件输出
        dataset = self.args["dataset"]
        self.logger = Logger(
            stdout=stdout,
            enable_log=self.args["save_log"],
            logfile_path=TRAIN_LOG / self.algorithm / f"{dataset}_log.log",
        )
        self.logger.log("=" * 20, "ALGORITHM:", self.algorithm, "=" * 20)
        formatted_args = json.dumps(self.args, indent=4)
        self.logger.log("Experiment Arguments:", formatted_args)

        # TODO:--------客户端选择:在有客户端选择的问题中要记得修改--------------------
        # To make sure all algorithms run through the same client sampling stream.
        # Some algorithms' implicit operations at client side may disturb the stream if sampling happens at each FL round's beginning.
        # 它生成了一个列表，包含在每个全球训练轮（global epoch）中随机选择的客户端;也就是说，在全局训练开始前，就确定了客户选择的顺序；
        partition_path = PROJECT_DIR / "data" / self.args["dataset"] / "partition.pkl"
        with open(partition_path, "rb") as f:
            partition = pickle.load(f)
        self.train_client_ids: List[int] = partition["separation"]["train"]  # 参与训练的客户编号
        self.client_num: int = partition["separation"]["total"]
        self.client_sample_stream = [
            random.sample(
                self.train_client_ids, max(1, int(self.client_num * self.args["client_join_ratio"]))
            )
            for _ in range(self.args["global_epoch"])
        ]

        self.current_selected_client_ids: List[int] = []
        self.client_to_server=queue.Queue()

        # TODO ------------init model(s) parameters-------
        torch.cuda.set_device("cuda:0")
        self.device = 'cuda:0'  # 全局模型的device默认为cuda:0
        self.data_num_classes = DATA_NUM_CLASSES_DICT[self.args['dataset']]
        self.model = MODEL_DICT[self.args["model"]](self.data_num_classes).to(self.device)
        # TODO ----------------------------- 数据加载器 --------------------------------
        self.testset = DATASETS[self.args['dataset']](PROJECT_DIR / "data" / args["dataset"], "test")
        self.testloader = DataLoader(Subset(self.testset, list(range(len(self.testset)))), batch_size=2048,
                                     shuffle=False, pin_memory=True, num_workers=8,collate_fn = DATASETS_COLLATE_FN[self.args['dataset']],
                                     persistent_workers=True, pin_memory_device='cuda:0')
        label_count = defaultdict(int)
        for inputs, targets in self.testloader:
            result = Counter(targets.tolist())
            for label, val in result.items():
                label_count[label] += val
        self.logger.log(f"test loader has been prepared in cuda:0 ; {label_count}")

        self.trainset = DATASETS[self.args['dataset']](PROJECT_DIR / "data" / args["dataset"], "train")
        self.train_sampler = CustomSampler(list(range(len(self.trainset))))
        self.trainloader = DataLoader(Subset(self.trainset, list(range(len(self.trainset)))), self.args["batch_size"],
                                      pin_memory=True, num_workers=8,collate_fn = DATASETS_COLLATE_FN[self.args['dataset']], persistent_workers=True,
                                      sampler=self.train_sampler, pin_memory_device='cuda:0')
        label_count = defaultdict(int)
        for inputs, targets in self.trainloader:
            result = Counter(targets.tolist())
            for label, val in result.items():
                label_count[label] += val
        self.logger.log(f"train loader has been prepared in cuda:0 ; {label_count}")

        # TODO------------------------优化器和学习率调整器----------------------------------
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args["lr"],
                                         momentum=self.args["momentum"], weight_decay=self.args["weight_decay"])
        # TODO ----------------- init client-----------------------
        self.data_indices = partition["data_indices"]  # 整个的训练集划分,data_indices的类型为list[np.array]

        self.client_instances: list[BaseClient] = []
        for client_id in self.train_client_ids:
            self.client_instances.append(client_type(client_id, self.data_indices[client_id].tolist(), self.args["batch_size"], ))

        # 这里的model需要深拷贝
        self.cuda_0_trainer = trainer_type(self.device, deepcopy(self.model), self.trainloader, self.testloader,
                                           self.args)

        self.logger.log("cuda:0 has been initialized")

    def train(self):
        for E in range(self.args["global_epoch"]):
            self.logger.log("-" * 30, f"[bold red]TRAINING EPOCH: {E + 1}[/bold red]", "-" * 30)
            self.current_selected_client_ids = self.client_sample_stream[E]
            self.logger.log(f"current selected clients: {self.current_selected_client_ids}")
            training_time = self.train_one_round( E + 1 )
            self.current_time += training_time
            accuracy = evaluate(torch.device("cuda:0"), self.model, self.testloader)
            self.logger.log(f"Finished training!!! Current global epoch training time: {training_time}.",
                            f"The global time is {self.current_time}",
                            f"The Global model accuracy is {accuracy:.3f}%.")
            if self.args["wandb"]:
                self.experiment.log({"acc": accuracy}, step=self.current_time)
        for client_instance in self.client_instances:
            self.logger.log(f"Client{client_instance.client_id}'s training time : {client_instance.training_time_record}")

    def train_one_round(self,global_round):
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
                f"client {client_id} has finished and has participate {modified_client_instance.participation_times}. The local train set size is {modified_client_instance.train_set_len}. ",
                f"The pretrained acc is {modified_client_instance.pretrained_accuracy:.3f}%. The local accuracy is {modified_client_instance.accuracy:.3f}%.",
                f"The time is {modified_client_instance.training_time}. Scaled time is {round(modified_client_instance.training_time * 10.0)}.")
            self.client_to_server.put(modified_client_instance)
        assert self.client_to_server.qsize() == len(self.current_selected_client_ids)
        while not self.client_to_server.empty():
            modified_client_instance = self.client_to_server.get()
            assert modified_client_instance.client_id in self.current_selected_client_ids
            client_model = {key: value for key, value in modified_client_instance.model_dict.items()}
            client_model_cache.append(client_model)
            weight_cache.append(modified_client_instance.train_set_len)
            client_training_time.append(round(modified_client_instance.training_time * 10.0))
            self.client_instances[modified_client_instance.client_id] = modified_client_instance  # 更新client信息

        # 聚合并更新参数
        self.aggregate(client_model_cache, weight_cache)  # 聚合梯度
        return max(client_training_time)

    def aggregate(
            self,
            client_model_cache,
            weight_cache,
    ):
        with torch.no_grad():
            weights = torch.tensor(weight_cache, device=self.device) / sum(weight_cache)
            model_list = [list(delta.values()) for delta in client_model_cache]
            aggregated_model = [
                torch.sum(weights * torch.stack(grad, dim=-1), dim=-1)
                for grad in zip(*model_list)
            ]
            averaged_state_dict = OrderedDict(zip(client_model_cache[0].keys(), aggregated_model))
            self.model.load_state_dict(averaged_state_dict)


if __name__ == "__main__":
    parser = get_argparser().parse_args()
    with open(parser.config_path, 'r') as file:
        args = yaml.safe_load(file)
    if args["set_seed"]:
        fix_random_seed(args["seed"])
    server = FedAvgServer(args=args)
    server.train()
