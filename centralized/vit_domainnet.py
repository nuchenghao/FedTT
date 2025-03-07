import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
import sys
import yaml
import os
import wandb
PROJECT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(PROJECT_DIR.as_posix())
sys.path.append(PROJECT_DIR.joinpath("src").as_posix())
from utls.utils import (
    fix_random_seed,
    get_argparser,
)
from utls.utils import Timer
from utls.models import MODEL_DICT
from data.utils.datasets import DATA_NUM_CLASSES_DICT, DATASETS

if __name__=='__main__':
    parser = get_argparser().parse_args()
    with open(parser.config_path, 'r') as file:
        args = yaml.safe_load(file)
    if args["set_seed"]:
        fix_random_seed(args["seed"])
    if args["wandb"]:
        log_dir = f"{PROJECT_DIR}/WANDB_LOG_DIR"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        experiment = wandb.init(
            project=f"{args['project']}",
            config=args,
            dir=log_dir,
            reinit=True,
        )
        # the name of the experiment run for Weights & Biases (W&B)
        experiment.name = args["experiment_name"]
        experiment.log({"acc": 0.0}, step=0)
        wandb.run.save()
    trainset = DATASETS[args['dataset']](PROJECT_DIR / "data" / args["dataset"], "train")
    trainloader = DataLoader(trainset,
                         batch_size=args["batch_size"],
                         shuffle=True,
                         pin_memory=True,
                         num_workers=8,
                         persistent_workers=True,
                         pin_memory_device='cuda:0')
    testset=DATASETS[args['dataset']](PROJECT_DIR / "data" / args["dataset"], "test")
    testloader = DataLoader(testset, batch_size=512, shuffle=False, pin_memory=True, num_workers=8,
                        persistent_workers=True, pin_memory_device='cuda:0')
    model = MODEL_DICT[args["model"]](DATA_NUM_CLASSES_DICT[args['dataset']]).to('cuda:0')
    for name, param in model.named_parameters():
        if 'head' in name or 'lora' in name or 'Prompt' in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=args["lr"],
                                         momentum=args["momentum"], weight_decay=args["weight_decay"])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none').to('cuda:0')

    def test(epoch):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to('cuda:0'), targets.to("cuda:0")
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        cur_acc = round(100. * correct / total, 3)
        print(f"epoch {epoch} : test acc is {cur_acc}")
        return cur_acc

    timer=Timer()
    for epoch in tqdm(range(args['global_epoch'])):
        model.train()
        timer.start()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = torch.mean(criterion(outputs, targets))
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        timer.stop()
        acc=test(epoch)
        experiment.log({"acc": acc}, step=int(timer.sum()))
