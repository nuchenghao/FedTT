import json
import os
import pickle
import sys
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import yaml
CURRENT_DIR = Path(__file__).parent.absolute()
FL_BENCH_ROOT = CURRENT_DIR.parent
sys.path.append(FL_BENCH_ROOT.as_posix())
from data.draw_data_distribution import draw_data_distribution
from utls.utils import get_argparser, fix_random_seed
from data.utils.datasets import DATASETS, DATA_NUM_CLASSES_DICT
from data.utils.schemes import (
    dirichlet,
    iid_partition,
    randomly_assign_classes,
    allocate_shards,
    semantic_partition,
)
from data.utils.process import (
    prune_args,
    generate_synthetic_data,
    process_celeba,
    process_femnist,
)
def main(args: dict, which: str):
    dataset_root = CURRENT_DIR / args["dataset"]
    if which == 'test':
        dataset = DATASETS[args["dataset"]](dataset_root, which)
        return
    if not os.path.isdir(dataset_root):
        os.mkdir(dataset_root)
    partition = {"separation": None, "data_indices": None}
    if args["dataset"] == "femnist":
        partition, stats, args.client_num = process_femnist()
    else:
        dataset = DATASETS[args["dataset"]](dataset_root, which)
        if not args["iid"]:
            if args["alpha"] > 0:
                partition, stats = dirichlet(
                    dataset=dataset,
                    client_num=args["client_num"],
                    alpha=args["alpha"],
                    least_samples=args["least_samples"],
                )
        else:
            partition, stats = iid_partition(
                dataset=dataset, client_num=args["client_num"]
            )
    partition["separation"] = {
        "train": list(range(args["client_num"])),
        "total": args["client_num"],
    }
    with open(dataset_root / "partition.pkl", "wb") as f:
        pickle.dump(partition, f)
    with open(dataset_root / "all_stats.json", "w") as f:
        json.dump(stats, f)
    with open(dataset_root / "args.json", "w") as f:
        json.dump(prune_args(args), f)
def generate_data(args: dict):
    main(args, "train")
    main(args, "test")
if __name__ == "__main__":
    parser = get_argparser().parse_args()
    with open(parser.config_path, 'r') as file:
        args = yaml.safe_load(file)
    fix_random_seed(args["seed"])
    generate_data(args)
