import argparse
import pickle
import sys
import yaml

import numpy as np
from sklearn.model_selection import train_test_split
import wandb

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from helpers import train_loop_dist

sys.path.append("../")
from models.Conditionners import StrAFConditioner
from models.Normalizers import MonotonicNormalizer
from models.NormalizingFlow import NormalizingFlowStep, FixedFCNormalizingFlow
from models.NormalizingFlowFactories import NormalLogDensity

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument("--random_seed", type=int, default=2547)
parser.add_argument("--full_ar", type=eval, required=True)
parser.add_argument("--config_name", type=str, required=True)
parser.add_argument("--n_steps", type=int)
parser.add_argument("--cond_hidden", type=eval)
parser.add_argument("--norm_hidden", type=eval)
parser.add_argument("--width", type=int)
parser.add_argument("--lr", type=float)
parser.add_argument("--dist_url", type=str)
args = parser.parse_args()


def load_config(args):
    with open("./train_args.yaml", "r") as f:
        train_args = yaml.safe_load(f)
        train_args = train_args[args.config_name]

    if args.lr:
        train_args["lr"] = args.lr

    with open("./model_args.yaml", "r") as f:
        model_args = yaml.safe_load(f)
        model_args = model_args[args.config_name]
    
    if args.cond_hidden:
        model_args["conditioner"]["hidden"] = args.cond_hidden
    if args.norm_hidden:
        model_args["normalizer"]["integrand_net"] = args.norm_hidden
    if args.width:
        model_args["normalizer"]["cond_size"] = args.width
        model_args["conditioner"]["out_dim"] = args.width
    if args.n_steps:
        model_args["n_steps"] = args.n_steps
    
    return model_args, train_args


def load_data(full_ar):
    with open("./data/gefcom2017_processed.pkl", "rb") as f:
        dataset = pickle.load(f)

    df = dataset["df"]
    if full_ar:
        n_col = len(df.columns)
        adj_mat = np.ones((n_col, n_col))
    else:
        adj_mat = dataset["adj_mat"]

    return df, adj_mat


def get_loaders(df, batch_size, device):
    train_df, rest_df = train_test_split(df, test_size=0.3)
    val_df, _ = train_test_split(rest_df, test_size=0.5)

    train_data = torch.Tensor(train_df.to_numpy()).to(device)
    val_data = torch.Tensor(val_df.to_numpy()).to(device)

    t_sampler = DistributedSampler(train_data)
    v_sampler = DistributedSampler(val_data)

    train_dl = DataLoader(train_data, batch_size=batch_size, sampler=t_sampler)
    val_dl = DataLoader(val_data, batch_size=batch_size, sampler=v_sampler)

    return train_dl, val_dl


def build_perm_nf(adj_mat, args):
    flow_steps = []

    for _ in range(args["n_steps"]):
        step_adj = np.tril(adj_mat, -1)

        conditioner = StrAFConditioner(
            args["dim"], 
            args["conditioner"]["hidden"],
            args["conditioner"]["out_dim"],
            args["conditioner"]["opt_type"],
            step_adj
        )

        normalizer = MonotonicNormalizer(
            args["normalizer"]["integrand_net"],
            args["normalizer"]["cond_size"],
            args["normalizer"]["nb_steps"],
        )
        
        if args["permute"]:
            adj_mat = adj_mat.T

        flow_step = NormalizingFlowStep(conditioner, normalizer)
        flow_steps.append(flow_step)

    flow = FixedFCNormalizingFlow(flow_steps,
                                  NormalLogDensity(),
                                  args["permute"])

    return flow


def main_loop(gpu, n_gpus, wandb_run, dist_url, model_args, train_args):
    dist.init_process_group("nccl", dist_url, rank=gpu, world_size=n_gpus)
    torch.cuda.set_device(gpu)
    torch.cuda.empty_cache()

    df, adj_mat = load_data(args.full_ar)
    model_args["dim"] = len(df.columns)

    train_dl, val_dl = get_loaders(df, train_args["batch_size"], gpu)

    flow = build_perm_nf(adj_mat, model_args).to(gpu)
    flow = nn.parallel.DistributedDataParallel(flow, device_ids=[gpu])
    opt = optim.Adam(flow.parameters(), lr=train_args["lr"])

    if "scheduler" in train_args and train_args["scheduler"] == "ReduceLROnPlateau":
        sched = ReduceLROnPlateau(opt, patience=train_args["scheduler_patience"])
    else:
        sched = None

    best_model = train_loop_dist(gpu, n_gpus, wandb_run, flow, opt, sched,
                                 train_dl, val_dl, train_args)

    return best_model


if __name__ == "__main__":
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    model_args, train_args = load_config(args)

    config = vars(args)
    config.update(model_args)
    config.update(train_args)
    print(config, flush=True)

    run = wandb.init(
        project="gefcom-straf",
        dir="./",
        tags=[],
        config=config,
    )

    n_gpus = torch.cuda.device_count()
    mp.spawn(main_loop, args=(n_gpus, run, args.dist_url, model_args, train_args),
             nprocs=n_gpus)

    run.finish()
