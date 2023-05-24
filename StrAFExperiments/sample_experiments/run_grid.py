import argparse
from datetime import datetime
import pathlib
import sys
import os

import numpy as np
import yaml

import torch
import wandb

from sample_exp_utils import generate_multimodal_data, load_model, train_loop

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=2547)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train_config", type=str, required=True)

    parser.add_argument("--cond_width", type=int, required=True)
    parser.add_argument("--cond_depth", type=int, required=True)
    parser.add_argument("--nf_step", type=int, required=True)
    parser.add_argument("--inet_width", type=int, required=True)
    parser.add_argument("--inet_depth", type=int, required=True)
    parser.add_argument("--h_size", type=int, required=True)

    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--scheduler", type=str, required=True)

    args = parser.parse_args()
    with open("../data_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        data_args = config[args.dataset]

    with open("../model_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        model_args = config[args.model_name]

    cond_net = [args.cond_width] * args.cond_depth
    model_args["conditioner_args"]["hidden"] = cond_net
    model_args["conditioner_args"]["out_size"] = args.h_size

    inet = [args.inet_width] * args.inet_depth
    model_args["normalizer_args"]["integrand_net"] = inet
    model_args["normalizer_args"]["cond_size"] = args.h_size
    model_args["nf_steps"] = args.nf_step

    with open("./train_args.yaml", "r") as f:
        config = yaml.safe_load(f)
        train_args = config[args.train_config]

    train_args["lr"] = args.lr
    if args.scheduler == "MultiStep":
        train_args["scheduler"] = "MultiStep"
        train_args["scheduler_args"] = {}
        train_args["scheduler_args"]["milestones"] = [40]
        train_args["scheduler_args"]["gamma"] = 0.1
    elif args.scheduler == "ReduceLROnPlateau":
        train_args["scheduler"] = "ReduceLROnPlateau"
        train_args["scheduler_args"] = {}
        train_args["scheduler_args"]["patience"] = 5
    else:
        train_args["scheduler"] = None
        train_args["scheduler_args"] = {}

    train_dl, val_dl, _, adj_mat = generate_multimodal_data(data_args, train_args, 
                                                         args.random_seed, device)

    model, opt, scheduler = load_model(data_args, model_args, train_args, 
                                       adj_mat, device)

    config = vars(args)
    config.update(model_args)
    config.update(data_args)
    config.update(train_args)

    os.environ["WANDB_CACHE_DIR"] = "/scratch/gobi1/VECTOR_USERNAME"

    run = wandb.init(
        project='straf-multimode-grid',
        config=config
    )

    best_model = train_loop(model, opt, scheduler, train_dl, val_dl, train_args)

    model_dir = pathlib.Path("./wandb")
    model_dir.mkdir(exist_ok=True)

    timestamp = str(datetime.now())[:19]
    model_path = "{}/{}.pt".format(model_dir, timestamp)
    torch.save(best_model, model_path)

    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(model_path)

    run.log_artifact(artifact)
    run.finish()

    os.remove(model_path)
