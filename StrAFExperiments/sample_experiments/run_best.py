"""
Takes hyperparameters from each model type and performs several random 
initializations.
"""
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

sys.path.append("..")
sys.path.remove("../..")
sys.path.append("../ffjord")
from helpers import build_ffjord_model, train_loop_cnf

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_best_run(runs, tag):
    for run in runs:
        if tag in run.tags:
            return run


parser = argparse.ArgumentParser()
parser.add_argument("--model_tag", type=str, required=True)
parser.add_argument("--run_seed", type=int, required=True)
parser.add_argument("--is_cnf", type=eval, default=False)
args = parser.parse_args()

if __name__ == "__main__":
    api = wandb.Api()
    runs = api.runs("straf-multimode-grid")
    best_run = get_best_run(runs, args.model_tag)
    config = best_run.config

    args = parser.parse_args()
    with open("../data_config.yaml", "r") as f:
        d_config = yaml.safe_load(f)
        data_args = d_config[config["dataset"]]

    if args.is_cnf:
        with open("../cnf_config.yaml", "r") as f:
            m_config = yaml.safe_load(f)
            model_args = m_config[config["model_name"]]

        train_args = {}
        train_args["scheduler"] = None
        train_args["scheduler_args"] = {}
        train_args["lr"] = config["lr"]
        train_args["batch_size"] = config["batch_size"]
        train_args["max_epoch"] = 200
        train_args["patience"] = 10
        train_args["use_wandb"] = True
        train_args["verbose"] = True
    else:
        with open("../model_config.yaml", "r") as f:
            m_config = yaml.safe_load(f)
            model_args = m_config[config["model_name"]]

        cond_net = [config["cond_width"]] * config["cond_depth"]
        model_args["conditioner_args"]["hidden"] = cond_net
        model_args["conditioner_args"]["out_size"] = config["h_size"]

        inet = [config["inet_width"]] * config["inet_depth"]
        model_args["normalizer_args"]["integrand_net"] = inet
        model_args["normalizer_args"]["cond_size"] = config["h_size"]
        model_args["nf_steps"] = config["nf_step"]

        with open("./train_args.yaml", "r") as f:
            t_config = yaml.safe_load(f)
            train_args = t_config[config["train_config"]]

        train_args["lr"] = config["lr"]

        scheduler_type = config["scheduler"]
        if scheduler_type == "MultiStep":
            train_args["scheduler"] = "MultiStep"
            train_args["scheduler_args"] = {}
            train_args["scheduler_args"]["milestones"] = [40]
            train_args["scheduler_args"]["gamma"] = 0.1
        elif scheduler_type == "ReduceLROnPlateau":
            train_args["scheduler"] = "ReduceLROnPlateau"
            train_args["scheduler_args"] = {}
            train_args["scheduler_args"]["patience"] = 5
        else:
            train_args["scheduler"] = None
            train_args["scheduler_args"] = {}

    train_dl, val_dl, _, adj_mat = generate_multimodal_data(data_args, train_args,
                                                            config["random_seed"],
                                                            device)

    np.random.seed(args.run_seed)
    torch.manual_seed(args.run_seed)
    if args.is_cnf:
        model_args["lr"] = config["lr"]
        model_args["dims"] = config["dims"]
        model_args["num_blocks"] = config["num_blocks"]

        adj_mat = adj_mat + np.identity(adj_mat.shape[0])

        model, opt = build_ffjord_model(argparse.Namespace(**model_args),
                                        data_args["dim"], device, adj_mat)
    else:
        model, opt, scheduler = load_model(data_args, model_args, train_args,
                                           adj_mat, device)

    config.update(vars(args))
    config.update(model_args)
    config.update(data_args)
    config.update(train_args)

    os.environ["WANDB_CACHE_DIR"] = "/scratch/hdd001/home/ruiashi/wandb"

    run = wandb.init(
        project='straf-multimode-final',
        config=config
    )

    if args.is_cnf:
        best_model = train_loop_cnf(model, opt, None, train_dl, val_dl,
                                    train_args, device)
    else:
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
