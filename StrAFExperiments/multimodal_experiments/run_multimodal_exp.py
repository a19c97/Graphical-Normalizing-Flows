import argparse
from datetime import datetime
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml

import wandb

from sklearn.model_selection import train_test_split

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from data_generators import (
    generate_adj_mat,
    generate_weight_mat,
    normalize_data,
    generate_multimodal_dataset
)

sys.path.append("../")

from models.NormalizingFlowFactories import buildFixedFCNormalizingFlow
from models.Conditionners import DAGConditioner, StrAFConditioner
from models.Normalizers import AffineNormalizer, MonotonicNormalizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


cond_map = {
    "GNF": DAGConditioner,
    "StrAF": StrAFConditioner
}

norm_map = {
    "Affine": AffineNormalizer,
    "Monotonic": MonotonicNormalizer
}


def get_cond_bias_param(model):
    bias_param = []
    for step in model.steps:
        bias = step.conditioner.masked_autoregressive_net.net[-1].bias
        bias_param.append(bias)
    return bias_param


def get_split_opt(model, lrs):
    bias = get_cond_bias_param(model)
    params = list(model.parameters())
    params = list(set(params) - set(bias))

    param_groups = [
        {'params': params, 'lr': lrs[0]},
        {'params': bias, 'lr': lrs[1]}]

    opt = torch.optim.Adam(param_groups)
    return opt


def train_loop(model, opt, scheduler, train_dl, val_dl, max_epoch):
    min_val = None
    patience = 4
    counter = 0

    for epoch in range(1, max_epoch):
        train_losses = []
        for batch in train_dl:
            opt.zero_grad()
            z, jac = model(batch)
            loss = model.loss(z, jac)
            train_loss = loss.item()

            loss.backward()
            opt.step()

            train_losses.append(train_loss)

        with torch.no_grad():
            val_losses = []
            for batch in val_dl:
                z, jac = model(batch)
                val_loss = model.loss(z, jac).item()
                val_losses.append(val_loss)

            epoch_val_loss = np.mean(val_losses)
            epoch_train_loss = np.mean(train_losses)

            if scheduler is not None:
                scheduler.step(epoch_val_loss)

            if min_val is None:
                min_val = epoch_val_loss
            elif epoch_val_loss < min_val:
                min_val = epoch_val_loss
                counter = 0
            else:
                counter += 1

            wandb.log({"train_loss": epoch_train_loss,
                       "val_loss": epoch_val_loss})

            if counter > patience:
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--random_seed", type=int, default=2541)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--n_samp", type=int, required=True)

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--nf_steps", type=int, required=True)
    parser.add_argument("--cond_width", type=int, required=True)
    parser.add_argument("--cond_depth", type=int, required=True)
    parser.add_argument("--full_ar", action="store_true", default=False)

    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--split_bias_lr", action="store_true", default=False)
    parser.add_argument("--permute_latent", action="store_true", default=False)
    parser.add_argument("--reduce_on_plateau", action="store_true", default=False)
    
    args = parser.parse_args()

    with open("data_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        data_args = config[args.dataset]

    # Generate data
    np.random.seed(args.random_seed)

    adj_mat = generate_adj_mat(data_args["dim"], data_args["adj_thres"])
    w_mat = generate_weight_mat(adj_mat, data_args["w_range"])
    data, _ = generate_multimodal_dataset(w_mat, args.n_samp, data_args["n_modes"],
                                          data_args["mean_range"], data_args["std_range"])
    normalized_data = normalize_data(data).T

    # Split dataset
    ratios = (0.6, 0.2, 0.2)
    temp_set, train_set = train_test_split(normalized_data, test_size=ratios[0])
    val_set, test_set = train_test_split(temp_set, test_size=ratios[1]/(ratios[1] + ratios[2]))

    train_set = torch.Tensor(train_set).to(device)
    val_set = torch.Tensor(val_set).to(device)

    train_dl = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_set, batch_size=args.batch_size)

    # Load model
    with open("model_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        model_args = config[args.model_name]

    conditioner = cond_map[model_args["conditioner_type"]]
    cond_net = [args.cond_width] * args.cond_depth + [2]

    cond_args = model_args["conditioner_args"]
    cond_args["in_size"] = data_args["dim"]
    cond_args["hidden"] = cond_net[:-1]
    cond_args["device"] = device

    if args.full_ar:
        far_mat = np.ones((data_args["dim"], data_args["dim"]))
        far_mat = np.tril(far_mat, -1)
        far_mat = torch.Tensor(far_mat).to(device)
        cond_args["A_prior"] = far_mat
    else:
        adj_mat = torch.Tensor(adj_mat).to(device)
        cond_args["A_prior"] = adj_mat

    normalizer = norm_map[model_args["normalizer_type"]]
    norm_args = model_args["normalizer_args"]

    model = buildFixedFCNormalizingFlow(args.nf_steps, conditioner, cond_args,
                                        normalizer, norm_args, args.permute_latent)
    model = model.to(device)

    # Setup optimizer
    if args.split_bias_lr:
        opt = get_split_opt(model, (args.lr, 5e-2))
    else:
        opt = Adam(model.parameters(), args.lr)

    if args.reduce_on_plateau:
        scheduler = ReduceLROnPlateau(opt)
    else:
        scheduler = None
        
    config = vars(args)
    config.update(data_args)
    config.update(model_args)

    run = wandb.init(
        project='straf-multimode',
        config=config
    )

    train_loop(model, opt, scheduler, train_dl, val_dl, args.max_epoch)
    
    model_dir = pathlib.Path("./wandb")
    model_dir.mkdir(exist_ok=True)

    timestamp = str(datetime.now())[:19]
    model_path = "{}/{}.pt".format(model_dir, timestamp)
    torch.save(model.state_dict(), model_path)

    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(model_path)

    run.log_artifact(artifact)
    run.finish()
