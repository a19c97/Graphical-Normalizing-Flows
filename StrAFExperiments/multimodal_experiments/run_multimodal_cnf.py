import argparse
import pathlib
import sys

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import yaml

import wandb

from sklearn.model_selection import train_test_split

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append("../")
from data_generators import (
    generate_adj_mat,
    generate_weight_mat,
    normalize_data,
    generate_multimodal_dataset
)

sys.path.append("../ffjord")
from ffjord.train_misc import build_model_tabular, set_cnf_options
from ffjord.train_misc import standard_normal_logprob
import ffjord.lib.layers as layers

sys.path.append("../../")
from models.Conditionners.StrAFConditioner import StrODENet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--random_seed", type=int, default=2547)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--model_name", type=str, required=True)

parser.add_argument("--max_epoch", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=int, default=5e-3)
parser.add_argument("--hidden_width", type=int, required=True)
parser.add_argument("--hidden_depth", type=int, required=True)
parser.add_argument("--nf_step", type=int, required=True)

args = parser.parse_args()


def build_model_strcnf(args, input_dim, adj_mat):

    hidden_dims = list(map(int, args.dims.split("-")))

    def build_cnf():
        odenet = StrODENet(input_dim, hidden_dims, adj_mat, device)
        odefunc = layers.ODEfunc(
            diffeq=odenet,
            divergence_fn=args.divergence_fn
        )
        cnf = layers.CNF(
            odefunc=odefunc,
            T=args.time_length,
            train_T=args.train_T,
            solver=args.solver,
        )
        return cnf

    chain = [build_cnf() for _ in range(args.num_blocks)]
    model = layers.SequentialFlow(chain)

    set_cnf_options(args, model)

    return model


def build_ffjord_model(args, input_dim, adj_mat=None):
    if args.layer_type == "strode":
        model = build_model_strcnf(args, input_dim, adj_mat).to(device)
    else:
        model = build_model_tabular(args, input_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    return model, optimizer


def get_transforms(model):
    def sample_fn(z, logpz=None):
        if logpz is not None:
            return model(z, logpz, reverse=True)
        else:
            return model(z, reverse=True)

    def density_fn(x, logpx=None):
        if logpx is not None:
            return model(x, logpx, reverse=False)
        else:
            return model(x, reverse=False)

    return sample_fn, density_fn


def train_loop(model, opt, train_dl, val_dl, max_epoch, viz=True):
    min_val = None
    patience = 10
    counter = 0

    for epoch in range(1, max_epoch):
        train_losses = []
        for batch in train_dl:
            opt.zero_grad()

            zero = torch.zeros(batch.shape[0], 1).to(device)
            z, delta_logp = model(batch, zero)

            logpz = standard_normal_logprob(z).sum(1, keepdim=True)
            logpx = logpz - delta_logp

            loss = -torch.mean(logpx)
            train_loss = loss.item()

            loss.backward()
            opt.step()
            train_losses.append(train_loss)

        with torch.no_grad():
            val_losses = []
            for batch in val_dl:
                zero = torch.zeros(batch.shape[0], 1).to(device)
                z, delta_logp = model(batch, zero)

                logpz = standard_normal_logprob(z).sum(1, keepdim=True)
                logpx = logpz - delta_logp

                val_loss = -torch.mean(logpx)
                val_losses.append(val_loss.item())

            epoch_val_loss = np.mean(val_losses)
            epoch_train_loss = np.mean(train_losses)

            if min_val is None:
                min_val = epoch_val_loss
            elif epoch_val_loss < min_val:
                min_val = epoch_val_loss
                counter = 0
            else:
                counter += 1

            print(epoch_train_loss, epoch_val_loss)
            wandb.log({"train_loss": epoch_train_loss,
                       "val_loss": epoch_val_loss})

            if viz:
                sample_fn, density_fn = get_transforms(model)
                
                z = density_fn(batch).cpu().numpy()

                f, ax = plt.subplots(1, 15, figsize=(20, 3))
                for i in range(15):
                    ax[i].hist(z[:, i], density=True, bins=20)
                    x = np.linspace(-3, 3, 100)
                    ax[i].plot(x, stats.norm.pdf(x, 0, 1))
                plt.savefig("hist.png")
                plt.close()

                z = torch.randn(500, 15).to(device)
                x = sample_fn(z).cpu().numpy()
                
                f, ax = plt.subplots(1, 4, figsize=(20, 3))
                for i in range(4):
                    ax[i].scatter(x[:, i], x[:, i+1])
                plt.savefig("sample.png")
                plt.close()

            if counter > patience:
                return min_val


if __name__ == "__main__":
    with open("../data_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        data_args = config[args.dataset]

    # Generate data
    np.random.seed(args.random_seed)

    adj_mat = generate_adj_mat(data_args["dim"], data_args["adj_thres"])
    w_mat = generate_weight_mat(adj_mat, data_args["w_range"])
    data, _ = generate_multimodal_dataset(w_mat, data_args["n_samp"],
                                          data_args["n_modes"],
                                          data_args["mean_range"],
                                          data_args["std_range"])
    normalized_data = normalize_data(data).T

    adj_mat = adj_mat + np.identity(adj_mat.shape[0])

    # Split dataset
    ratios = (0.6, 0.2, 0.2)
    temp_set, train_set = train_test_split(normalized_data, test_size=ratios[0])
    test_size = ratios[1]/(ratios[1] + ratios[2])
    val_set, test_set = train_test_split(temp_set, test_size=test_size)

    train_set = torch.Tensor(train_set).to(device)
    val_set = torch.Tensor(val_set).to(device)

    train_dl = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_set, batch_size=args.batch_size)

    # Initialize model
    with open("../cnf_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        model_args = config[args.model_name]

    model_args["lr"] = args.lr
    model_args["dims"] = "-".join([str(args.hidden_width)] * args.hidden_depth)
    model_args["num_blocks"] = args.nf_step
    model, optimizer = build_ffjord_model(argparse.Namespace(**model_args),
                                          data_args["dim"], adj_mat)

    config = vars(args)
    config.update(data_args)
    config.update(model_args)
    print(config, flush=True)

    run = wandb.init(
        project='straf-multimode-grid',
        config=config
    )

    min_val = train_loop(model, optimizer, train_dl, val_dl, args.max_epoch)

    model_dir = pathlib.Path("./wandb")
    model_dir.mkdir(exist_ok=True)

    timestamp = str(datetime.now())[:19]
    model_path = "{}/{}.pt".format(model_dir, timestamp)
    torch.save(model.state_dict(), model_path)

    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(model_path)

    run.log_artifact(artifact)
    run.finish()
