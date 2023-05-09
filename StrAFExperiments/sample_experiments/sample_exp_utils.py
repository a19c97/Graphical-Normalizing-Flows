import sys
import yaml

import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import normaltest
from sklearn.model_selection import train_test_split

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

import wandb

sys.path.append("../")

from data_generators import (
    generate_adj_mat,
    generate_weight_mat,
    normalize_data,
    generate_multimodal_dataset
)

sys.path.append("../..")

from plot_utils import viz_data

from models.NormalizingFlowFactories import buildFixedFCNormalizingFlow
from models.Conditionners import DAGConditioner, StrAFConditioner
from models.Normalizers import AffineNormalizer, MonotonicNormalizer


COND_MAP = {
    "GNF": DAGConditioner,
    "StrAF": StrAFConditioner
}

NORM_MAP = {
    "Affine": AffineNormalizer,
    "Monotonic": MonotonicNormalizer
}

def plot_hist(z):
    f, ax = plt.subplots(1, z.shape[1], figsize=(20, 3))

    def gaussian(x, mu, sig):
        const = 1 / (sig * np.sqrt(2 * np.pi))
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) * const

    normality = []
    for i in range(z.shape[1]):
        x = np.linspace(-4, 4, 100)
        ax[i].plot(x, gaussian(x, 0, 1))
        z_np = z[:, i].cpu().numpy()
        ax[i].hist(z_np, density=True)
        normality.append(round(normaltest(z_np).pvalue, 3))
    print(normality)
    plt.show()


def load_model(data_args, model_args, train_args, adj_mat, device):
    conditioner = COND_MAP[model_args["conditioner_type"]]

    cond_args = model_args["conditioner_args"]
    cond_args["in_size"] = data_args["dim"]
    cond_args["device"] = device

    if model_args["full_ar"]:
        far_mat = np.ones((data_args["dim"], data_args["dim"]))
        far_mat = np.tril(far_mat, -1)
        far_mat = torch.Tensor(far_mat).to(device)
        cond_args["A_prior"] = far_mat
    else:
        if not torch.is_tensor(adj_mat):
            adj_mat = torch.Tensor(adj_mat).to(device)
        cond_args["A_prior"] = adj_mat

    normalizer = NORM_MAP[model_args["normalizer_type"]]
    norm_args = model_args["normalizer_args"]

    model = buildFixedFCNormalizingFlow(model_args["nf_steps"], conditioner, cond_args,
                                        normalizer, norm_args, model_args["permute_latent"])
    model = model.to(device)
    opt = Adam(model.parameters(), train_args["lr"])

    if train_args["scheduler"] == "MultiStep":
        scheduler = MultiStepLR(opt, **train_args["scheduler_args"])
    elif train_args["scheduler"] == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(opt, **train_args["scheduler_args"])
    else:
        scheduler = None

    return model, opt, scheduler


def generate_multimodal_data(data_args, train_args, seed, device):
    np.random.seed(seed)

    adj_mat = generate_adj_mat(data_args["dim"], data_args["adj_thres"])
    w_mat = generate_weight_mat(adj_mat, data_args["w_range"])
    data, _ = generate_multimodal_dataset(w_mat, data_args["n_samp"], data_args["n_modes"],
                                          data_args["mean_range"], data_args["std_range"])
    normalized_data = normalize_data(data).T

    # Split dataset
    ratios = data_args["split_ratio"]
    temp_set, train_set = train_test_split(normalized_data, test_size=ratios[0])
    val_set, test_set = train_test_split(temp_set, test_size=ratios[1]/(ratios[1] + ratios[2]))

    train_set = torch.Tensor(train_set).to(device)
    val_set = torch.Tensor(val_set).to(device)
    test_set = torch.Tensor(test_set).to(device)

    train_dl = DataLoader(train_set, batch_size=train_args["batch_size"], shuffle=True)
    val_dl = DataLoader(val_set, batch_size=train_args["batch_size"])
    test_dl = DataLoader(test_set, batch_size=train_args["batch_size"])

    return train_dl, val_dl, test_dl, adj_mat


def train_loop(model, opt, scheduler, train_dl, val_dl, train_args):
    train_hist = []
    val_hist = []

    best_model = None
    best_val = None
    counter = 0

    for epoch in range(1, train_args["max_epoch"]):
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

            epoch_train_loss = np.mean(train_losses)
            epoch_val_loss = np.mean(val_losses)

            if scheduler is not None:
                if train_args["scheduler"] == "ReduceLROnPlateau":
                    scheduler.step(epoch_val_loss)
                else:
                    scheduler.step()

            if train_args["use_wandb"]:
                wandb.log({"train_loss": epoch_train_loss,
                           "val_loss": epoch_val_loss})

            if train_args["verbose"]:
                train_hist.append(epoch_train_loss)
                val_hist.append(epoch_val_loss)

                print("Epoch:", epoch)

                print("Current LR:", scheduler._last_lr)
                print("Train NLL:", epoch_train_loss, "Val NLL:", epoch_val_loss)

            if train_args["visualize"]:
                f, ax = plt.subplots(1, 2, figsize=(8, 3))

                ax[0].plot(train_hist)
                ax[1].plot(val_hist)
                plt.show()

                plot_hist(z)

                if torch.is_tensor(z):
                    z = z.detach().cpu().numpy()
                viz_data(z)

            if best_val is None or epoch_val_loss < best_val:
                best_val = epoch_val_loss
                best_model = model.state_dict()
                counter = 0
            else:
                counter += 1
                if counter > train_args["patience"]:
                    return best_model

    return best_model
