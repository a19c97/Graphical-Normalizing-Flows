import argparse
import sys
import os

import numpy as np
from sklearn import metrics
import yaml

import wandb
import torch


from sample_exp_utils import generate_multimodal_data, load_model

sys.path.append("../")
sys.path.remove("../..")
sys.path.append("../ffjord")
from helpers import build_ffjord_model

from ffjord.train_misc import standard_normal_logprob

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def eval_loss(model, dl):
    losses = []
    for batch in dl:
        z, jac = model(batch)
        loss = model.loss(z, jac)
        losses.append(loss.item())
    return np.mean(losses)

def eval_loss_cnf(model, dl):
    losses = []
    for batch in dl:
        zero = torch.zeros(batch.shape[0], 1).to(device)
        z, delta_logp = model(batch, zero)

        logpz = standard_normal_logprob(z).sum(1, keepdim=True)
        logpx = logpz - delta_logp

        loss = -torch.mean(logpx)
        losses.append(loss.item())
    return np.mean(losses)


def mmd_rbf(X, Y, gamma=0.1):
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def generate_samples(model, n_iter):
    samples = []
    for _ in range(n_iter):
        z_samp = np.random.normal(0, 1, size=(128, 15))
        z_samp = torch.Tensor(z_samp).to(device)

        x = model.invert(z_samp).detach().cpu().numpy()
        samples.append(x)
    samples = np.concatenate(samples)
    return samples


def generate_samples_cnf(model, n_iter):
    samples = []
    for _ in range(n_iter):
        z_samp = np.random.normal(0, 1, size=(128, 15))
        z_samp = torch.Tensor(z_samp).to(device)

        x = model(z_samp, reverse=True).detach().cpu().numpy()
        samples.append(x)
    samples = np.concatenate(samples)
    return samples


def load_from_wandb(run):
    config = run.config

    # TODO: Make best runs track model name
    if config["model_tag"] == "best_gnf":
        config["model_name"] = "gnf_umnn_grid"
    elif config["model_tag"] == "best_straf":
        config["model_name"] = "straf_umnn_grid"
    elif config["model_tag"] == "best_gnf_f":
        config["model_name"] = "gnf_f_umnn_grid"
    elif config["model_tag"] == "best_gnf_s":
        config["model_name"] = "gnf_s_umnn_grid"
    elif config["model_tag"] == "best_straf_s":
        config["model_name"] = "straf_s_umnn_grid"
    elif config["model_tag"] == "best_ffjord":
        config["model_name"] = "ffjord_baseline"
    elif config["model_tag"] == "best_strode":
        config["model_name"] = "ffjord_strode"
    elif config["model_tag"] == "best_daphne":
        config["model_name"] = "ffjord_daphne"

    print(config["model_tag"], flush=True)

    with open("../data_config.yaml", "r") as f:
        d_config = yaml.safe_load(f)
        # TODO: Also persist dataset
        data_args = d_config["multimodal_s"]

    is_cnf = config["model_name"][:6] == "ffjord"

    if is_cnf:
        with open("../cnf_config.yaml", "r") as f:
            m_config = yaml.safe_load(f)
            model_args = m_config[config["model_name"]]

        train_args = {}
        train_args["scheduler"] = None
        train_args["scheduler_args"] = {}
        train_args["lr"] = config["lr"]
        train_args["batch_size"] = config["batch_size"]
        train_args["max_epoch"] = 125
        train_args["patience"] = 10
        train_args["use_wandb"] = True
        train_args["verbose"] = True

    else:
        with open("../model_config.yaml", "r") as f:
            m_config = yaml.safe_load(f)
            model_args = m_config[config["model_name"]]

        model_args["conditioner_args"]["hidden"] = config["conditioner_args"]["hidden"]
        model_args["conditioner_args"]["out_size"] = config["conditioner_args"]["out_size"]

        model_args["normalizer_args"]["integrand_net"] = config["normalizer_args"]["integrand_net"]
        model_args["normalizer_args"]["cond_size"] = config["normalizer_args"]["cond_size"]
        model_args["nf_steps"] = config["nf_steps"]

        with open("./train_args.yaml", "r") as f:
            t_config = yaml.safe_load(f)
            train_args = t_config["multimodal_grid"]

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

    # TODO add random seed
    _, _, test_dl, adj_mat = generate_multimodal_data(data_args, train_args,
                                                      2547,
                                                      device)

    if is_cnf:        
        model_args["lr"] = config["lr"]
        model_args["dims"] = config["dims"]
        model_args["num_blocks"] = config["num_blocks"]

        adj_mat = adj_mat + np.identity(adj_mat.shape[0])

        model, _ = build_ffjord_model(argparse.Namespace(**model_args),
                                        data_args["dim"], device, adj_mat)
    else:
        model, _, _ = load_model(data_args, model_args, train_args,
                                adj_mat, device)
    
    weights = [m for m in run.logged_artifacts() if m.type == "model"][0]
    path = weights.download()
    model_path = os.listdir(path)[0]
    model_weights = torch.load(path + "/" + model_path)
    model.load_state_dict(model_weights)
    return model, test_dl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_tag", type=str, required=True)
    args = parser.parse_args()

    api = wandb.Api()
    runs = api.runs("straf-multimode-final")

    results = {"test_NLL": [], "sample_MMD": []}
    best_samples = None
    min_mmd = None

    for run in runs:
        config = run.config
        model_tag = config["model_tag"]

        if model_tag != args.model_tag:
            continue

        model, test_dl = load_from_wandb(run)

        is_cnf = args.model_tag in ["best_ffjord", "best_strode", "best_daphne"]

        if is_cnf:
            test_loss = eval_loss_cnf(model, test_dl)
        else:
            test_loss = eval_loss(model, test_dl)

        results["test_NLL"].append(test_loss)

        if is_cnf:
            samples = generate_samples_cnf(model, 5)
        else:
            samples = generate_samples(model, 5)

        true_dist = []
        for batch in test_dl:
            true_dist.append(batch)
        true_dist = torch.cat(true_dist)
        true_dist = true_dist.cpu().numpy()

        mmd = mmd_rbf(true_dist, samples)
        if min_mmd is None or mmd < min_mmd:
            min_mmd = mmd
            best_samples = samples

        results["sample_MMD"].append(mmd)

    for k, v in results.items():
        print(k, np.mean(v))
        print(k, np.std(v) / np.sqrt(len(v)))

    torch.save(results, "./output/mm_{}.pt".format(args.model_tag))
    torch.save(best_samples, "./output/mm_{}_samps.pt".format(args.model_tag))
