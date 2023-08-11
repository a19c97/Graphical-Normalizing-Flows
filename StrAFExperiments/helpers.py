import sys

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import wandb

sys.path.append("../")
import ffjord.lib.layers as layers
from ffjord.train_misc import set_cnf_options, build_model_tabular
from ffjord.train_misc import standard_normal_logprob

sys.path.append("../")
from models.Conditionners.StrAFConditioner import StrODENet


class ConcatSquashLinearSparse(nn.Module):
    def __init__(self, dim_in, dim_out, adjacency, device):
        super(ConcatSquashLinearSparse, self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out

        self._adjacency = adjacency
        _weight_mask = torch.zeros([dim_out, dim_in])
        _weight_mask[:adjacency.shape[0], :adjacency.shape[1]] = torch.Tensor(adjacency)
        self._weight_mask = _weight_mask.to(device)

        lin = nn.Linear(dim_in, dim_out)
        self._weights = lin.weight
        self._bias = lin.bias

        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1, dim_out)

    def forward(self, t, x):
        w = torch.mul(self._weight_mask, self._weights)
        res = torch.addmm(self._bias, x, w.transpose(0,1))

        return res * torch.sigmoid(self._hyper_gate(t.view(1, 1))) \
            + self._hyper_bias(t.view(1, 1))


class SparseODENet(nn.Module):
    def __init__(self, dims, full_adjacency, device, num_layers=4):
        super(SparseODENet, self).__init__()
        self.num_squeeze=0
        layers = [ConcatSquashLinearSparse(dims+1, dims, full_adjacency, device)]
        layers = layers + [ConcatSquashLinearSparse(dims, dims, full_adjacency, device) for _ in range(num_layers-1)]
        activation_fns = [nn.Tanh() for _ in range(num_layers)]
        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns[:-1])

    def forward(self, t, x):
        batch_dim = x.shape[0]
        dx = torch.cat([x, t * torch.ones([batch_dim, 1]).to(x.device)], dim=1)
        for l, layer in enumerate(self.layers):
            # if not last layer, use nonlinearity
            if l < len(self.layers) - 1:
                acti = layer(t, dx)
                if l == 0:
                    dx = self.activation_fns[l](acti)
                else:
                    dx = self.activation_fns[l](acti) + dx
            else:
                dx = layer(t, dx)
        return dx


def build_model_daphne(args, input_dim, adj_mat, device):
    hidden_dims = list(map(int, args.dims.split("-")))
    ode_net = SparseODENet(input_dim, adj_mat, device, len(hidden_dims))
    ode_net = ode_net.to(device)
    
    def build_cnf():
        odenet = ode_net
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


def build_model_strcnf(args, input_dim, adj_mat, device):
    hidden_dims = list(map(int, args.dims.split("-")))

    def build_cnf(device):
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

    chain = [build_cnf(device) for _ in range(args.num_blocks)]
    model = layers.SequentialFlow(chain)

    set_cnf_options(args, model)

    return model


def build_ffjord_model(args, input_dim, device, adj_mat=None):
    if args.layer_type == "strode":
        model = build_model_strcnf(args, input_dim, adj_mat, device).to(device)
    elif args.layer_type == "daphne":
        model = build_model_daphne(args, input_dim, adj_mat, device).to(device)
    else:
        model = build_model_tabular(args, input_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    return model, optimizer


def train_loop_cnf(model, opt, scheduler, train_dl, val_dl, train_args, device):
    train_hist = []
    val_hist = []

    best_model = None
    best_val = None
    counter = 0

    for epoch in range(1, train_args["max_epoch"]):
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

                if scheduler is not None:
                    print("Current LR:", scheduler._last_lr)

                print("Train NLL:", epoch_train_loss, "Val NLL:", epoch_val_loss)

            if best_val is None or epoch_val_loss < best_val:
                best_val = epoch_val_loss
                best_model = model.state_dict()
                counter = 0
            else:
                counter += 1
                if counter > train_args["patience"]:
                    return best_model

    return best_model
