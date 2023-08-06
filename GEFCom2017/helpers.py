import datetime
import numpy as np

import torch
import torch.distributed as dist

import wandb

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

                print("Epoch:", epoch, flush=True)
                if scheduler is not None:
                    print("Current LR:", scheduler._last_lr, flush=True)
                print("Train NLL:", epoch_train_loss, "Val NLL:", epoch_val_loss, flush=True)

            if best_val is None or epoch_val_loss < best_val:
                best_val = epoch_val_loss
                best_model = model.state_dict()
                counter = 0
            else:
                counter += 1
                if counter > train_args["patience"]:
                    return best_model

    return best_model


def train_epoch(flow, opt, dl):
    losses = []
    for batch in dl:
        opt.zero_grad()
        z, jac = flow(batch)
        loss = flow.module.loss(z, jac)
        loss_val = loss.item()
        print(loss_val, flush=True)
        loss.backward()
        opt.step()

        losses.append(loss_val)
    return losses


def train_loop_dist(gpu, n_gpus, run, model, opt, scheduler, train_dl,
                    val_dl, train_args):
    train_hist = []
    val_hist = []

    best_model = None
    best_val = None
    counter = 0

    for epoch in range(1, train_args["max_epoch"]):
        dist.barrier()

        train_losses = train_epoch(model, opt, train_dl)

        with torch.no_grad():
            val_losses = []
            for batch in val_dl:
                z, jac = model(batch)
                val_loss = model.module.loss(z, jac).item()
                val_losses.append(val_loss)

            epoch_train_loss = np.mean(train_losses)
            epoch_val_loss = np.mean(val_losses)
            
            train_loss = torch.Tensor([epoch_train_loss]).to(gpu)
            val_loss = torch.Tensor([epoch_val_loss]).to(gpu)

            dist.all_reduce(train_loss)
            dist.all_reduce(val_loss)

            all_train_loss = train_loss.item() / n_gpus
            all_val_loss = val_loss.item() / n_gpus

            if scheduler is not None:
                if train_args["scheduler"] == "ReduceLROnPlateau":
                    scheduler.step(all_val_loss)
                else:
                    scheduler.step()

            dist.barrier()
    
            if gpu == 0:
                if train_args["use_wandb"]:
                    run.log({"train_loss": all_train_loss,
                               "val_loss": all_val_loss})

                if train_args["verbose"]:
                    train_hist.append(all_train_loss)
                    val_hist.append(all_val_loss)

                    print("Epoch:", epoch, flush=True)
                    if scheduler is not None:
                        print("Current LR:", scheduler._last_lr, flush=True)
                    
                    info = "Train NLL: {}, Val NLL: {}"
                    print(info.format(all_train_loss, all_val_loss), flush=True)

            dist.barrier()

            if best_val is None or all_val_loss < best_val:
                best_val = all_val_loss
                best_model = model.state_dict()
                counter = 0
            else:
                counter += 1
                if counter > train_args["patience"]:
                    print(best_val, flush=True)
                    save_model(best_model, run, "./wandb")
                    return best_model

    print(best_val, flush=True)
    save_model(best_model, run, "./wandb")

    dist.destroy_process_group()
    return best_model


def save_model(model_state, run, path):
    timestamp = str(datetime.datetime.now())[:19]
    out_tmp = "{}/{}.pt"
    model_path = out_tmp.format(path, timestamp)
    torch.save(model_state, model_path)

    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(model_path)

    run.log_artifact(artifact)
