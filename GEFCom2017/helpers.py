import numpy as np

import torch

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

                print("Epoch:", epoch)

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
