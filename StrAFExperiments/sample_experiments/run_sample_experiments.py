import argparse
from datetime import datetime
import pathlib
import yaml

import torch
import wandb

from sample_exp_utils import generate_multimodal_data, load_model, train_loop

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=2541)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train_config", type=str, required=True)
    parser.add_argument("--is_cnf", type="eval", default=False)

    args = parser.parse_args()
    with open("../data_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        data_args = config[args.dataset]

    with open("../model_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        model_args = config[args.model_name]

    with open("./train_args.yaml", "r") as f:
        config = yaml.safe_load(f)
        train_args = config[args.train_config]

    train_dl, val_dl, _, adj_mat = generate_multimodal_data(data_args, train_args, 
                                                         args.random_seed, device)

    model, opt, scheduler = load_model(data_args, model_args, train_args, 
                                       adj_mat, device)

    config = vars(args)
    config.update(model_args)
    config.update(data_args)
    config.update(train_args)

    run = wandb.init(
        project='straf-multimode-sample',
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
