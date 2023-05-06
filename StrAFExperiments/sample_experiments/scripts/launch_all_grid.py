import os
import wandb

model_name = ["gnf_umnn_grid", "straf_umnn_grid", "gnf_f_umnn_grid"]
cond_width = [50, 500]
cond_depth = [3, 4]
nf_step = [5, 10]
inet_width = [250, 500]
inet_depth = [4, 6]
h_size = [25, 50]
lrs = [1e-3, 1e-4]
scheduler = ["ReduceLROnPlateau", "MultiStep", "None"]

api = wandb.Api()
runs = api.runs("straf-multimode-grid")

completed_runs = set()

for run in runs:
    config = run.config
    state = run.state
    if state != "finished":
        continue
    run_repr = "{}{}{}{}{}{}{}{}{}".format(
        config["model_name"],
        config["cond_width"],
        config["cond_depth"],
        config["nf_step"],
        config["inet_width"],
        config["inet_depth"], 
        config["h_size"],
        config["lr"],
        config["scheduler"])

    completed_runs.add(run_repr)

counter = 0

for mn in model_name:
    for cw in cond_width:
        for cnd in cond_depth:
            for ns in nf_step:
                for iw in inet_width:
                    for ind in inet_depth:
                        for h in h_size:
                            for lr in lrs:
                                for s in scheduler:
                                    run_repr = "{}{}{}{}{}{}{}{}{}".format(mn, cw, cnd, ns, iw, ind, h, lr, s)

                                    if run_repr not in completed_runs:
                                        command = "sbatch launch_grid.sh {} {} {} {} {} {} {} {} {}".format(
                                            mn, cw, cnd, ns, iw, ind, h, lr, s)
                                        os.system(command)
