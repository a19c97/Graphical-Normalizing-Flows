#!/bin/bash

#SBATCH -n 1
#SBATCH -p t4v2
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64GB
#SBATCH --output=./logs/%j.out
#SBATCH --error=./logs/%j.err
#SBATCH --gres=gpu:8
#SBATCH --qos=normal

module load cuda-11.7
cd ..

export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"

python train_model.py --dist_url "tcp://127.0.0.1:$MASTER_PORT" \
    --full_ar $1 --lr $2 --cond_hidden $3 --n_steps $4 --config_name "gefcom" 
